# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2023
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements the composite adversarial attack by sequentially perturbing different components of the inputs.
It uses order scheduling to search for the attack sequence and uses the iterative gradient sign method to optimize the
perturbations in semantic space and Lp-ball (see `FastGradientMethod` and `BasicIterativeMethod`).

| Paper link: https://arxiv.org/abs/2202.04235
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import logging

from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np
from tqdm.auto import tqdm

from art.attacks.attack import EvasionAttack
from art.config import ART_NUMPY_DTYPE
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import (
    compute_success,
    check_and_transform_label_format,
    get_labels_np_array
)

if TYPE_CHECKING:
    # pylint: disable=C0412
    import torch
    import torch.nn.functional as F
    from art.estimators.classification.pytorch import PyTorchClassifier
    from math import pi

logger = logging.getLogger(__name__)


class CompositeAdversarialAttackPyTorch(EvasionAttack):
    """
    Implementation of the composite adversarial attack on image classifiers in PyTorch. The attack is constructed by adversarially
    perturbing the hue component of the inputs. It uses the iterative gradient sign method to optimise the semantic
    perturbations (see `FastGradientMethod` and `BasicIterativeMethod`). This implementation extends the original
    optimisation method to other norms as well.

    Note that this attack is intended for only PyTorch image classifiers with RGB images in the range [0, 1] as inputs.

    | Paper link: https://arxiv.org/abs/2202.04235
    """

    attack_params = EvasionAttack.attack_params + [
        "enabled_attack",
        "hue_epsilon",
        "sat_epsilon",
        "rot_epsilon",
        "bri_epsilon",
        "con_epsilon",
        "pgd_epsilon",
        "early_stop",
        "max_iter",
        "max_inner_iter",
        "schedule",
        "batch_size",
        "verbose",
    ]
    _estimator_requirements = (BaseEstimator, LossGradientsMixin, ClassifierMixin)  # type: ignore

    def __init__(
            self,
            classifier: "PyTorchClassifier",
            enabled_attack: Tuple = (0, 1, 2, 3, 4, 5),
            # Default: Full Attacks; 0: Hue, 1: Saturation, 2: Rotation, 3: Brightness, 4: Contrast, 5: PGD (L-infinity)
            hue_epsilon: Tuple = (-pi, pi),
            sat_epsilon: Tuple = (0.7, 1.3),
            rot_epsilon: Tuple = (-10, 10),
            bri_epsilon: Tuple = (-0.2, 0.2),
            con_epsilon: Tuple = (0.7, 1.3),
            pgd_epsilon: Tuple = (-8 / 255, 8 / 255),  # L-infinity
            early_stop: bool = True,
            max_iter: int = 5,
            max_inner_iter: int = 10,
            attack_order: str = "scheduled",
            batch_size: int = 1,
            verbose: bool = True,
    ) -> None:
        """
        Create an instance of the :class:`.CompositeAdversarialAttackPyTorch`.

        :param classifier: A trained PyTorch classifier.
        :param enabled_attack: The norm of the adversarial perturbation. Possible values: `"inf"`, `np.inf`, `1` or `2`.
        :param hue_epsilon: The boundary of the hue perturbation. The value is expected to be in the interval
                           `[-pi, pi]`. Perturbation of `0` means no shift and `-pi` and `pi` give a complete reversal
                           of the hue channel in the HSV colour space in the positive and negative directions,
                           respectively. See `kornia.enhance.adjust_hue` for more details.
        :param sat_epsilon: The boundary of the saturation perturbation. The value is expected to be in the interval
                           `[0, infinity)`. Perturbation of `0` gives a black and white image, `1` gives the original
                           image, while `2` enhances the saturation by a factor of 2. See
                           `kornia.geometry.transform.rotate` for more details.
        :param rot_epsilon: The boundary of the rotation perturbation (in degrees). Positive values mean
                            counter-clockwise rotation. See `kornia.geometry.transform.rotate` for more details.
        :param bri_epsilon: The boundary of the brightness perturbation. The value is expected to be in the interval
                           `[-1, 1]`. Perturbation of `0` means no shift, `-1` gives a complete black image, and `1`
                           gives a complete white image. See `kornia.enhance.adjust_brightness` for more details.
        :param con_epsilon: The boundary of the contrast perturbation. The value is expected to be in the interval
                           `[0, infinity)`. Perturbation of `0` gives a complete black image, `1` does not modify the
                           image, and any other value modifies the brightness by this factor. See
                           `kornia.enhance.adjust_contrast` for more details.
        :param pgd_epsilon: The maximum perturbation that the attacker can introduce in the L-infinity ball.
        :param early_stop: When True, the attack will stop if the perturbed example is classified incorrectly by the
                           classifier.
        :param max_iter: The maximum number of iterations for attack order optimization.
        :param max_inner_iter: The maximum number of iterations for each attack optimization.
        :param attack_order: Specify the scheduling type for composite adversarial attack. The value is expected to be
                             `fixed`, `random', or `scheduled`. `fixed` means the attack order is the same as specified
                             in `enabled_attack`. `random` means the attack order is randomly generated at each iteration.
                             `scheduled` means to enable the attack order optimization proposed in the paper. If only one
                              attack is enabled, `fixed` will be used.
        :param batch_size: The batch size to use during the generation of adversarial samples.
        :param verbose: Show progress bars.
        """
        import torch

        super().__init__(estimator=classifier)
        self.classifier = classifier
        self.model = classifier.model
        self.device = next(self.model.parameters()).device
        self.fixed_order = enabled_attack
        self.enabled_attack = tuple(sorted(enabled_attack))
        self.seq_num = len(enabled_attack)  # attack_num
        self.early_stop = early_stop
        self.linf_idx = self.enabled_attack.index(5) if 5 in self.enabled_attack else None
        self.eps_pool = torch.tensor(
            [hue_epsilon, sat_epsilon, rot_epsilon, bri_epsilon, con_epsilon, pgd_epsilon], device=self.device)
        self.attack_order = attack_order
        self.max_inner_iter = max_inner_iter
        self.max_iter = max_iter if self.attack_order == 'scheduled' else 1
        self.targeted = False
        self.batch_size = batch_size
        self.verbose = verbose
        self.attack_pool = (
            self.caa_hue, self.caa_saturation, self.caa_rotation, self.caa_brightness, self.caa_contrast, self.caa_linf)

        import kornia
        self.attack_pool_base = (
            kornia.enhance.adjust_hue, kornia.enhance.adjust_saturation, kornia.geometry.transform.rotate,
            kornia.enhance.adjust_brightness, kornia.enhance.adjust_contrast, self.get_linf_perturbation)
        self.attack_dict = tuple([self.attack_pool[i] for i in self.enabled_attack])
        self.step_size_pool = [2.5 * ((eps[1] - eps[0]) / 2) / self.max_inner_iter for eps in
                               self.eps_pool]  # 2.5 * ε-test / num_steps

        self._check_params()
        self._description = "Composite Adversarial Attack"
        self._is_scheduling = False
        self.adv_val_pool = self.eps_space = self.adv_val_space = self.curr_dsm = \
            self.curr_seq = self.is_attacked = self.is_not_attacked = None

    def _check_params(self) -> None:
        super()._check_params()
        if self.attack_order not in ('fixed', 'random', 'scheduled'):
            logger.info("attack_order: {}, should be either 'fixed', 'random', or 'scheduled'.".format(self.attack_order))
            raise ValueError

    def _set_targets(
            self,
            x: np.ndarray,
            y: Optional[np.ndarray],
            classifier_mixin: bool = True
    ) -> np.ndarray:
        """
        Check and set up targets.

        :param x: An array with all the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  `(nb_samples,)`. Only provide this parameter if you'd like to use true labels when crafting
                  adversarial samples. Otherwise, model predictions are used as labels to avoid the "label leaking"
                  effect (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :param classifier_mixin: Whether the estimator is of type `ClassifierMixin`.
        :return: The targets.
        """
        if classifier_mixin:
            if y is not None:
                y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes)

        if y is None:
            # Throw an error if the attack is targeted, but no targets are provided.
            if self.targeted:  # pragma: no cover
                raise ValueError("Target labels `y` need to be provided for a targeted attack.")

            # Use the model predictions as correct outputs.
            if classifier_mixin:
                targets = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))
            else:
                targets = self.estimator.predict(x, batch_size=self.batch_size)

        else:
            targets = y

        return targets

    def _setup_attack(self):
    import torch

        hue_space = torch.rand(self.batch_size, device=self.device) * (
                self.eps_pool[0][1] - self.eps_pool[0][0]) + self.eps_pool[0][0]
        sat_space = torch.rand(self.batch_size, device=self.device) * (
                self.eps_pool[1][1] - self.eps_pool[1][0]) + self.eps_pool[1][0]
        rot_space = torch.rand(self.batch_size, device=self.device) * (
                self.eps_pool[2][1] - self.eps_pool[2][0]) + self.eps_pool[2][0]
        bri_space = torch.rand(self.batch_size, device=self.device) * (
                self.eps_pool[3][1] - self.eps_pool[3][0]) + self.eps_pool[3][0]
        con_space = torch.rand(self.batch_size, device=self.device) * (
                self.eps_pool[4][1] - self.eps_pool[4][0]) + self.eps_pool[4][0]
        pgd_space = 0.001 * torch.randn([self.batch_size, 3, 32, 32], device=self.device)
        self.adv_val_pool = [hue_space, sat_space, rot_space, bri_space, con_space, pgd_space]

        self.eps_space = [self.eps_pool[i] for i in self.enabled_attack]
        self.adv_val_space = [self.adv_val_pool[i] for i in self.enabled_attack]

    def generate(
            self,
            x: np.ndarray,
            y: Optional[np.ndarray] = None,
            **kwargs
    ) -> np.ndarray:
    import torch

        targets = self._set_targets(x, y)
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(x.astype(ART_NUMPY_DTYPE)),
            torch.from_numpy(y.astype(ART_NUMPY_DTYPE)),
        )
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=self.batch_size, shuffle=False, drop_last=False
        )

        # Start to compute adversarial examples.
        x_adv = x.copy().astype(ART_NUMPY_DTYPE)

        # Compute perturbations with batching.
        for (batch_id, batch_all) in enumerate(
                tqdm(data_loader, desc=self._description, leave=False, disable=not self.verbose)
        ):
            self._batch_id = batch_id
            (batch_x, batch_targets, batch_mask) = batch_all[0], batch_all[1], None
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size

            x_adv[batch_index_1:batch_index_2] = self._generate_batch(
                x=batch_x,
                y=batch_targets,
                mask=batch_mask,
            )

        logger.info(
            "Success rate of attack: %.2f%%",
            100 * compute_success(self.estimator, x, targets, x_adv, self.targeted, batch_size=self.batch_size),
        )

        return x_adv

    def _generate_batch(
            self,
            x: "torch.Tensor",
            y: "torch.Tensor",
            mask: "torch.Tensor"
    ) -> np.ndarray:
        """
        Generate a batch of adversarial samples and return them in a NumPy array.

        :param x: Original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)`.
        :param mask: A 1D array of masks defining which samples to perturb. Shape needs to be `(nb_samples,)`.
                     Samples for which the mask is zero will not be adversarially perturbed.
        :return: Adversarial examples.
        """
        import torch


        self.batch_size = x.shape[0]
        self._setup_attack()
        self.is_attacked = torch.zeros(self.batch_size, device=self.device).bool()
        self.is_not_attacked = torch.ones(self.batch_size, device=self.device).bool()
        x, y = x.to(self.device), y.to(self.device)

        return self.caa_attack(x, y).cpu().detach().numpy()

    def _comp_pgd(self, data, labels, attack_idx, attack_parameter, ori_is_attacked):
    import torch

        adv_data = self.attack_pool_base[attack_idx](data, attack_parameter)
        for _ in range(self.max_inner_iter):
            outputs = self.model(adv_data)

            if not self._is_scheduling and self.early_stop:
                cur_pred = outputs.max(1, keepdim=True)[1].squeeze()
                self.is_attacked = torch.logical_or(ori_is_attacked,
                                                    cur_pred != labels.max(1, keepdim=True)[1].squeeze())

            with torch.enable_grad():
                cost = F.cross_entropy(outputs, labels)
            _grad = torch.autograd.grad(cost, attack_parameter)[0]
            if not self._is_scheduling:
                _grad[self.is_attacked] = 0
            attack_parameter = torch.clamp(attack_parameter + torch.sign(_grad) * self.step_size_pool[attack_idx],
                                           self.eps_pool[attack_idx][0],
                                           self.eps_pool[attack_idx][1]).detach().requires_grad_()
            adv_data = self.attack_pool_base[attack_idx](data, attack_parameter)

        return adv_data, attack_parameter

    def caa_hue(self, data, hue, labels):
        hue = hue.detach().clone()
        hue[self.is_attacked] = 0
        hue.requires_grad_()
        sur_data = data.detach().requires_grad_()

        return self._comp_pgd(data=sur_data, labels=labels, attack_idx=0, attack_parameter=hue,
                              ori_is_attacked=self.is_attacked.clone())

    def caa_saturation(self, data, saturation, labels):
        saturation = saturation.detach().clone()
        saturation[self.is_attacked] = 1
        saturation.requires_grad_()
        sur_data = data.detach().requires_grad_()

        return self._comp_pgd(data=sur_data, labels=labels, attack_idx=1, attack_parameter=saturation,
                              ori_is_attacked=self.is_attacked.clone())

    def caa_rotation(self, data, theta, labels):
        theta = theta.detach().clone()
        theta[self.is_attacked] = 0
        theta.requires_grad_()
        sur_data = data.detach().requires_grad_()

        return self._comp_pgd(data=sur_data, labels=labels, attack_idx=2, attack_parameter=theta,
                              ori_is_attacked=self.is_attacked.clone())

    def caa_brightness(self, data, brightness, labels):
        brightness = brightness.detach().clone()
        brightness[self.is_attacked] = 0
        brightness.requires_grad_()
        sur_data = data.detach().requires_grad_()

        return self._comp_pgd(data=sur_data, labels=labels, attack_idx=3, attack_parameter=brightness,
                              ori_is_attacked=self.is_attacked.clone())

    def caa_contrast(self, data, contrast, labels):
        contrast = contrast.detach().clone()
        contrast[self.is_attacked] = 1
        contrast.requires_grad_()
        sur_data = data.detach().requires_grad_()

        return self._comp_pgd(data=sur_data, labels=labels, attack_idx=4, attack_parameter=contrast,
                              ori_is_attacked=self.is_attacked.clone())

    def caa_linf(self, data, labels):
    import torch

        sur_data = data.detach()
        adv_data = data.detach().requires_grad_()
        ori_is_attacked = self.is_attacked.clone()
        for _ in range(self.max_inner_iter):
            outputs = self.model(adv_data)

            if not self._is_scheduling and self.early_stop:
                cur_pred = outputs.max(1, keepdim=True)[1].squeeze()
                self.is_attacked = torch.logical_or(ori_is_attacked,
                                                    cur_pred != labels.max(1, keepdim=True)[1].squeeze())

            with torch.enable_grad():
                cost = F.cross_entropy(outputs, labels)
            _grad = torch.autograd.grad(cost, adv_data)[0]
            if not self._is_scheduling:
                _grad[self.is_attacked] = 0
            adv_data = adv_data + self.step_size_pool[5] * torch.sign(_grad)
            eta = torch.clamp(adv_data - sur_data, min=self.eps_pool[5][0], max=self.eps_pool[5][1])
            adv_data = torch.clamp(sur_data + eta, min=0., max=1.).detach_().requires_grad_()

        return adv_data

    def get_linf_perturbation(self, data, noise):
    import torch

        return torch.clamp(data + noise, 0.0, 1.0)

    def update_attack_order(self, images, labels, adv_val=None):
    import torch

        def hungarian(matrix_batch):
            sol = torch.tensor([-i for i in range(1, matrix_batch.shape[0] + 1)], dtype=torch.int32)
            for i in range(matrix_batch.shape[0]):
                topk = 1
                sol[i] = torch.topk(matrix_batch[i], topk)[1][topk - 1]
                while sol.shape != torch.unique(sol).shape:
                    topk = topk + 1
                    sol[i] = torch.topk(matrix_batch[i], topk)[1][topk - 1]
            return sol

        def sinkhorn_normalization(ori_dsm, n_iters=20):
            for _ in range(n_iters):
                ori_dsm /= ori_dsm.sum(dim=0, keepdim=True)
                ori_dsm /= ori_dsm.sum(dim=1, keepdim=True)
            return ori_dsm

        if self.attack_order == 'fixed':
            if self.curr_seq is None:
                self.fixed_order = tuple([self.enabled_attack.index(i) for i in self.fixed_order])
                self.curr_seq = torch.tensor(self.fixed_order, device=self.device)
        elif self.attack_order == 'random':
            self.curr_seq = torch.randperm(self.seq_num)
        elif self.attack_order == 'scheduled':
            if self.curr_dsm is None:
                self.curr_dsm = sinkhorn_normalization(torch.rand((self.seq_num, self.seq_num)))
                self.curr_seq = hungarian(self.curr_dsm)
            self.curr_dsm = self.curr_dsm.detach().requires_grad_()
            adv_img = images.clone().detach().requires_grad_()
            original_iter_num = self.max_inner_iter
            self.max_inner_iter = 3
            self._is_scheduling = True
            for tdx in range(self.seq_num):
                prev_img = adv_img.clone()
                adv_img = torch.zeros_like(adv_img)
                for idx in range(self.seq_num):
                    if idx == self.linf_idx:
                        adv_img = adv_img + self.curr_dsm[tdx][idx] * self.attack_dict[idx](prev_img, labels)
                    else:
                        _adv_img, _ = self.attack_dict[idx](prev_img, adv_val[idx], labels)
                        adv_img = adv_img + self.curr_dsm[tdx][idx] * _adv_img
            self._is_scheduling = False
            self.max_inner_iter = original_iter_num
            outputs = self.model(adv_img)
            with torch.enable_grad():
                cost = F.cross_entropy(outputs, labels)

            dsm_grad = torch.autograd.grad(cost, self.curr_dsm)[0]

            prev_seq = self.curr_seq.clone()
            dsm_noise = torch.zeros_like(self.curr_dsm)
            while torch.equal(prev_seq, self.curr_seq):
                self.curr_dsm = sinkhorn_normalization(torch.exp(self.curr_dsm + dsm_grad + dsm_noise).detach())
                self.curr_seq = hungarian(self.curr_dsm.detach())
                dsm_noise = (torch.randn_like(self.curr_dsm) + 1) * 2  # Escaping local optimum
        else:
            raise ValueError()

    def caa_attack(self, images, labels):
    import torch

        attack = self.attack_dict
        adv_img = images.detach().clone()
        adv_val_saved = torch.zeros((self.seq_num, self.batch_size), device=self.device)
        adv_val = [self.adv_val_space[idx] for idx in range(self.seq_num)]

        if self.is_attacked.sum() > 0:
            for att_id in range(self.seq_num):
                if att_id == self.linf_idx:
                    continue
                adv_val[att_id].detach()
                adv_val[att_id][self.is_attacked] = adv_val_saved[att_id][self.is_attacked]
                adv_val[att_id].requires_grad_()

        for _ in range(self.max_iter):
            self.update_attack_order(images, labels, adv_val)

            adv_img = adv_img.detach().clone()
            self.is_not_attacked = torch.logical_not(self.is_attacked)
            adv_img[self.is_not_attacked] = images[self.is_not_attacked].clone()
            adv_img.requires_grad = True

            for tdx in range(self.seq_num):
                idx = self.curr_seq[tdx]
                if idx == self.linf_idx:
                    adv_img = attack[idx](adv_img, labels)
                else:
                    adv_img, adv_val_updated = attack[idx](adv_img, adv_val[idx], labels)
                    adv_val[idx] = adv_val_updated

            outputs = self.model(adv_img)
            cur_pred = outputs.max(1, keepdim=True)[1].squeeze()
            self.is_attacked = torch.logical_or(self.is_attacked, cur_pred != labels.max(1, keepdim=True)[1].squeeze())

            if self.is_attacked.sum() > 0:
                for att_id in range(self.seq_num):
                    if att_id == self.linf_idx:
                        continue
                    adv_val_saved[att_id][self.is_attacked] = adv_val[att_id][self.is_attacked].detach()

            if self.is_attacked.sum() == self.batch_size:
                break

        return adv_img
