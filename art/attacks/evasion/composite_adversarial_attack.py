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

from typing import Optional, Tuple, List, TYPE_CHECKING

import numpy as np
from tqdm.auto import tqdm

from art.attacks.attack import EvasionAttack
from art.config import ART_NUMPY_DTYPE
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import compute_success, check_and_transform_label_format

if TYPE_CHECKING:
    # pylint: disable=C0412
    import torch
    from art.estimators.classification.pytorch import PyTorchClassifier

logger = logging.getLogger(__name__)


class CompositeAdversarialAttackPyTorch(EvasionAttack):
    """
    Implementation of the composite adversarial attack on image classifiers in PyTorch. The attack is constructed by
    adversarially perturbing the hue component of the inputs. It uses order scheduling to search for the attack sequence
    and uses the iterative gradient sign method to optimize the perturbations in semantic space and Lp-ball (see
    `FastGradientMethod` and `BasicIterativeMethod`).

    | Note that this attack is intended for only PyTorch image classifiers with RGB images in the range [0, 1] as inputs

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
        hue_epsilon: Tuple[float, float] = (-np.pi, np.pi),
        sat_epsilon: Tuple[float, float] = (0.7, 1.3),
        rot_epsilon: Tuple[float, float] = (-10.0, 10.0),
        bri_epsilon: Tuple[float, float] = (-0.2, 0.2),
        con_epsilon: Tuple[float, float] = (0.7, 1.3),
        pgd_epsilon: Tuple[float, float] = (-8 / 255, 8 / 255),  # L-infinity
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
        :param enabled_attack: Attack pool selection, and attack order designation for fixed order. For simplicity,
                               we use the following abbreviations to specify each attack types. 0: Hue, 1: Saturation,
                               2: Rotation, 3: Brightness, 4: Contrast, 5: PGD(L-infinity). Therefore, `(0,1,2)` means
                               that the attack combines hue, saturation, and rotation; `(0,1,2,3,4)` means the
                               semantic attacks; `(0,1,2,3,4,5)` means the full attacks.
        :param hue_epsilon: The boundary of the hue perturbation. The value is expected to be in the interval
                           `[-np.pi, np.pi]`. Perturbation of `0` means no shift and `-np.pi` and `np.pi` give a
                           complete reversal of the hue channel in the HSV color space in the positive and negative
                           directions, respectively. See `kornia.enhance.adjust_hue` for more details.
        :param sat_epsilon: The boundary of the saturation perturbation. The value is expected to be in the interval
                            `[0, infinity]`. The perturbation of `0` gives a black-and-white image, `1` gives the
                            original image, and `2` enhances the saturation by a factor of 2. See
                            `kornia.geometry.transform.rotate` for more details.
        :param rot_epsilon: The boundary of the rotation perturbation (in degrees). Positive values mean
                            counter-clockwise rotation. See `kornia.geometry.transform.rotate` for more details.
        :param bri_epsilon: The boundary of the brightness perturbation. The value is expected to be in the interval
                           `[-1, 1]`. Perturbation of `0` means no shift, `-1` gives a complete black image, and `1`
                           gives a complete white image. See `kornia.enhance.adjust_brightness` for more details.
        :param con_epsilon: The boundary of the contrast perturbation. The value is expected to be in the interval
                           `[0, infinity]`. Perturbation of `0` gives a complete black image, `1` does not modify the
                           image, and any other value modifies the brightness by this factor. See
                           `kornia.enhance.adjust_contrast` for more details.
        :param pgd_epsilon: The maximum perturbation that the attacker can introduce in the L-infinity ball.
        :param early_stop: When True, the attack will stop if the perturbed example is classified incorrectly by the
                           classifier.
        :param max_iter: The maximum number of iterations for attack order optimization.
        :param max_inner_iter: The maximum number of iterations for each attack optimization.
        :param attack_order: Specify the scheduling type for composite adversarial attack. The value is expected to be
                             `fixed`, `random`, or `scheduled`. `fixed` means the attack order is the same as specified
                             in `enabled_attack`. `random` means the attack order is randomly generated at each
                             iteration. `scheduled` means to enable the attack order optimization proposed in the paper.
                             If only one attack is enabled, `fixed` will be used.
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
        self.epsilons = [hue_epsilon, sat_epsilon, rot_epsilon, bri_epsilon, con_epsilon, pgd_epsilon]
        self.early_stop = early_stop
        self.attack_order = attack_order
        self.max_iter = max_iter if self.attack_order == "scheduled" else 1
        self.max_inner_iter = max_inner_iter
        self.batch_size = batch_size
        self.verbose = verbose
        self._check_params()

        import kornia

        self.seq_num = len(self.enabled_attack)  # attack_num
        self.linf_idx = self.enabled_attack.index(5) if 5 in self.enabled_attack else None
        self.attack_pool = (
            self.caa_hue,
            self.caa_saturation,
            self.caa_rotation,
            self.caa_brightness,
            self.caa_contrast,
            self.caa_linf,
        )
        self.eps_pool = torch.tensor(self.epsilons, device=self.device)
        self.attack_pool_base = (
            kornia.enhance.adjust_hue,
            kornia.enhance.adjust_saturation,
            kornia.geometry.transform.rotate,
            kornia.enhance.adjust_brightness,
            kornia.enhance.adjust_contrast,
        )
        self.attack_dict = tuple(self.attack_pool[i] for i in self.enabled_attack)
        self.step_size_pool = [
            2.5 * ((eps[1] - eps[0]) / 2) / self.max_inner_iter for eps in self.eps_pool
        ]  # 2.5 * ε-test / num_steps

        self._description = "Composite Adversarial Attack"
        self._is_scheduling: bool = False
        self.eps_space: List = []
        self.adv_val_space: List = []
        self.curr_dsm: "torch.Tensor" = torch.zeros((len(self.enabled_attack), len(self.enabled_attack)))
        self.curr_seq: "torch.Tensor" = torch.zeros(len(self.enabled_attack))
        self.is_attacked: "torch.Tensor" = torch.zeros(self.batch_size, device=self.device).bool()
        self.is_not_attacked: "torch.Tensor" = torch.ones(self.batch_size, device=self.device).bool()

    def _check_params(self) -> None:
        """
        Check validity of parameters.
        """
        super()._check_params()
        if not isinstance(self.enabled_attack, tuple) or not all(
            value in [0, 1, 2, 3, 4, 5] for value in self.enabled_attack
        ):
            raise ValueError(
                "The parameter `enabled_attack` must be a tuple specifying the attack to launch. For simplicity, we use"
                + " the following abbreviations to specify each attack types. 0: Hue, 1: Saturation, 2: Rotation,"
                + " 3: Brightness, 4: Contrast, 5: PGD(L-infinity). Therefore, `(0,1,2)` means that the attack combines"
                + " hue, saturation, and rotation; `(0,1,2,3,4)` means the all semantic attacks; `(0,1,2,3,4,5)` means"
                + " the full attacks."
            )
        _epsilons_range = (
            ("hue_epsilon", (-np.pi, np.pi), "(-np.pi, np.pi)"),
            ("sat_epsilon", (0.0, np.inf), "(0.0, np.inf)"),
            ("rot_epsilon", (-360.0, 360.0), "(-360.0, 360.0)"),
            ("bri_epsilon", (-1.0, 1.0), "(-1.0, 1.0)"),
            ("con_epsilon", (0.0, np.inf), "(0.0, np.inf)"),
            ("pgd_epsilon", (-1.0, 1.0), "(-1.0, 1.0)"),
        )
        for i in range(6):
            if (
                not isinstance(self.epsilons[i], tuple)
                or not len(self.epsilons[i]) == 2
                or not (isinstance(self.epsilons[i][0], float) and isinstance(self.epsilons[i][1], float))
            ):
                logger.info(
                    "The argument `%s` must be an interval within %s of type tuple.",
                    _epsilons_range[i][0],
                    _epsilons_range[i][2],
                )
                raise TypeError(
                    f"The argument `{_epsilons_range[i][0]}` must be an interval "
                    f"within {_epsilons_range[i][2]} of type tuple."
                )

            if not _epsilons_range[i][1][0] <= self.epsilons[i][0] <= self.epsilons[i][1] <= _epsilons_range[i][1][1]:
                logger.info(
                    "The argument `%s` must be an interval within %s of type tuple.",
                    _epsilons_range[i][0],
                    _epsilons_range[i][2],
                )
                raise ValueError(
                    f"The argument `{_epsilons_range[i][0]}` must be an interval "
                    f"within {_epsilons_range[i][2]} of type tuple."
                )

        if not isinstance(self.early_stop, bool):
            logger.info("The flag `early_stop` has to be of type bool.")
            raise TypeError("The flag `early_stop` has to be of type bool.")

        if not isinstance(self.max_iter, int):
            logger.info("The argument `max_iter` must be positive of type int.")
            raise TypeError("The argument `max_iter` must be positive of type int.")

        if self.max_iter <= 0:
            logger.info("The argument `max_iter` must be positive of type int.")
            raise ValueError("The argument `max_iter` must be positive of type int.")

        if not isinstance(self.max_inner_iter, int):
            logger.info("The argument `max_inner_iter` must be positive of type int.")
            raise TypeError("The argument `max_inner_iter` must be positive of type int.")

        if self.max_inner_iter <= 0:
            logger.info("The argument `max_inner_iter` must be positive of type int.")
            raise ValueError("The argument `max_inner_iter` must be positive of type int.")

        if self.attack_order not in ("fixed", "random", "scheduled"):
            logger.info("The argument `attack_order` should be either `fixed`, `random`, or `scheduled`.")
            raise ValueError("The argument `attack_order` should be either `fixed`, `random`, or `scheduled`.")

        if self.batch_size <= 0:
            logger.info("The batch size has to be positive.")
            raise ValueError("The batch size has to be positive.")

        if not isinstance(self.verbose, bool):
            logger.info("The argument `verbose` has to be a Boolean.")
            raise TypeError("The argument `verbose` has to be a Boolean.")

    def _setup_attack(self):
        """
        Set up the initial parameter for each attack component.
        """
        import torch

        hue_space = (
            torch.rand(self.batch_size, device=self.device) * (self.eps_pool[0][1] - self.eps_pool[0][0])
            + self.eps_pool[0][0]
        )
        sat_space = (
            torch.rand(self.batch_size, device=self.device) * (self.eps_pool[1][1] - self.eps_pool[1][0])
            + self.eps_pool[1][0]
        )
        rot_space = (
            torch.rand(self.batch_size, device=self.device) * (self.eps_pool[2][1] - self.eps_pool[2][0])
            + self.eps_pool[2][0]
        )
        bri_space = (
            torch.rand(self.batch_size, device=self.device) * (self.eps_pool[3][1] - self.eps_pool[3][0])
            + self.eps_pool[3][0]
        )
        con_space = (
            torch.rand(self.batch_size, device=self.device) * (self.eps_pool[4][1] - self.eps_pool[4][0])
            + self.eps_pool[4][0]
        )
        pgd_space = 0.001 * torch.randn([self.batch_size, 3, 32, 32], device=self.device)

        self.eps_space = [self.eps_pool[i] for i in self.enabled_attack]
        self.adv_val_space = [
            [hue_space, sat_space, rot_space, bri_space, con_space, pgd_space][i] for i in self.enabled_attack
        ]

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate the composite adversarial samples and return them in a Numpy array.

        :param x: An array with the original inputs to be attacked.
        :param y: An array with the original labels to be predicted.
        :return: An array holding the composite adversarial examples.
        """
        if y is None:
            raise ValueError("The argument `y` must be provided.")

        import torch

        y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes)
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
        for batch_id, batch_all in enumerate(
            tqdm(data_loader, desc=self._description, leave=False, disable=not self.verbose)
        ):
            (batch_x, batch_y) = batch_all[0], batch_all[1]
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size

            x_adv[batch_index_1:batch_index_2] = self._generate_batch(x=batch_x, y=batch_y)

        logger.info(
            "Success rate of attack: %.2f%%",
            100 * compute_success(self.estimator, x, y, x_adv, batch_size=self.batch_size),
        )

        return x_adv

    def _generate_batch(self, x: "torch.Tensor", y: "torch.Tensor") -> np.ndarray:
        """
        Generate a batch of composite adversarial examples and return them in a NumPy array.

        :param x: A tensor of a batch of original inputs to be attacked.
        :param y: A tensor of a batch of the original labels to be predicted.
        :return: An array holding the composite adversarial examples.
        """
        import torch

        self.batch_size = x.shape[0]
        self._setup_attack()
        self.is_attacked = torch.zeros(self.batch_size, device=self.device).bool()
        self.is_not_attacked = torch.ones(self.batch_size, device=self.device).bool()
        x, y = x.to(self.device), y.to(self.device)

        return self.caa_attack(x, y).cpu().detach().numpy()

    def _comp_pgd(
        self,
        data: "torch.Tensor",
        labels: "torch.Tensor",
        attack_idx: int,
        attack_parameter: "torch.Tensor",
        ori_is_attacked: "torch.Tensor",
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        Compute the adversarial examples for each attack component.

        :param data: A tensor of a batch of original inputs to be attacked.
        :param labels: A tensor of a batch of the original labels to be predicted.
        :param attack_idx: The index of the attack component (one of the enabled attacks) in the attack pool.
        :param attack_parameter: Specify the parameter of the attack component. For example, hue shift angle, saturation
         factor, etc.
        :param ori_is_attacked: Specify whether the perturbed data is already attacked.
        :return: The perturbed data and the corresponding attack parameter.
        """
        import torch
        import torch.nn.functional as F

        adv_data = self.attack_pool_base[attack_idx](data, attack_parameter)
        for _ in range(self.max_inner_iter):
            outputs = self.model(adv_data)

            if not self._is_scheduling and self.early_stop:
                cur_pred = outputs.max(1, keepdim=True)[1].squeeze()
                self.is_attacked = torch.logical_or(
                    ori_is_attacked, cur_pred != labels.max(1, keepdim=True)[1].squeeze()
                )

            with torch.enable_grad():
                cost = F.cross_entropy(outputs, labels)
            _grad = torch.autograd.grad(cost, attack_parameter)[0]
            if not self._is_scheduling:
                _grad[self.is_attacked] = 0
            attack_parameter = (
                torch.clamp(
                    attack_parameter + torch.sign(_grad) * self.step_size_pool[attack_idx],
                    self.eps_pool[attack_idx][0],
                    self.eps_pool[attack_idx][1],
                )
                .detach()
                .requires_grad_()
            )
            adv_data = self.attack_pool_base[attack_idx](data, attack_parameter)

        return adv_data, attack_parameter

    def caa_hue(
        self, data: "torch.Tensor", hue: "torch.Tensor", labels: "torch.Tensor"
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        Compute the adversarial examples for hue component.

        :param data: A tensor of a batch of original inputs to be attacked.
        :param hue: Specify the hue shift angle.
        :param labels: A tensor of a batch of the original labels to be predicted.
        :return: The perturbed data and the corresponding hue shift angle.
        """
        hue = hue.detach().clone()
        hue[self.is_attacked] = 0
        hue.requires_grad_()
        sur_data = data.detach().requires_grad_()

        return self._comp_pgd(
            data=sur_data, labels=labels, attack_idx=0, attack_parameter=hue, ori_is_attacked=self.is_attacked.clone()
        )

    def caa_saturation(
        self, data: "torch.Tensor", saturation: "torch.Tensor", labels: "torch.Tensor"
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        Compute the adversarial examples for saturation component.

        :param data: A tensor of a batch of original inputs to be attacked.
        :param saturation: Specify the saturation factor.
        :param labels: A tensor of a batch of the original labels to be predicted.
        :return: The perturbed data and the corresponding saturation factor.
        """
        saturation = saturation.detach().clone()
        saturation[self.is_attacked] = 1
        saturation.requires_grad_()
        sur_data = data.detach().requires_grad_()

        return self._comp_pgd(
            data=sur_data,
            labels=labels,
            attack_idx=1,
            attack_parameter=saturation,
            ori_is_attacked=self.is_attacked.clone(),
        )

    def caa_rotation(
        self, data: "torch.Tensor", theta: "torch.Tensor", labels: "torch.Tensor"
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        Compute the adversarial examples for rotation component.

        :param data: A tensor of a batch of original inputs to be attacked.
        :param theta: Specify the rotation angle.
        :param labels: A tensor of a batch of the original labels to be predicted.
        :return: The perturbed data and the corresponding rotation angle.
        """
        theta = theta.detach().clone()
        theta[self.is_attacked] = 0
        theta.requires_grad_()
        sur_data = data.detach().requires_grad_()

        return self._comp_pgd(
            data=sur_data, labels=labels, attack_idx=2, attack_parameter=theta, ori_is_attacked=self.is_attacked.clone()
        )

    def caa_brightness(
        self, data: "torch.Tensor", brightness: "torch.Tensor", labels: "torch.Tensor"
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        Compute the adversarial examples for brightness component.

        :param data: A tensor of a batch of original inputs to be attacked.
        :param brightness: Specify the brightness factor.
        :param labels: A tensor of a batch of the original labels to be predicted.
        :return: The perturbed data and the corresponding brightness factor.
        """
        brightness = brightness.detach().clone()
        brightness[self.is_attacked] = 0
        brightness.requires_grad_()
        sur_data = data.detach().requires_grad_()

        return self._comp_pgd(
            data=sur_data,
            labels=labels,
            attack_idx=3,
            attack_parameter=brightness,
            ori_is_attacked=self.is_attacked.clone(),
        )

    def caa_contrast(
        self, data: "torch.Tensor", contrast: "torch.Tensor", labels: "torch.Tensor"
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        Compute the adversarial examples for contrast component.

        :param data: A tensor of a batch of original inputs to be attacked.
        :param contrast: Specify the contrast factor.
        :param labels: A tensor of a batch of the original labels to be predicted.
        :return: The perturbed data and the corresponding contrast factor.
        """
        contrast = contrast.detach().clone()
        contrast[self.is_attacked] = 1
        contrast.requires_grad_()
        sur_data = data.detach().requires_grad_()

        return self._comp_pgd(
            data=sur_data,
            labels=labels,
            attack_idx=4,
            attack_parameter=contrast,
            ori_is_attacked=self.is_attacked.clone(),
        )

    def caa_linf(
        self, data: "torch.Tensor", eta: "torch.Tensor", labels: "torch.Tensor"
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        Compute the adversarial examples for L-infinity (PGD) component.

        :param data: A tensor of a batch of original inputs to be attacked.
        :param eta: The perturbation in the L-infinity ball.
        :param labels: A tensor of a batch of the original labels to be predicted.
        :return: The perturbed data.
        """
        import torch
        import torch.nn.functional as F

        sur_data = data.detach()
        adv_data = data.detach().requires_grad_()
        ori_is_attacked = self.is_attacked.clone()
        for _ in range(self.max_inner_iter):
            outputs = self.model(adv_data)

            if not self._is_scheduling and self.early_stop:
                cur_pred = outputs.max(1, keepdim=True)[1].squeeze()
                self.is_attacked = torch.logical_or(
                    ori_is_attacked, cur_pred != labels.max(1, keepdim=True)[1].squeeze()
                )

            with torch.enable_grad():
                cost = F.cross_entropy(outputs, labels)
            _grad = torch.autograd.grad(cost, adv_data)[0]
            if not self._is_scheduling:
                _grad[self.is_attacked] = 0
            adv_data = adv_data + self.step_size_pool[5] * torch.sign(_grad)
            eta = torch.clamp(adv_data - sur_data, min=self.eps_pool[5][0], max=self.eps_pool[5][1])
            adv_data = torch.clamp(sur_data + eta, min=0.0, max=1.0).detach_().requires_grad_()

        return adv_data, eta

    def update_attack_order(self, images: "torch.Tensor", labels: "torch.Tensor", adv_val: List) -> None:
        """
        Update the specified attack ordering.

        :param images: A tensor of a batch of original inputs to be attacked.
        :param labels: A tensor of a batch of the original labels to be predicted.
        :param adv_val: Optional; A list of a batch of current attack parameters.
        """
        import torch
        import torch.nn.functional as F

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

        if self.attack_order == "fixed":
            if self.curr_seq.sum() == 0:
                self.fixed_order = tuple(self.enabled_attack.index(i) for i in self.fixed_order)
                self.curr_seq = torch.tensor(self.fixed_order, device=self.device)
        elif self.attack_order == "random":
            self.curr_seq = torch.randperm(self.seq_num)
        elif self.attack_order == "scheduled":
            if self.curr_seq.sum() == 0:
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

    def caa_attack(self, images: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
        """
        The main algorithm to generate the adversarial examples for composite adversarial attack.

        :param images: A tensor of a batch of original inputs to be attacked.
        :param labels: A tensor of a batch of the original labels to be predicted.
        :return: The perturbed data.
        """
        import torch

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
                adv_img, adv_val_updated = self.attack_dict[idx](adv_img, adv_val[idx], labels)  # type: ignore
                if idx != self.linf_idx:
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
