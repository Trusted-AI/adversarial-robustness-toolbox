# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
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
This module implements the Projected Gradient Descent attack `ProjectedGradientDescent` as an iterative method in which,
after each iteration, the perturbation is projected on an lp-ball of specified radius (in addition to clipping the
values of the adversarial sample so that it lies in the permitted data range). This is the attack proposed by Madry et
al. for adversarial training.

| Paper link: https://arxiv.org/abs/1706.06083
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, Union, TYPE_CHECKING

import numpy as np
from tqdm.auto import tqdm

from art.config import ART_NUMPY_DTYPE
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.estimators.classification.classifier import ClassifierMixin
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent_numpy import (
    ProjectedGradientDescentCommon,
)
from art.utils import compute_success, random_sphere, compute_success_array

if TYPE_CHECKING:
    # pylint: disable=C0412
    import torch
    from art.estimators.classification.pytorch import PyTorchClassifier

logger = logging.getLogger(__name__)


class ProjectedGradientDescentPyTorch(ProjectedGradientDescentCommon):
    """
    The Projected Gradient Descent attack is an iterative method in which, after each iteration, the perturbation is
    projected on an lp-ball of specified radius (in addition to clipping the values of the adversarial sample so that it
    lies in the permitted data range). This is the attack proposed by Madry et al. for adversarial training.

    | Paper link: https://arxiv.org/abs/1706.06083
    """

    _estimator_requirements = (BaseEstimator, LossGradientsMixin, ClassifierMixin)  # type: ignore

    def __init__(
        self,
        estimator: Union["PyTorchClassifier"],
        norm: Union[int, float, str] = np.inf,
        eps: Union[int, float, np.ndarray] = 0.3,
        eps_step: Union[int, float, np.ndarray] = 0.1,
        max_iter: int = 100,
        targeted: bool = False,
        num_random_init: int = 0,
        batch_size: int = 32,
        random_eps: bool = False,
        tensor_board: Union[str, bool] = False,
        verbose: bool = True,
    ):
        """
        Create a :class:`.ProjectedGradientDescentPyTorch` instance.

        :param estimator: An trained estimator.
        :param norm: The norm of the adversarial perturbation. Possible values: "inf", np.inf, 1 or 2.
        :param eps: Maximum perturbation that the attacker can introduce.
        :param eps_step: Attack step size (input variation) at each iteration.
        :param random_eps: When True, epsilon is drawn randomly from truncated normal distribution. The literature
                           suggests this for FGSM based training to generalize across different epsilons. eps_step is
                           modified to preserve the ratio of eps / eps_step. The effectiveness of this method with PGD
                           is untested (https://arxiv.org/pdf/1611.01236.pdf).
        :param max_iter: The maximum number of iterations.
        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False).
        :param num_random_init: Number of random initialisations within the epsilon ball. For num_random_init=0 starting
                                at the original input.
        :param batch_size: Size of the batch on which adversarial samples are generated.
        :param tensor_board: Activate summary writer for TensorBoard: Default is `False` and deactivated summary writer.
                             If `True` save runs/CURRENT_DATETIME_HOSTNAME in current directory. Provide `path` in type
                             `str` to save in path/CURRENT_DATETIME_HOSTNAME.
                             Use hierarchical folder structure to compare between runs easily. e.g. pass in ‘runs/exp1’,
                             ‘runs/exp2’, etc. for each new experiment to compare across them.
        :param verbose: Show progress bars.
        """
        if not estimator.all_framework_preprocessing:
            raise NotImplementedError(
                "The framework-specific implementation only supports framework-specific preprocessing."
            )

        super().__init__(
            estimator=estimator,
            norm=norm,
            eps=eps,
            eps_step=eps_step,
            max_iter=max_iter,
            targeted=targeted,
            num_random_init=num_random_init,
            batch_size=batch_size,
            random_eps=random_eps,
            verbose=verbose,
            tensor_board=tensor_board,
        )

        self._batch_id = 0
        self._i_max_iter = 0

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :param mask: An array with a mask broadcastable to input `x` defining where to apply adversarial perturbations.
                     Shape needs to be broadcastable to the shape of x and can also be of the same shape as `x`. Any
                     features for which the mask is zero will not be adversarially perturbed.
        :type mask: `np.ndarray`
        :return: An array holding the adversarial examples.
        """
        import torch  # lgtm [py/repeated-import]

        mask = self._get_mask(x, **kwargs)

        # Ensure eps is broadcastable
        self._check_compatibility_input_and_eps(x=x)

        # Check whether random eps is enabled
        self._random_eps()

        # Set up targets
        targets = self._set_targets(x, y)

        # Create dataset
        if mask is not None:
            # Here we need to make a distinction: if the masks are different for each input, we need to index
            # those for the current batch. Otherwise (i.e. mask is meant to be broadcasted), keep it as it is.
            if len(mask.shape) == len(x.shape):
                dataset = torch.utils.data.TensorDataset(
                    torch.from_numpy(x.astype(ART_NUMPY_DTYPE)),
                    torch.from_numpy(targets.astype(ART_NUMPY_DTYPE)),
                    torch.from_numpy(mask.astype(ART_NUMPY_DTYPE)),
                )

            else:
                dataset = torch.utils.data.TensorDataset(
                    torch.from_numpy(x.astype(ART_NUMPY_DTYPE)),
                    torch.from_numpy(targets.astype(ART_NUMPY_DTYPE)),
                    torch.from_numpy(np.array([mask.astype(ART_NUMPY_DTYPE)] * x.shape[0])),
                )

        else:
            dataset = torch.utils.data.TensorDataset(
                torch.from_numpy(x.astype(ART_NUMPY_DTYPE)),
                torch.from_numpy(targets.astype(ART_NUMPY_DTYPE)),
            )

        data_loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=self.batch_size, shuffle=False, drop_last=False
        )

        # Start to compute adversarial examples
        adv_x = x.astype(ART_NUMPY_DTYPE)

        # Compute perturbation with batching
        for (batch_id, batch_all) in enumerate(
            tqdm(data_loader, desc="PGD - Batches", leave=False, disable=not self.verbose)
        ):

            self._batch_id = batch_id

            if mask is not None:
                (batch, batch_labels, mask_batch) = batch_all[0], batch_all[1], batch_all[2]
            else:
                (batch, batch_labels, mask_batch) = batch_all[0], batch_all[1], None

            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size

            # Compute batch_eps and batch_eps_step
            if isinstance(self.eps, np.ndarray) and isinstance(self.eps_step, np.ndarray):
                if len(self.eps.shape) == len(x.shape) and self.eps.shape[0] == x.shape[0]:
                    batch_eps = self.eps[batch_index_1:batch_index_2]
                    batch_eps_step = self.eps_step[batch_index_1:batch_index_2]

                else:
                    batch_eps = self.eps
                    batch_eps_step = self.eps_step

            else:
                batch_eps = self.eps
                batch_eps_step = self.eps_step

            for rand_init_num in range(max(1, self.num_random_init)):
                if rand_init_num == 0:
                    # first iteration: use the adversarial examples as they are the only ones we have now
                    adv_x[batch_index_1:batch_index_2] = self._generate_batch(
                        x=batch, targets=batch_labels, mask=mask_batch, eps=batch_eps, eps_step=batch_eps_step
                    )
                else:
                    adversarial_batch = self._generate_batch(
                        x=batch, targets=batch_labels, mask=mask_batch, eps=batch_eps, eps_step=batch_eps_step
                    )

                    # return the successful adversarial examples
                    attack_success = compute_success_array(
                        self.estimator,
                        batch,
                        batch_labels,
                        adversarial_batch,
                        self.targeted,
                        batch_size=self.batch_size,
                    )
                    adv_x[batch_index_1:batch_index_2][attack_success] = adversarial_batch[attack_success]

        logger.info(
            "Success rate of attack: %.2f%%",
            100 * compute_success(self.estimator, x, targets, adv_x, self.targeted, batch_size=self.batch_size),
        )

        return adv_x

    def _generate_batch(
        self,
        x: "torch.Tensor",
        targets: "torch.Tensor",
        mask: "torch.Tensor",
        eps: Union[int, float, np.ndarray],
        eps_step: Union[int, float, np.ndarray],
    ) -> np.ndarray:
        """
        Generate a batch of adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :param targets: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)`.
        :param mask: An array with a mask to be applied to the adversarial perturbations. Shape needs to be
                     broadcastable to the shape of x. Any features for which the mask is zero will not be adversarially
                     perturbed.
        :param eps: Maximum perturbation that the attacker can introduce.
        :param eps_step: Attack step size (input variation) at each iteration.
        :return: Adversarial examples.
        """
        import torch  # lgtm [py/repeated-import]

        inputs = x.to(self.estimator.device)
        targets = targets.to(self.estimator.device)
        adv_x = torch.clone(inputs)

        if mask is not None:
            mask = mask.to(self.estimator.device)

        for i_max_iter in range(self.max_iter):
            self._i_max_iter = i_max_iter
            adv_x = self._compute_torch(
                adv_x,
                inputs,
                targets,
                mask,
                eps,
                eps_step,
                self.num_random_init > 0 and i_max_iter == 0,
            )

        return adv_x.cpu().detach().numpy()

    def _compute_perturbation(  # pylint: disable=W0221
        self, x: "torch.Tensor", y: "torch.Tensor", mask: Optional["torch.Tensor"]
    ) -> "torch.Tensor":
        """
        Compute perturbations.

        :param x: Current adversarial examples.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :param mask: An array with a mask broadcastable to input `x` defining where to apply adversarial perturbations.
                     Shape needs to be broadcastable to the shape of x and can also be of the same shape as `x`. Any
                     features for which the mask is zero will not be adversarially perturbed.
        :return: Perturbations.
        """
        import torch  # lgtm [py/repeated-import]

        # Pick a small scalar to avoid division by 0
        tol = 10e-8

        # Get gradient wrt loss; invert it if attack is targeted
        grad = self.estimator.loss_gradient(x=x, y=y) * (1 - 2 * int(self.targeted))

        # Write summary
        if self.summary_writer is not None:  # pragma: no cover
            self.summary_writer.add_scalar(
                "gradients/norm-L1/batch-{}".format(self._batch_id),
                np.linalg.norm(grad.flatten(), ord=1),
                global_step=self._i_max_iter,
            )
            self.summary_writer.add_scalar(
                "gradients/norm-L2/batch-{}".format(self._batch_id),
                np.linalg.norm(grad.flatten(), ord=2),
                global_step=self._i_max_iter,
            )
            self.summary_writer.add_scalar(
                "gradients/norm-Linf/batch-{}".format(self._batch_id),
                np.linalg.norm(grad.flatten(), ord=np.inf),
                global_step=self._i_max_iter,
            )

            if hasattr(self.estimator, "compute_losses"):
                losses = self.estimator.compute_losses(x=x, y=y)

                for key, value in losses.items():
                    self.summary_writer.add_scalar(
                        "loss/{}/batch-{}".format(key, self._batch_id),
                        np.mean(value.detach().cpu().numpy()),
                        global_step=self._i_max_iter,
                    )

        # Check for nan before normalisation an replace with 0
        if torch.any(grad.isnan()):  # pragma: no cover
            logger.warning("Elements of the loss gradient are NaN and have been replaced with 0.0.")
            grad[grad.isnan()] = 0.0

        # Apply mask
        if mask is not None:
            grad = torch.where(mask == 0.0, torch.tensor(0.0).to(self.estimator.device), grad)

        # Apply norm bound
        if self.norm in ["inf", np.inf]:
            grad = grad.sign()

        elif self.norm == 1:
            ind = tuple(range(1, len(x.shape)))
            grad = grad / (torch.sum(grad.abs(), dim=ind, keepdims=True) + tol)  # type: ignore

        elif self.norm == 2:
            ind = tuple(range(1, len(x.shape)))
            grad = grad / (torch.sqrt(torch.sum(grad * grad, axis=ind, keepdims=True)) + tol)  # type: ignore

        assert x.shape == grad.shape

        return grad

    def _apply_perturbation(  # pylint: disable=W0221
        self, x: "torch.Tensor", perturbation: "torch.Tensor", eps_step: Union[int, float, np.ndarray]
    ) -> "torch.Tensor":
        """
        Apply perturbation on examples.

        :param x: Current adversarial examples.
        :param perturbation: Current perturbations.
        :param eps_step: Attack step size (input variation) at each iteration.
        :return: Adversarial examples.
        """
        import torch  # lgtm [py/repeated-import]

        eps_step = np.array(eps_step, dtype=ART_NUMPY_DTYPE)
        perturbation_step = torch.tensor(eps_step).to(self.estimator.device) * perturbation
        perturbation_step[torch.isnan(perturbation_step)] = 0
        x = x + perturbation_step
        if self.estimator.clip_values is not None:
            clip_min, clip_max = self.estimator.clip_values
            x = torch.max(
                torch.min(x, torch.tensor(clip_max).to(self.estimator.device)),
                torch.tensor(clip_min).to(self.estimator.device),
            )

        return x

    def _compute_torch(
        self,
        x: "torch.Tensor",
        x_init: "torch.Tensor",
        y: "torch.Tensor",
        mask: "torch.Tensor",
        eps: Union[int, float, np.ndarray],
        eps_step: Union[int, float, np.ndarray],
        random_init: bool,
    ) -> "torch.Tensor":
        """
        Compute adversarial examples for one iteration.

        :param x: Current adversarial examples.
        :param x_init: An array with the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236).
        :param mask: An array with a mask broadcastable to input `x` defining where to apply adversarial perturbations.
                     Shape needs to be broadcastable to the shape of x and can also be of the same shape as `x`. Any
                     features for which the mask is zero will not be adversarially perturbed.
        :param eps: Maximum perturbation that the attacker can introduce.
        :param eps_step: Attack step size (input variation) at each iteration.
        :param random_init: Random initialisation within the epsilon ball. For random_init=False starting at the
                            original input.
        :return: Adversarial examples.
        """
        import torch  # lgtm [py/repeated-import]

        if random_init:
            n = x.shape[0]
            m = np.prod(x.shape[1:]).item()

            random_perturbation = random_sphere(n, m, eps, self.norm).reshape(x.shape).astype(ART_NUMPY_DTYPE)
            random_perturbation = torch.from_numpy(random_perturbation).to(self.estimator.device)

            if mask is not None:
                random_perturbation = random_perturbation * mask

            x_adv = x + random_perturbation

            if self.estimator.clip_values is not None:
                clip_min, clip_max = self.estimator.clip_values
                x_adv = torch.max(
                    torch.min(x_adv, torch.tensor(clip_max).to(self.estimator.device)),
                    torch.tensor(clip_min).to(self.estimator.device),
                )

        else:
            x_adv = x

        # Get perturbation
        perturbation = self._compute_perturbation(x_adv, y, mask)

        # Apply perturbation and clip
        x_adv = self._apply_perturbation(x_adv, perturbation, eps_step)

        # Do projection
        perturbation = self._projection(x_adv - x_init, eps, self.norm)

        # Recompute x_adv
        x_adv = perturbation + x_init

        return x_adv

    def _projection(
        self, values: "torch.Tensor", eps: Union[int, float, np.ndarray], norm_p: Union[int, float, str]
    ) -> "torch.Tensor":
        """
        Project `values` on the L_p norm ball of size `eps`.

        :param values: Values to clip.
        :param eps: Maximum norm allowed.
        :param norm_p: L_p norm to use for clipping supporting 1, 2, `np.Inf` and "inf".
        :return: Values of `values` after projection.
        """
        import torch  # lgtm [py/repeated-import]

        # Pick a small scalar to avoid division by 0
        tol = 10e-8
        values_tmp = values.reshape(values.shape[0], -1)

        if norm_p == 2:
            if isinstance(eps, np.ndarray):
                raise NotImplementedError(
                    "The parameter `eps` of type `np.ndarray` is not supported to use with norm 2."
                )

            values_tmp = (
                values_tmp
                * torch.min(
                    torch.tensor([1.0], dtype=torch.float32).to(self.estimator.device),
                    eps / (torch.norm(values_tmp, p=2, dim=1) + tol),
                ).unsqueeze_(-1)
            )

        elif norm_p == 1:
            if isinstance(eps, np.ndarray):
                raise NotImplementedError(
                    "The parameter `eps` of type `np.ndarray` is not supported to use with norm 1."
                )

            values_tmp = (
                values_tmp
                * torch.min(
                    torch.tensor([1.0], dtype=torch.float32).to(self.estimator.device),
                    eps / (torch.norm(values_tmp, p=1, dim=1) + tol),
                ).unsqueeze_(-1)
            )

        elif norm_p in [np.inf, "inf"]:
            if isinstance(eps, np.ndarray):
                eps = eps * np.ones_like(values.cpu())
                eps = eps.reshape([eps.shape[0], -1])

            values_tmp = values_tmp.sign() * torch.min(
                values_tmp.abs(), torch.tensor([eps], dtype=torch.float32).to(self.estimator.device)
            )

        else:
            raise NotImplementedError(
                "Values of `norm_p` different from 1, 2 and `np.inf` are currently not supported."
            )

        values = values_tmp.reshape(values.shape)

        return values
