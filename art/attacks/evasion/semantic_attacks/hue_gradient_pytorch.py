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
This module implements a semantic attack by adversarially perturbing the hue component of the inputs. It uses
iterative gradient sign method to optimise the semantic perturbations (see `FastGradientmethod` and
`BasicIterativeMethod`). This implementation extends the original optimization method to other norms as well.

| Paper link: https://arxiv.org/abs/2202.04235
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
from tqdm.auto import tqdm

from art.attacks.attack import EvasionAttack
from art.config import ART_NUMPY_DTYPE
from art.summary_writer import SummaryWriter
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import (
    compute_success,
    compute_success_array,
    check_and_transform_label_format,
    get_labels_np_array
)

if TYPE_CHECKING:
    # pylint: disable=C0412
    import torch
    from art.estimators.classification.pytorch import PyTorchClassifier

logger = logging.getLogger(__name__)


class HueGradientPyTorch(EvasionAttack):
    """
    Implementation of the hue attack on image classifiers in PyTorch. The attack is constructed by adversarially
    perturbing the hue component of the inputs. It uses iterative gradient sign method to optimise the semantic
    perturbations (see `FastGradientmethod` and `BasicIterativeMethod`). This implementation extends the original
    optimisation method to other norms as well.

    Note that this attack is intended for only PyTorch image classifiers with RGB images as inputs.

    | Paper link: https://arxiv.org/abs/2202.04235
    """

    attack_params = EvasionAttack.attack_params + [
        "norm",
        "factor_min",
        "factor_max",
        "step_size",
        "max_iter",
        "targeted",
        "num_random_init",
        "batch_size",
        "summary_writer",
        "verbose",
    ]

    _estimator_requirements = (BaseEstimator, LossGradientsMixin, ClassifierMixin)  # type: ignore

    def __init__(
        self,
        classifier: "PyTorchClassifier",
        norm: Union[int, float, str] = np.inf,
        factor_min: float = -np.pi,
        factor_max: float = np.pi,
        step_size: Union[int, float] = 0.1,
        max_iter: int = 10,
        targeted: bool = False,
        num_random_init: int = 0,
        batch_size: int = 32,
        summary_writer: Union[str, bool, SummaryWriter] = False,
        verbose: bool = True,
    ) -> None:
        """
        Create an instance of the :class:`.HueGradientPyTorch`.

        :param classifier: A trained PyTorch classifier.
        :param norm: The norm of the adversarial perturbation. Possible values: `"inf"`, `np.inf`, `1` or `2`.
        :param factor_min: The lower bound of the hue perturbation. The value is expected to be in the interval
                           `[-np.pi, np.pi]`.
        :param factor_max: The upper bound of the hue perturbation. The value is expected to be in the interval
                           `[-np.pi, np.pi]` and greater than `factor_min`.
        :param step_size: Attack step size at each iteration.
        :param max_iter: The maximum number of iterations.
        :param targeted: Indicates whether the attack is targeted (`True`) or untargeted (`False`).
        :param num_random_init: Number of random hue initialisations to try within the `[factor_min, factor_max]`
                                interval. For `num_random_init=0`, the attack starts at the original input i.e., the
                                initial hue factor (perturbation) is set to `0`.
        :param batch_size: The batch size to use during the generation of adversarial samples.
        :param summary_writer: Activate summary writer for TensorBoard.
                               Default is `False` and deactivated summary writer.
                               If `True` save runs/CURRENT_DATETIME_HOSTNAME in the current directory.
                               If of type `str` save in path.
                               If of type `SummaryWriter` apply provided custom summary writer.
                               Use hierarchical folder structure to compare between runs easily. e.g., pass in
                               ‘runs/exp1’, ‘runs/exp2’, etc., for each new experiment to compare across them.
        :param verbose: Show progress bars.
        """
        if summary_writer and num_random_init > 1:
            raise ValueError("TensorBoard is not yet supported for more than 1 random restart (num_random_init>1).")

        super().__init__(estimator=classifier, summary_writer=summary_writer)
        self.norm = norm
        self.factor_min = factor_min
        self.factor_max = factor_max
        self.step_size = step_size
        self.max_iter = max_iter
        self.targeted = targeted
        self.num_random_init = num_random_init
        self.batch_size = batch_size
        self.verbose = verbose

        self._check_params()

        self._batch_id = 0
        self._i_max_iter = 0

    def _check_params(self) -> None:

        if self.norm not in [1, 2, np.inf, "inf"]:
            raise ValueError('Norm order must be either 1, 2, `np.inf` or "inf".')

        if not isinstance(self.factor_min, float) or not (-np.pi <= self.factor_min <= np.pi):
            raise ValueError("The argument `factor_min` must be in [-np.pi, np.pi] and of type float.")

        if not isinstance(self.factor_max, float) or not (-np.pi <= self.factor_max <= np.pi):
            raise ValueError("The argument `factor_max` must be in [-np.pi, np.pi] and of type float.")

        if not (self.factor_min < self.factor_max):
            raise ValueError("The argument `factor_min` must be less than the argument `factor_max`.")

        if not isinstance(self.step_size, (int, float)) or self.step_size <= 0.0:
            raise ValueError("The argument `step_size` must be positive of type int or float.")

        if not isinstance(self.max_iter, int) or self.max_iter <= 0:
            raise ValueError("The argument `max_iter` must be positive of type int.")

        if not isinstance(self.targeted, bool):
            raise ValueError("The flag `targeted` has to be of type bool.")

        if not isinstance(self.num_random_init, int):
            raise TypeError("The argument `num_random_init` has to be of type integer")

        if self.num_random_init < 0:
            raise ValueError("The number of random initialisations `random_init` has to be greater than or equal to 0.")

        if self.batch_size <= 0:
            raise ValueError("The batch size has to be positive.")

        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be a Boolean.")

    @staticmethod
    def _get_mask(x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Get the mask from the kwargs.

        :param x: An array with all the original inputs.
        :param mask: A 1D array of masks defining which samples to perturb. Shape needs to be `(nb_samples,)`.
                     Samples for which the mask is zero will not be adversarially perturbed.
        :type mask: `np.ndarray`
        :return: The mask.
        """
        mask = kwargs.get("mask")

        if mask is not None:
            if not (np.issubdtype(mask.dtype, np.floating) or mask.dtype == bool):  # pragma: no cover
                raise ValueError(
                    f"The `mask` has to be either of type np.float32, np.float64 or bool. The provided"
                    f"`mask` is of type {mask.dtype}."
                )

            if np.issubdtype(mask.dtype, np.floating) and np.amin(mask) < 0.0:  # pragma: no cover
                raise ValueError(
                    "The `mask` of type np.float32 or np.float64 requires all elements to be either zero"
                    "or positive values."
                )

            if not (mask.shape == (x.shape[0],)):  # pragma: no cover
                raise ValueError(
                    "The `mask` should be 1D and of shape `(nb_samples,)`."
                )

        return mask

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

    def _clip_input(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Rounds the inputs to the correct level of granularity. Useful to ensure the data passed to the classifier is
        represented in the correct domain. For e.g., `[0, 255]` integers versus `[0, 1]` or `[0, 255]` floating points.

        :param x: Sample inputs with shape as expected by the model.
        :return: Clipped and rounded inputs.
        """
        import torch

        if self.estimator.clip_values is not None:
            x = torch.clamp(x, self.estimator.clip_values[0], self.estimator.clip_values[1])

        return x

    def generate(
            self,
            x: np.ndarray,
            y: Optional[np.ndarray] = None,
            **kwargs
    ) -> np.ndarray:
        """
        Generate adversarial samples and return them in a NumPy array.

        :param x: An array with the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  `(nb_samples,)`. Only provide this parameter if you'd like to use true labels when crafting
                  adversarial samples. Otherwise, model predictions are used as labels to avoid the "label leaking"
                  effect (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :param mask: A 1D array of masks defining which samples to perturb. Shape needs to be `(nb_samples,)`.
                     Samples for which the mask is zero will not be adversarially perturbed.
        :type mask: `np.ndarray`
        :return: An array holding the adversarial examples.
        """
        import torch

        mask = self._get_mask(x, **kwargs)

        targets = self._set_targets(x, y)

        # Create the data loader.
        if mask is not None:
            dataset = torch.utils.data.TensorDataset(
                torch.from_numpy(x.astype(ART_NUMPY_DTYPE)),
                torch.from_numpy(targets.astype(ART_NUMPY_DTYPE)),
                torch.from_numpy(mask.astype(ART_NUMPY_DTYPE)),
            )
        else:
            dataset = torch.utils.data.TensorDataset(
                torch.from_numpy(x.astype(ART_NUMPY_DTYPE)),
                torch.from_numpy(targets.astype(ART_NUMPY_DTYPE)),
            )

        data_loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=self.batch_size, shuffle=False, drop_last=False
        )

        # Start to compute adversarial examples.
        x_adv = x.copy().astype(ART_NUMPY_DTYPE)

        # Compute perturbations with batching.
        for (batch_id, batch_all) in enumerate(
            tqdm(data_loader, desc="HueGradientAttack", leave=False, disable=not self.verbose)
        ):
            self._batch_id = batch_id

            if mask is not None:
                (batch_x, batch_targets, batch_mask) = batch_all[0], batch_all[1], batch_all[2]
            else:
                (batch_x, batch_targets, batch_mask) = batch_all[0], batch_all[1], None

            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size

            for rand_init_num in range(max(1, self.num_random_init)):
                if rand_init_num == 0:
                    # First iteration: use the current adversarial examples as they are the only ones we have now.
                    x_adv[batch_index_1:batch_index_2] = self._generate_batch(
                        x=batch_x,
                        y=batch_targets,
                        mask=batch_mask,
                    )
                else:
                    batch_x_adv = self._generate_batch(
                        x=batch_x,
                        y=batch_targets,
                        mask=batch_mask
                    )

                    # Return the successful adversarial examples.
                    attack_success = compute_success_array(
                        self.estimator,
                        batch_x,
                        batch_targets,
                        batch_x_adv,
                        self.targeted,
                        batch_size=self.batch_size,
                    )
                    x_adv[batch_index_1:batch_index_2][attack_success] = batch_x_adv[attack_success]

        logger.info(
            "Success rate of attack: %.2f%%",
            100 * compute_success(self.estimator, x, targets, x_adv, self.targeted, batch_size=self.batch_size),
        )

        if self.summary_writer is not None:
            self.summary_writer.reset()

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
        import kornia

        x = x.to(self.estimator.device)
        y = y.to(self.estimator.device)

        if self.num_random_init > 0:
            # Initialise factors to random values sampled from [factor_min, factor_max].
            f_np = np.random.uniform(self.factor_min, self.factor_max, size=x.size(0))
        else:
            # Initialise factors to 0 i.e., no perturbation.
            f_np = np.zeros(shape=x.size(0))

        f_init = torch.asarray(f_np, dtype=torch.float32, device=x.device)
        if mask is not None:
            f_init = torch.where(mask == 0.0, torch.tensor(0.0).to(self.estimator.device), f_init)

        # Start to compute adversarial factors.
        f_adv = f_init.clone().detach()

        x_adv = kornia.enhance.adjust_hue(image=x, factor=f_adv)
        x_adv = self._clip_input(x_adv)

        if mask is not None:
            mask = mask.to(self.estimator.device)

        for i_max_iter in range(self.max_iter):
            self._i_max_iter = i_max_iter
            x_adv, f_adv = self._compute_torch(x, y, mask, f_adv)

        return x_adv.cpu().detach().numpy()

    def _compute_torch(
            self,
            x: "torch.Tensor",
            y: "torch.Tensor",
            mask: "torch.Tensor",
            factor: "torch.Tensor"
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        Compute adversarial samples and hue factors for one iteration.

        :param x: Original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)`
        :param mask: A 1D array of masks defining which samples to perturb. Shape needs to be `(nb_samples,)`.
                     Samples for which the mask is zero will not be adversarially perturbed.
        :param factor: Current hue factors.
        :return: A tuple holding the current adversarial examples and the hue factors.
        """
        import kornia

        # Compute the current factor perturbation.
        f_perturbation = self._compute_factor_perturbation(factor, x, y, mask)

        # Apply the perturbation and clip.
        f_adv = self._apply_factor_perturbation(factor, f_perturbation)

        x_adv = kornia.enhance.adjust_hue(x, f_adv)
        x_adv = self._clip_input(x_adv)

        return x_adv, f_adv

    def _compute_factor_perturbation(
            self,
            factor: "torch.Tensor",
            x: "torch.Tensor",
            y: "torch.Tensor",
            mask: Optional["torch.Tensor"]
    ) -> "torch.Tensor":
        """
        Compute hue perturbations for the given batch of inputs.

        :param factor: Current hue factors.
        :param x: Original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)`.
        :param mask: A 1D array of masks defining which samples to perturb. Shape needs to be `(nb_samples,)`.
                     Samples for which the mask is zero will not be adversarially perturbed.
        :return: Hue perturbations.
        """
        import torch
        import kornia

        # Pick a small scalar to avoid division by 0.
        tol = 10e-8

        x = x.clone().detach().requires_grad_(True)
        f = factor.clone().detach().requires_grad_(True)

        # Compute the current adversarial examples by applying
        # the current hue factors to the original inputs.
        x = kornia.enhance.adjust_hue(x, f)
        x = self._clip_input(x)

        # Get the gradient of the loss w.r.t. factors; invert them if the attack is targeted.
        # Step 1: get the gradient of the loss w.r.t. x.
        x_grad = self.estimator.loss_gradient(x=x, y=y) * (1 - 2 * int(self.targeted))
        # Step 2: backprop the gradients from x to f.
        x.backward(x_grad)
        f_grad = f.grad.detach()
        f.grad = None

        # Write summary.
        if self.summary_writer is not None:  # pragma: no cover
            self.summary_writer.update(
                batch_id=self._batch_id,
                global_step=self._i_max_iter,
                patch=None,
                estimator=self.estimator,
                factor=f.cpu().detach().numpy(),
                factor_grad=f_grad.cpu().detach().numpy(),
                x=x.cpu().detach().numpy(),
                y=y.cpu().detach().numpy(),
                targeted=self.targeted,
            )

        # Check for NaN before normalisation and replace with 0.
        if torch.any(f_grad.isnan()):  # pragma: no cover
            logger.warning("Elements of the loss gradient are NaN and have been replaced with 0.0.")
            f_grad[f_grad.isnan()] = 0.0

        # Apply the mask.
        if mask is not None:
            f_grad = torch.where(mask == 0.0, torch.tensor(0.0).to(self.estimator.device), f_grad)

        # Apply the norm bound.
        if self.norm in ["inf", np.inf]:
            f_grad = f_grad.sign()

        elif self.norm == 1:
            ind = tuple(range(1, len(x.shape)))
            f_grad = f_grad / (torch.sum(f_grad.abs(), dim=ind, keepdims=True) + tol)  # type: ignore

        elif self.norm == 2:
            ind = tuple(range(1, len(x.shape)))
            f_grad = f_grad / (torch.sqrt(torch.sum(f_grad * f_grad, axis=ind, keepdims=True)) + tol)  # type: ignore

        return f_grad

    def _apply_factor_perturbation(
            self,
            factor: "torch.Tensor",
            factor_perturbation: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Apply perturbations to the hue factors.

        :param factor: Current hue factors.
        :param factor_perturbation: Current hue perturbations.
        :return: Updated hue factors.
        """
        import torch

        step_size = np.array(self.step_size, dtype=ART_NUMPY_DTYPE)
        f_perturbation_step = torch.tensor(step_size).to(self.estimator.device) * factor_perturbation
        f_perturbation_step[torch.isnan(f_perturbation_step)] = 0
        f = torch.clamp(factor + f_perturbation_step, min=self.factor_min, max=self.factor_max)

        return f
