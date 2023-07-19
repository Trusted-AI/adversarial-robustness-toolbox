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
This module implements affine transformation attack by adversarially rotating, scaling, shearing, and/or translating
the inputs. It uses the iterative gradient sign method to optimise the transformation parameters (see
`FastGradientMethod` and `BasicIterativeMethod`). This implementation extends the rotation attack to all the affine
transformations. It also extends the optimisation method to other norms as well.

| Paper link: https://arxiv.org/abs/2202.04235
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, Dict, Tuple, Union, TYPE_CHECKING

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


class AffineGradientPyTorch(EvasionAttack):
    """
    Implementation of the affine transformation attack on image classifiers in PyTorch. The attack is constructed by
    adversarially rotating, scaling, shearing, and/or translating the inputs. It uses the iterative gradient sign method
    to optimise the transformation parameters (see `FastGradientMethod` and `BasicIterativeMethod`). This implementation
    extends the original rotation attack to all the affine transformations and also extends the optimisation method to
    other norms as well.

    Note that this attack is intended for only PyTorch image classifiers with RGB images in the range [0, 1] as inputs.

    | Paper link: https://arxiv.org/abs/2202.04235
    """

    attack_params = EvasionAttack.attack_params + [
        "norm",
        "angle_range",
        "scale_x_range",
        "scale_y_range",
        "shear_x_range",
        "shear_y_range",
        "translate_x_range",
        "translate_y_range",
        "center",
        "angle_step_size",
        "scale_x_step_size",
        "scale_y_step_size",
        "shear_x_step_size",
        "shear_y_step_size",
        "translate_x_step_size",
        "translate_y_step_size",
        "interpolation",
        "padding_mode",
        "align_corners",
        "fill_value",
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
        angle_range: Optional[Tuple[float, float]] = (-10.0, 10.0),
        scale_x_range: Optional[Tuple[float, float]] = (0.8, 1.2),
        scale_y_range: Optional[Tuple[float, float]] = (0.8, 1.2),
        shear_x_range: Optional[Tuple[float, float]] = (-10.0, 10.0),
        shear_y_range: Optional[Tuple[float, float]] = (-10.0, 10.0),
        translate_x_range: Optional[Tuple[float, float]] = (-2.0, 2.0),
        translate_y_range: Optional[Tuple[float, float]] = (-2.0, 2.0),
        center: Optional[Tuple[Union[int, float], Union[int, float]]] = None,
        angle_step_size: Optional[Union[int, float]] = None,
        scale_x_step_size: Optional[Union[int, float]] = None,
        scale_y_step_size: Optional[Union[int, float]] = None,
        shear_x_step_size: Optional[Union[int, float]] = None,
        shear_y_step_size: Optional[Union[int, float]] = None,
        translate_x_step_size: Optional[Union[int, float]] = None,
        translate_y_step_size: Optional[Union[int, float]] = None,
        interpolation: str = "bilinear",
        padding_mode: str = "zeros",
        fill_value: Tuple[float, float, float] = (0, 0, 0),
        align_corners: bool = True,
        max_iter: int = 10,
        targeted: bool = False,
        num_random_init: int = 0,
        batch_size: int = 32,
        summary_writer: Union[str, bool, SummaryWriter] = False,
        verbose: bool = True,
    ) -> None:
        """
        Create an instance of the :class:`.AffineGradientPyTorch`.

        :param classifier: A trained PyTorch classifier.
        :param norm: The norm of the adversarial perturbation. Possible values: `"inf"`, `np.inf`, `1` or `2`.
        :param angle_range: The range of rotation in the affine transformation in degrees. Positive values mean
                            counter-clockwise rotation. `None` means no perturbation.
        :param scale_x_range: The range of the scaling factor in the x-direction.
        :param scale_y_range: The range of the scaling factor in the y-direction.
        :param shear_x_range: The range of the shearing factor in the x-direction in degrees.
        :param shear_y_range: The range of the shearing factor in the y-direction in degrees.
        :param translate_x_range: The range of translation in the x-direction in pixels.
        :param translate_y_range: The range of translation in the y-direction in pixels.
        :param center: The center of rotation and shearing. Default is the center of the image.
        :param angle_step_size: Attack step size for rotation at each iteration. If `None`, the step_size is computed
                                using `2.5 * (range[1] - range[0]) / (2 * max_iter)`. See Appendix A. of the paper.
        :param scale_x_step_size: Attack step size for x-axis scaling at each iteration.
        :param scale_y_step_size: Attack step size for y-axis scaling at each iteration.
        :param shear_x_step_size: Attack step size for x-axis shearing at each iteration.
        :param shear_y_step_size: Attack step size for y-axis shearing at each iteration.
        :param translate_x_step_size: Attack step size for x-axis translation at each iteration.
        :param translate_y_step_size: Attack step size for y-axis translation at each iteration.
        :param interpolation: The interpolation method to use for calculating the output values. Possible values:
                              `"bilinear"`, `"nearest"`, or `"bicubic"`.
        :param padding_mode: The padding mode for outside grid values. Possible values: `"zeros"`, `"border"`,
                             `"reflection"`, or `"fill"`.
        :param fill_value: Value to fill in the padding area. The value is expected to be RGB in the range of `[0, 1]`.
                           This only has an effect when `padding_mode` is `"fill"`.
        :param align_corners: Mode for grid_generation. See `torch.nn.functional.grid_sample` for more details.
        :param max_iter: The maximum number of iterations.
        :param targeted: Indicates whether the attack is targeted (`True`) or untargeted (`False`).
        :param num_random_init: Number of random initialisations to try for the affine transformation parameters within
                                their respective intervals. For `num_random_init=0`, the attack starts at the original
                                input.
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
        self.center = center
        self.ranges = {
            "angle": angle_range,
            "scale_x": scale_x_range,
            "scale_y": scale_y_range,
            "shear_x": shear_x_range,
            "shear_y": shear_y_range,
            "translate_x": translate_x_range,
            "translate_y": translate_y_range
        }
        self.step_sizes = {
            "angle": angle_step_size,
            "scale_x": scale_x_step_size,
            "scale_y": scale_y_step_size,
            "shear_x": shear_x_step_size,
            "shear_y": shear_y_step_size,
            "translate_x": translate_x_step_size,
            "translate_y": translate_y_step_size
        }
        self.interpolation = interpolation
        self.padding_mode = padding_mode
        self.fill_value = fill_value
        self.align_corners = align_corners
        self.max_iter = max_iter
        self.targeted = targeted
        self.num_random_init = num_random_init
        self.batch_size = batch_size
        self.verbose = verbose

        self._check_params()

        # Convert degrees to radians as kornia uses radians for shearing.
        if self.ranges["shear_x"] is not None:
            self.ranges["shear_x"] = (self.ranges["shear_x"][0] * np.pi / 180.0,
                                      self.ranges["shear_x"][1] * np.pi / 180.0)
        if self.ranges["shear_y"] is not None:
            self.ranges["shear_y"] = (self.ranges["shear_y"][0] * np.pi / 180.0,
                                      self.ranges["shear_y"][1] * np.pi / 180.0)

        self._description = "Affine Attack"
        self._batch_id = 0
        self._i_max_iter = 0

    def _check_params(self) -> None:

        if self.norm not in [1, 2, np.inf, "inf"]:
            raise ValueError('Norm order must be either 1, 2, `np.inf` or "inf".')

        for key, value in self.ranges.items():
            if value is not None and not (isinstance(value, tuple) and len(value) == 2):
                raise ValueError(f"The argument `{key}_range` must be `None` or a tuple of min and max values.")
            if isinstance(value, tuple):
                if not (isinstance(value[0], float) and isinstance(value[1], float)):
                    raise ValueError(f"The argument `{key}_range` must be a tuple of floats.")
                if value[0] >= value[1]:
                    raise ValueError(f"The first element of `{key}_range` must be less than the second element.")

        if self.center is not None and not (isinstance(self.center, tuple) and len(self.center) == 2):
            raise ValueError("The argument `center` must be `None` or a tuple of x and y values.")
        if isinstance(self.center, tuple):
            if not (isinstance(self.center[0], (int, float)) and isinstance(self.center[1], (int, float))):
                raise ValueError("The argument `center` must be a tuple of ints or floats.")

        for key, step_size in self.step_sizes.items():
            if step_size is not None and not (isinstance(step_size, (int, float)) and step_size > 0.0):
                raise ValueError(f"The argument `{key}_step_size` must be positive of type int or float or `None`.")

        if not isinstance(self.interpolation, str):
            raise TypeError("The argument `interpolation` must be a string.")
        if self.interpolation not in ["bilinear", "nearest", "bicubic"]:
            raise ValueError("The argument `interpolation` must be 'bilinear', 'nearest', or 'bicubic'.")

        if not isinstance(self.padding_mode, str):
            raise TypeError("The argument `padding_mode` must be a string.")
        if self.padding_mode not in ["zeros", "border", "reflection", "fill"]:
            raise ValueError("The argument `padding_mode` must be  'zeros', 'border', 'reflection', or 'fill'.")

        if not (isinstance(self.fill_value, tuple) and len(self.fill_value) == 3):
            raise ValueError("The argument `fill_value` must be a tuple of RGB values.")
        if not (0.0 <= self.fill_value[0] <= 1.0 and 0.0 <= self.fill_value[1] <= 1.0 and
                0.0 <= self.fill_value[2] <= 1.0):
            raise ValueError("The argument `fill_value` must be RGB values in the range of `[0, 1]`.")

        if not isinstance(self.align_corners, bool):
            raise ValueError("The flag `align_corners` has to be of type bool.")

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

            if not mask.shape == (x.shape[0],):  # pragma: no cover
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

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
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
            tqdm(data_loader, desc=self._description, leave=False, disable=not self.verbose)
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

    def _init_factors(self, x: "torch.Tensor", mask: "torch.Tensor") -> Dict[str, "torch.Tensor"]:
        """
        Initialise the affine transformation parameters: angle, scales, shears, and translations.

        :param x: Original inputs.
        :param mask: A 1D array of masks defining which samples to perturb. Shape needs to be `(nb_samples,)`.
                     Samples for which the mask is zero will not be adversarially perturbed.
        :return: Dictionary of initial affine transformation parameters.
        """
        import torch

        f_shape = torch.Size((x.size(0),))

        factors = {}
        for key, f_range in self.ranges.items():
            if self.num_random_init > 0 and f_range is not None:
                # Initialise the factor to a random value sampled from [min, max].
                f_init = torch.distributions.uniform.Uniform(f_range[0], f_range[1]).sample(f_shape)
            else:
                # Initialise the factor to the no perturbation value.
                if key not in ["scale_x", "scale_y"]:
                    f_init = torch.zeros(f_shape)
                else:
                    f_init = torch.ones(f_shape)
            f_init = torch.asarray(f_init, dtype=torch.float32, device=self.estimator.device)

            if mask is not None:
                if key not in ["scale_x", "scale_y"]:
                    f_init = torch.where(mask == 0.0, torch.tensor(0.0).to(self.estimator.device), f_init)
                else:
                    f_init = torch.where(mask == 0.0, torch.tensor(1.0).to(self.estimator.device), f_init)

            factors[key] = f_init

        return factors

    def _warp_affine(self, x: "torch.Tensor", factors: Dict[str, "torch.Tensor"]) -> "torch.Tensor":
        """
        Perform the affine transformation of the inputs.

        :param x: Original inputs.
        :param factors: Dictionary of the affine transformation parameters.
        :return: Warped inputs.
        """
        import torch
        import kornia

        batch_size, _, height, width = x.size()

        if self.center is None:
            center = torch.asarray((width / 2, height / 2), dtype=torch.float32, device=self.estimator.device)
        else:
            center = torch.asarray(self.center, dtype=torch.float32, device=self.estimator.device)
        center = center.unsqueeze(0)
        center = center.repeat_interleave(repeats=batch_size, dim=0)

        angle = factors["angle"]
        scale = torch.stack([factors["scale_x"], factors["scale_y"]], dim=-1)
        shear_x = factors["shear_x"]
        shear_y = factors["shear_y"]
        translate = torch.stack([factors["translate_x"], factors["translate_y"]], dim=-1)

        M = kornia.geometry.get_affine_matrix2d(translate, center, scale, angle, shear_x, shear_y)
        M = M[:, :2, :]

        x_adv = kornia.geometry.warp_affine(
            x,
            M,
            dsize=(height, width),
            mode=self.interpolation,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
            fill_value=torch.asarray(self.fill_value)
        )

        return x_adv

    def _generate_batch(self, x: "torch.Tensor", y: "torch.Tensor", mask: "torch.Tensor") -> np.ndarray:
        """
        Generate a batch of adversarial samples and return them in a NumPy array.

        :param x: Original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)`.
        :param mask: A 1D array of masks defining which samples to perturb. Shape needs to be `(nb_samples,)`.
                     Samples for which the mask is zero will not be adversarially perturbed.
        :return: Adversarial examples.
        """
        x = x.to(self.estimator.device)
        y = y.to(self.estimator.device)
        if mask is not None:
            mask = mask.to(self.estimator.device)

        # Start to compute adversarial factors.
        f_adv = self._init_factors(x, mask)

        x_adv = self._warp_affine(x, f_adv)
        x_adv = self._clip_input(x_adv)

        for i_max_iter in range(self.max_iter):
            self._i_max_iter = i_max_iter
            x_adv, f_adv = self._compute(x, y, mask, f_adv)

        return x_adv.cpu().detach().numpy()

    def _compute(
            self,
            x: "torch.Tensor",
            y: "torch.Tensor",
            mask: "torch.Tensor",
            factors: Dict[str, "torch.Tensor"]
    ) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:
        """
        Compute adversarial samples and affine transformation parameters for one iteration.

        :param x: Original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)`
        :param mask: A 1D array of masks defining which samples to perturb. Shape needs to be `(nb_samples,)`.
                     Samples for which the mask is zero will not be adversarially perturbed.
        :param factors: Current affine transformation parameters.
        :return: A tuple holding the current adversarial examples and affine transformation parameters.
        """
        # Compute the current perturbation.
        f_perturbation = self._compute_factors_perturbation(factors, x, y, mask)

        # Apply the perturbation and clip.
        f_adv = self._update_factors(factors, f_perturbation)

        x_adv = self._warp_affine(x, f_adv)
        x_adv = self._clip_input(x_adv)

        return x_adv, f_adv

    def _compute_factors_perturbation(
            self,
            factors: Dict[str, "torch.Tensor"],
            x: "torch.Tensor",
            y: "torch.Tensor",
            mask: Optional["torch.Tensor"]
    ) -> Dict[str, Union[None, "torch.Tensor"]]:
        """
        Compute affine transformation perturbations for the given batch of inputs.

        :param factors: Current affine transformation parameters.
        :param x: Original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)`.
        :param mask: A 1D array of masks defining which samples to perturb. Shape needs to be `(nb_samples,)`.
                     Samples for which the mask is zero will not be adversarially perturbed.
        :return: Dictionary of perturbations.
        """
        import torch

        # Pick a small scalar to avoid division by 0.
        tol = 10e-8

        x = x.clone().detach().requires_grad_(True)
        for key, f in factors.items():
            if self.ranges[key] is None:
                # This affine parameter will not be perturbed.
                factors[key] = f.clone().detach().requires_grad_(False)
            else:
                factors[key] = f.clone().detach().requires_grad_(True)

        # Compute the current adversarial examples by applying
        # the current affine parameters to the original inputs.
        x = self._warp_affine(x, factors)
        x = self._clip_input(x)

        # Get the gradient of the loss w.r.t. affine parameters; invert them if the attack is targeted.
        # Step 1: get the gradient of the loss w.r.t. x.
        x_grad = self.estimator.loss_gradient(x=x, y=y) * (1 - 2 * int(self.targeted))
        # Step 2: backprop the gradients from x to factors.
        x.backward(x_grad)

        factors_grad: Dict[str, Union[None, "torch.Tensor"]] = {}
        for key, f in factors.items():
            if self.ranges[key] is None:
                # This affine parameter will not be perturbed.
                factors_grad[key] = None
            else:
                factors_grad[key] = f.grad.detach()  # type: ignore
            f.grad = None

        # Write summary.
        if self.summary_writer is not None:  # pragma: no cover
            self.summary_writer.update(
                batch_id=self._batch_id,
                global_step=self._i_max_iter,
                patch=None,
                estimator=self.estimator,
                x=x.cpu().detach().numpy(),
                y=y.cpu().detach().numpy(),
                targeted=self.targeted,
                angle=factors["angle"].cpu().detach().numpy(),
                scale_x=factors["scale_x"].cpu().detach().numpy(),
                scale_y=factors["scale_y"].cpu().detach().numpy(),
                shear_x=factors["shear_x"].cpu().detach().numpy(),
                shear_y=factors["shear_y"].cpu().detach().numpy(),
                translate_x=factors["translate_x"].cpu().detach().numpy(),
                translate_y=factors["translate_y"].cpu().detach().numpy(),
            )

        for key, f_grad in factors_grad.items():
            if f_grad is not None:
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
                    ind = tuple(range(1, len(f_grad.shape)))
                    f_grad = f_grad / (torch.sum(f_grad.abs(), dim=ind, keepdims=True) + tol)  # type: ignore

                elif self.norm == 2:
                    ind = tuple(range(1, len(f_grad.shape)))
                    f_grad =\
                        f_grad / (torch.sqrt(torch.sum(f_grad * f_grad, axis=ind, keepdims=True)) + tol)  # type: ignore

            factors_grad[key] = f_grad

        return factors_grad

    def _update_factors(
            self,
            factors: Dict[str, "torch.Tensor"],
            factors_perturbation: Dict[str, Union[None, "torch.Tensor"]],
    ) -> Dict[str, "torch.Tensor"]:
        """
        Apply the perturbations to the affine transformation parameters.

        :param factors: Dictionary of the current affine transformation parameters.
        :param factors_perturbation: Dictionary of the current perturbations.
        :return: Dictionary of updated affine transformation parameters.
        """
        import torch

        for key, f in factors.items():
            f_perturbation = factors_perturbation[key]
            f_range = self.ranges[key]
            step_size = self.step_sizes[key]

            if f_range is not None and f_perturbation is not None:
                if step_size is None:
                    step_size = 2.5 * (f_range[1] - f_range[0]) / (2 * self.max_iter)

                f_perturbation_step =\
                    torch.tensor(step_size, dtype=torch.float32, device=self.estimator.device) * f_perturbation
                f_perturbation_step[torch.isnan(f_perturbation_step)] = 0

                f = torch.clamp(f + f_perturbation_step, min=f_range[0], max=f_range[1])

            factors[key] = f

        return factors
