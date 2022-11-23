# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2022
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
#
# MIT License
#
# Copyright (c) 2022 University of Michigan and University of Wisconsin-Madison
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements the white-box PyTorch version of the GRAPHITE attack `GRAPHITEWhiteboxPyTorch`.
This is a robust physical perturbation attack.

| Paper link: https://arxiv.org/abs/2002.07088
| Original github link: https://github.com/ryan-feng/GRAPHITE
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, Tuple, Union, TYPE_CHECKING, List

import numpy as np

from art.config import ART_NUMPY_DTYPE
from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator
from art.estimators.classification import ClassifierMixin
from art.utils import compute_success, to_categorical, check_and_transform_label_format, is_probability
from art.attacks.evasion.graphite.utils import convert_to_network, get_transform_params, transform_wb

if TYPE_CHECKING:
    from art.estimators.classification.pytorch import PyTorchClassifier
    import torch

logger = logging.getLogger(__name__)


class GRAPHITEWhiteboxPyTorch(EvasionAttack):
    """
    Implementation of the white-box PyTorch GRAPHITE attack from Feng et al. (2022). This is a physical attack
    that generates robust physical perturbations that can be applied as stickers.

    | Paper link: https://arxiv.org/abs/2002.07088
    | Original github link: https://github.com/ryan-feng/GRAPHITE
    """

    attack_params = EvasionAttack.attack_params + [
        "net_size",
        "min_tr",
        "num_xforms",
        "step_size",
        "steps",
        "first_steps",
        "patch_removal_size",
        "patch_removal_interval",
        "num_patches_to_remove",
        "rand_start_epsilon_range",
        "rotation_range",
        "dist_range",
        "gamma_range",
        "crop_percent_range",
        "off_x_range",
        "off_y_range",
        "blur_kernels",
        "batch_size",
    ]
    _estimator_requirements = (BaseEstimator, ClassifierMixin)

    def __init__(
        self,
        classifier: "PyTorchClassifier",
        net_size: Tuple[int, int],
        min_tr: float = 0.8,
        num_xforms: int = 100,
        step_size: float = 0.0157,
        steps: int = 50,
        first_steps: int = 500,
        patch_removal_size: float = 4,
        patch_removal_interval: float = 2,
        num_patches_to_remove: int = 4,
        rand_start_epsilon_range: Tuple[float, float] = (-8 / 255, 8 / 255),
        rotation_range: Tuple[float, float] = (-30.0, 30.0),
        dist_range: Tuple[float, float] = (0.0, 0.0),
        gamma_range: Tuple[float, float] = (1.0, 2.0),
        crop_percent_range: Tuple[float, float] = (-0.03125, 0.03125),
        off_x_range: Tuple[float, float] = (-0.03125, 0.03125),
        off_y_range: Tuple[float, float] = (-0.03125, 0.03125),
        blur_kernels: Union[Tuple[int, int], List[int]] = (0, 3),
        batch_size: int = 64,
    ) -> None:
        """
        Create a GRAPHITEWhiteboxPyTorch attack instance.

        :param classifier: A trained classifier.
        :param net_size: The resolution to resize images to before feeding to the model in (w, h).
        :param min_tr: minimum threshold for EoT PGD to reach.
        :param num_xforms: The number of transforms to use.
        :param step_size: The step size.
        :param steps: The number of steps for EoT PGD after the first iteration.
        :param first_steps: The number of steps for EoT PGD for the first iteration.
        :param patch_removal_size: size of patch removal.
        :param patch_removal_interval: stride for patch removal.
        :param num_patches_to_remove: the number of patches to remove per iteration.
        :param rand_start_epsilon_range: the range for random start init.
        :param rotation_range: The range of the rotation in the perspective transform.
        :param dist_range: The range of the dist (in ft) to be added to the focal length in the perspective transform.
        :param gamma_range: The range of the gamma in the gamma transform.
        :param crop_percent_range: The range of the crop percent in the perspective transform.
        :param off_x_range: The range of the x offset (percent) in the perspective transform.
        :param off_y_range: The range of the y offset (percent) in the perspective transform.
        :param blur_kernels: The kernels to blur with.
        :param batch_size: The size of the batch used by the estimator during inference.
        """
        super().__init__(estimator=classifier)
        self.net_size = net_size
        self.min_tr = min_tr
        self.num_xforms = num_xforms
        self.step_size = step_size
        self.steps = steps
        self.first_steps = first_steps
        self.patch_removal_size = patch_removal_size
        self.patch_removal_interval = patch_removal_interval
        self.num_patches_to_remove = num_patches_to_remove
        self.rand_start_epsilon_range = rand_start_epsilon_range
        self.rotation_range = rotation_range
        self.dist_range = dist_range
        self.gamma_range = gamma_range
        self.crop_percent_range = crop_percent_range
        self.off_x_range = off_x_range
        self.off_y_range = off_y_range
        self.blur_kernels = blur_kernels
        self.batch_size = batch_size

        self._check_params()

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs to be attacked.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,).
        :param mask: An array with a mask broadcastable to input `x` defining where to apply adversarial perturbations.
                     Shape needs to be broadcastable to the shape of x and can also be of the same shape as `x`. Any
                     features for which the mask is zero will not be adversarially perturbed.
        :param pts: Optional points to consider when cropping the perspective transform.
                    An array of points in [x, y, scale] with shape [num points, 3, 1].
        :param obj_width: The estimated object width (inches) for perspective transform. 30 by default.
        :param focal: The estimated focal length (ft) for perspective transform. 3 by default.
        :return: An array holding the adversarial examples.
        """
        mask = kwargs.get("mask")
        obj_width = kwargs.get("obj_width") if "obj_width" in kwargs else 30
        focal = kwargs.get("focal") if "focal" in kwargs else 3
        pts = kwargs.get("pts") if "pts" in kwargs else None

        if y is None:
            raise ValueError("Target labels `y` need to be provided.")

        y = check_and_transform_label_format(y, self.estimator.nb_classes)

        if self.estimator.nb_classes == 2 and y.shape[1] == 1:  # pragma: no cover
            raise ValueError(
                "This attack has not yet been tested for binary classification with a single output classifier."
            )

        if not isinstance(obj_width, int) and not isinstance(obj_width, float):
            raise ValueError("obj_width must be int or float")
        obj_width = float(obj_width)

        if not isinstance(focal, int) and not isinstance(focal, float):
            raise ValueError("focal must be int or float")
        focal = float(focal)

        # Check the mask
        if mask is not None:
            if len(mask.shape) == len(x.shape):
                mask = mask.astype(ART_NUMPY_DTYPE)
            else:
                mask = np.array([mask.astype(ART_NUMPY_DTYPE)] * x.shape[0])
        else:
            mask = np.ones((x.shape))

        # Get clip_min and clip_max from the classifier or infer them from data
        if self.estimator.clip_values is not None:
            clip_min, clip_max = self.estimator.clip_values
        else:
            clip_min, clip_max = np.min(x), np.max(x)

        # Some initial setups
        x_adv = x.astype(ART_NUMPY_DTYPE)

        if not self.estimator.channels_first:
            x = np.transpose(x, (0, 3, 1, 2))
            x_adv = np.transpose(x_adv, (0, 3, 1, 2))
            mask = np.transpose(mask, (0, 3, 1, 2))

        y_pred = self.estimator.predict(
            np.transpose(
                convert_to_network(np.transpose(x_adv[0], (1, 2, 0)), self.net_size, clip_min, clip_max)[
                    np.newaxis, :, :, :
                ],
                (0, 3, 1, 2),
            )
        )
        if is_probability(y_pred):
            self.use_logits = False
        else:
            self.use_logits = True

        # Generate the adversarial samples
        for i in range(x_adv.shape[0]):
            x_adv[i] = self._perturb(
                x=x_adv[i],
                y=y[i],
                mask=mask[i],
                obj_width=obj_width,
                focal=focal,
                clip_min=clip_min,
                clip_max=clip_max,
                pts=pts,
            )

        y = np.argmax(y, axis=1)
        y = to_categorical(y, self.estimator.nb_classes)  # type: ignore

        # COMPUTE SUCCESS RATE
        x_copy = np.zeros((x.shape[0], self.net_size[1], self.net_size[0], x.shape[1]))
        x_adv_copy = np.zeros((x_adv.shape[0], self.net_size[1], self.net_size[0], x_adv.shape[1]))
        x_copy_trans = np.transpose(x, (0, 2, 3, 1))
        x_adv_copy_trans = np.transpose(x_adv, (0, 2, 3, 1))
        for i in range(x_copy.shape[0]):
            x_copy[i] = convert_to_network(x_copy_trans[i], self.net_size, clip_min, clip_max)
            x_adv_copy[i] = convert_to_network(x_adv_copy_trans[i], self.net_size, clip_min, clip_max)

        if self.estimator.channels_first:
            x_copy = np.transpose(x_copy, (0, 3, 1, 2))
            x_adv_copy = np.transpose(x_adv_copy, (0, 3, 1, 2))

        logger.info(
            "Success rate of GRAPHITE white-box attack: %.2f%%",
            100
            * compute_success(
                self.estimator,
                x_copy.astype(np.float32),
                y,
                x_adv_copy.astype(np.float32),
                self.targeted,
                batch_size=self.batch_size,
            ),
        )

        if not self.estimator.channels_first:
            x_adv = np.transpose(x_adv, (0, 2, 3, 1))

        return x_adv

    def _eval(
        self,
        x: "torch.Tensor",
        x_adv: "torch.Tensor",
        mask: "torch.Tensor",
        target_label: np.ndarray,
        y_onehot: "torch.Tensor",
        xforms: List[Tuple[float, float, float, int, float, float, float, float, float]],
        clip_min: float,
        clip_max: float,
        pts: Optional[np.ndarray],
    ) -> float:
        """
        Compute transform-robustness.

        :param x: Original image.
        :param x_adv: Attacked image.
        :param mask: The mask.
        :param target_label: The target label.
        :param y_onehot: The target label in one hot form.
        :param xforms: Ths list of transformation parameters.
        :param clip_min: Minimum value of an example.
        :param clip_max: Maximum value of an example.
        :param pts: Optional. A set of points that will set the crop size in the perspective transform.
        :return: Transform-robustness of the attack.
        """
        import torch

        successes = 0
        for xform in xforms:
            with torch.no_grad():
                if len(x_adv.shape) == 3:
                    x_adv = x_adv.unsqueeze(0)
                transformed_x_adv = transform_wb(x, x_adv, mask, xform, self.net_size, clip_min, clip_max, pts)
                logits, _ = self.estimator._predict_framework(  # pylint: disable=W0212
                    transformed_x_adv.to(self.estimator.device), y_onehot
                )
                success = int(logits.argmax(dim=1).detach().cpu().numpy()[0] == target_label)
                successes += success
        return successes / len(xforms)

    def _perturb(
        self,
        x: np.ndarray,
        y: np.ndarray,
        mask: np.ndarray,
        obj_width: float,
        focal: float,
        clip_min: float,
        clip_max: float,
        pts: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Internal attack function for one example.

        :param x: An array with one original input to be attacked.
        :param y: The target label.
        :param mask: An array with a mask to be applied to the adversarial perturbations. Shape needs to be
                     broadcastable to the shape of x. Any features for which the mask is zero will not be adversarially
                     perturbed.
        :param obj_width: Estimated width of object in inches for perspective transform.
        :param focal: Estimated focal length in ft for perspective transform.
        :param clip_min: Minimum value of an example.
        :param clip_max: Maximum value of an example.
        :param pts: Optional. A set of points that will set the crop size in the perspective transform.
        :return: An adversarial example.
        """
        import torch

        x = (x.copy() - clip_min) / (clip_max - clip_min)

        mask = mask / np.max(mask)
        mask = np.where(mask > 0.5, 1.0, 0.0)

        x_copy = x.copy()
        mask_copy = mask.copy()
        y_onehot = y.copy()
        y = np.argmax(y, axis=0)

        # Load victim img and starting mask.
        img = torch.tensor(x_copy, requires_grad=True, device=self.estimator.device)
        mask_tensor = torch.tensor(mask_copy, requires_grad=True, device=self.estimator.device).to(img.dtype)
        y_onehot_tensor = torch.tensor(y_onehot, requires_grad=True, device=self.estimator.device)

        # Load transforms.
        xforms = get_transform_params(
            self.num_xforms,
            self.rotation_range,
            self.dist_range,
            self.gamma_range,
            self.crop_percent_range,
            self.off_x_range,
            self.off_y_range,
            self.blur_kernels,
            obj_width,
            focal,
        )

        target_label = y
        # Attack
        rounds = 0
        transform_robustness = -1.0
        prev_attack = img.detach().clone()
        while True:
            adv_img = img.detach().clone()
            adv_img = torch.where(mask_tensor > 0.5, prev_attack, adv_img)
            rand_start = (
                torch.FloatTensor(*img.size())
                .uniform_(self.rand_start_epsilon_range[0], self.rand_start_epsilon_range[1])
                .to(adv_img.device)
            )
            adv_img = (adv_img + mask_tensor * rand_start).detach()
            adv_img = torch.clamp(adv_img, 0, 1).to(img.dtype)
            adv_img.requires_grad = True
            final_avg_grad = torch.zeros(img.size())

            # Do EOT adversarial attack with current mask.
            loop_length = self.steps if rounds > 0 else self.first_steps
            for _ in range(loop_length):
                avg_grad = torch.zeros(adv_img.size()).to(self.estimator.device)
                for xform in xforms:
                    xform_img = transform_wb(
                        img.clone().unsqueeze(0),
                        adv_img.unsqueeze(0),
                        mask_tensor,
                        xform,
                        self.net_size,
                        clip_min,
                        clip_max,
                        pts,
                    )

                    logits, _ = self.estimator._predict_framework(  # pylint: disable=W0212
                        xform_img.to(self.estimator.device), y_onehot_tensor
                    )
                    if self.use_logits:
                        loss = torch.nn.functional.cross_entropy(
                            input=logits,
                            target=torch.tensor(target_label).unsqueeze(0).to(logits.device),
                            reduction="mean",
                        )
                    else:
                        loss = torch.nn.functional.nll_loss(
                            input=logits, target=torch.tensor(target_label).unsqueeze(0), reduction="mean"
                        )

                    grad = torch.autograd.grad(loss, adv_img)[0]
                    avg_grad += grad
                avg_grad /= len(xforms)

                avg_grad_sign = avg_grad.clone()
                avg_grad_sign = torch.sign(avg_grad_sign)
                avg_grad_sign[torch.isnan(avg_grad_sign)] = 0
                adv_img = adv_img - mask_tensor * self.step_size * avg_grad_sign

                adv_img = adv_img.clamp(0, 1)
                transform_robustness = self._eval(
                    img.detach().clone(),
                    adv_img,
                    mask_tensor,
                    target_label,
                    y_onehot_tensor,
                    xforms,
                    clip_min,
                    clip_max,
                    pts,
                )
                final_avg_grad = avg_grad
                if transform_robustness >= self.min_tr:
                    break

            if transform_robustness < self.min_tr:
                break

            prev_attack = adv_img.detach().clone()

            # Remove low-impact pixel-patches or pixels in mask.
            pert = adv_img - img
            final_avg_grad[torch.isnan(final_avg_grad)] = 0
            final_avg_grad = mask_tensor * final_avg_grad * pert
            pixelwise_avg_grads = torch.sum(torch.abs(final_avg_grad), dim=0)

            for _ in range(self.num_patches_to_remove):
                # Find minimum gradient patch and remove it.
                min_patch_grad = None
                min_patch_grad_idx = None
                for i in np.arange(
                    0, pixelwise_avg_grads.shape[0] - self.patch_removal_size + 0.0001, self.patch_removal_interval
                ):
                    for j in np.arange(
                        0, pixelwise_avg_grads.shape[1] - self.patch_removal_size + 0.0001, self.patch_removal_interval
                    ):
                        patch_grad = pixelwise_avg_grads[
                            int(round(i)) : int(round(i + self.patch_removal_size)),
                            int(round(j)) : int(round(j + self.patch_removal_size)),
                        ].sum()

                        if (
                            mask_tensor[
                                0,
                                int(round(i)) : int(round(i + self.patch_removal_size)),
                                int(round(j)) : int(round(j + self.patch_removal_size)),
                            ].sum()
                            > 0
                        ):
                            patch_grad = (
                                patch_grad
                                / mask_tensor[
                                    0,
                                    int(round(i)) : int(round(i + self.patch_removal_size)),
                                    int(round(j)) : int(round(j + self.patch_removal_size)),
                                ].sum()
                            )  # TODO1
                            if min_patch_grad is None or patch_grad.item() < min_patch_grad:
                                min_patch_grad = patch_grad.item()
                                min_patch_grad_idx = (i, j)
                if min_patch_grad_idx is None:
                    break
                i, j = min_patch_grad_idx
                mask_tensor[
                    0,
                    int(round(i)) : int(round(i + self.patch_removal_size)),
                    int(round(j)) : int(round(j + self.patch_removal_size)),
                ] = 0
                if img.size()[0] == 3:
                    mask_tensor[
                        1,
                        int(round(i)) : int(round(i + self.patch_removal_size)),
                        int(round(j)) : int(round(j + self.patch_removal_size)),
                    ] = 0
                    mask_tensor[
                        2,
                        int(round(i)) : int(round(i + self.patch_removal_size)),
                        int(round(j)) : int(round(j + self.patch_removal_size)),
                    ] = 0

            rounds += 1

        out = prev_attack.detach().clone().cpu().numpy()
        adversarial = np.clip(out.copy() * (clip_max - clip_min) + clip_min, clip_min, clip_max)
        return adversarial

    def _check_params(self) -> None:
        if self.min_tr < 0 or self.min_tr > 1:
            raise ValueError("min_tr must be between 0 and 1.")

        if self.num_xforms < 0 or not isinstance(self.num_xforms, int):
            raise ValueError("num_xforms must be non-negative integer.")

        if self.step_size <= 0:
            raise ValueError("step size must be positive.")

        if self.steps <= 0 or not isinstance(self.steps, int):
            raise ValueError("steps must be a positive integer.")

        if self.first_steps <= 0 or not isinstance(self.first_steps, int):
            raise ValueError("first_steps must be a positive integer.")

        if self.patch_removal_size <= 0:
            raise ValueError("patch_removal_size must be positive.")

        if self.patch_removal_interval <= 0:
            raise ValueError("patch_removal_interval must be positive.")

        if self.num_patches_to_remove <= 0 or not isinstance(self.num_patches_to_remove, int):
            raise ValueError("num_patches_to_remove must be a positive integer.")

        if (
            self.rotation_range[0] <= -90
            or self.rotation_range[1] >= 90
            or self.rotation_range[1] < self.rotation_range[0]
        ):
            raise ValueError("rotation range must be within (-90, 90).")

        if self.dist_range[1] < self.dist_range[0] or self.dist_range[0] < 0:
            raise ValueError("distance range invalid. max must be greater than min, and must be nonnegative.")

        if self.gamma_range[1] < self.gamma_range[0] or self.gamma_range[0] < 1:
            raise ValueError("gamma range max must be greater than min and the range must be at 1.0 or greater.")

        if self.crop_percent_range[1] < self.crop_percent_range[0]:
            raise ValueError("max of crop percent range must be greater or equal to the min.")

        if self.off_x_range[1] < self.off_x_range[0]:
            raise ValueError("max of off x range must be greater or equal to the min.")

        if self.off_y_range[1] < self.off_y_range[0]:
            raise ValueError("max of off y range must be greater or equal to the min.")

        if min(self.blur_kernels) < 0:
            raise ValueError("blur kernels must be positive.")
