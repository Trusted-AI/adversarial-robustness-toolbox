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
This module implements the black-box (hard-label) GRAPHITE attack `GRAPHITEBlackbox`. This is a physical black-box
attack that only requires class predictions.

| Paper link: https://arxiv.org/abs/2002.07088
| Original github link: https://github.com/ryan-feng/GRAPHITE
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, Tuple, Union, TYPE_CHECKING, List

import random
import numpy as np

from art.config import ART_NUMPY_DTYPE
from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator
from art.estimators.classification import ClassifierMixin
from art.utils import compute_success, to_categorical, check_and_transform_label_format
from art.attacks.evasion.graphite.utils import (
    convert_to_network,
    get_transform_params,
    add_noise,
    get_transformed_images,
    run_predictions,
    score_fn,
)

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_NEURALNETWORK_TYPE

logger = logging.getLogger(__name__)


class GRAPHITEBlackbox(EvasionAttack):
    """
    Implementation of the hard-label GRAPHITE attack from Feng et al. (2022). This is a physical, black-box attack
    that only requires final class prediction and generates robust physical perturbations that can be applied as
    stickers.

    | Paper link: https://arxiv.org/abs/2002.07088
    | Original github link: https://github.com/ryan-feng/GRAPHITE
    """

    attack_params = EvasionAttack.attack_params + [
        "noise_size",
        "net_size",
        "heat_patch_size",
        "heat_patch_stride",
        "heatmap_mode",
        "tr_lo",
        "tr_hi",
        "num_xforms_mask",
        "max_mask_size",
        "rotation_range",
        "beta",
        "eta",
        "num_xforms_boost",
        "num_boost_queries",
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
        classifier: "CLASSIFIER_NEURALNETWORK_TYPE",
        noise_size: Tuple[int, int],
        net_size: Tuple[int, int],
        heat_patch_size: Tuple[int, int] = (4, 4),
        heat_patch_stride: Tuple[int, int] = (1, 1),
        heatmap_mode: str = "Target",
        tr_lo: float = 0.65,
        tr_hi: float = 0.85,
        num_xforms_mask: int = 100,
        max_mask_size: int = -1,
        beta: float = 1.0,
        eta: float = 500,
        num_xforms_boost: int = 100,
        num_boost_queries: int = 20000,
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
        Create a GRAPHITEBlackbox attack instance.

        :param classifier: A trained classifier.
        :param noise_size: The resolution to generate perturbations in (w, h).
        :param net_size: The resolution to resize images to before feeding to the model in (w, h).
        :param heat_patch_size: The size of the heatmap patches in (w, h).
        :param heat_patch_stride: The stride of the heatmap patching in (w, h).
        :param heatmap_mode: The mode of heatmap in ['Target', 'Random'].
        :param tr_lo: tr_lo, threshold for fine-grained reduction.
        :param tr_hi: tr_hi, threshold for coarse-grained reduction.
        :param num_xforms_mask: The number of transforms to use in mask generation.
        :param max_mask_size: Optionally specify that you just want to optimize until a mask size of <= max_mask_size.
        :param beta: The parameter beta for RGF optimization in boosting.
        :param eta: The step size for RGF optimization in boosting.
        :param num_xforms_boost: The number of transforms to use in boosting.
        :param num_boost_queries: The number of queries to use in boosting.
        :param rotation_range: The range of the rotation in the perspective transform.
        :param dist_range: The range of the distance (in ft) to be added to the focal length in perspective transform.
        :param gamma_range: The range of the gamma in the gamma transform.
        :param crop_percent_range: The range of the crop percent in the perspective transform.
        :param off_x_range: The range of the x offset (percent) in the perspective transform.
        :param off_y_range: The range of the y offset (percent) in the perspective transform.
        :param blur_kernels: The kernels to blur with.
        :param batch_size: The size of the batch used by the estimator during inference.
        """
        super().__init__(estimator=classifier)
        self.noise_size = noise_size
        self.net_size = net_size
        self.heat_patch_size = heat_patch_size
        self.heat_patch_stride = heat_patch_stride
        self.heatmap_mode = heatmap_mode
        self.batch_size = batch_size
        self.tr_lo = tr_lo
        self.tr_hi = tr_hi
        self.num_xforms_mask = num_xforms_mask
        self.max_mask_size = max_mask_size
        self.beta = beta
        self.eta = eta
        self.num_xforms_boost = num_xforms_boost
        self.num_boost_queries = num_boost_queries
        self.rotation_range = rotation_range
        self.dist_range = dist_range
        self.gamma_range = gamma_range
        self.crop_percent_range = crop_percent_range
        self.off_x_range = off_x_range
        self.off_y_range = off_y_range
        self.blur_kernels = blur_kernels

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
        :param x_tar: Initial array to act as the example target image.
        :param pts: Optional points to consider when cropping the perspective transform. An array of points in
                    [x, y, scale] with shape [num points, 3, 1].
        :param obj_width: The estimated object width (inches) for perspective transform. 30 by default.
        :param focal: The estimated focal length (ft) for perspective transform. 3 by default.
        :return: An array holding the adversarial examples.
        """
        mask = kwargs.get("mask")
        x_tar = kwargs.get("x_tar")
        obj_width = kwargs.get("obj_width") if "obj_width" in kwargs else 30
        focal = kwargs.get("focal") if "focal" in kwargs else 3
        pts = kwargs.get("pts") if "pts" in kwargs else None

        if y is None:
            raise ValueError("Target labels `y` need to be provided.")

        if x_tar is None:
            raise ValueError("Target image example `x_tar` needs to be provided.")

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
            mask = np.array([None] * x.shape[0])

        # Get clip_min and clip_max from the classifier or infer them from data
        if self.estimator.clip_values is not None:
            clip_min, clip_max = self.estimator.clip_values
        else:
            clip_min, clip_max = np.min(x), np.max(x)

        # target initialization image
        x_tar = kwargs.get("x_tar")

        # Some initial setups
        x_adv = x.astype(ART_NUMPY_DTYPE)

        if self.estimator.channels_first:
            x = np.transpose(x, (0, 2, 3, 1))
            x_adv = np.transpose(x_adv, (0, 2, 3, 1))
            x_tar = np.transpose(x_tar, (0, 2, 3, 1))
            if len(mask.shape) == 4:
                mask = np.transpose(mask, (0, 2, 3, 1))

        y = np.argmax(y, axis=1)

        # Generate the adversarial samples
        for i in range(x_adv.shape[0]):
            x_adv[i] = self._perturb(
                x=x_adv[i],
                y=y[i],  # type: ignore
                x_tar=x_tar[i],
                obj_width=obj_width,
                focal=focal,
                clip_min=clip_min,
                clip_max=clip_max,
                mask=mask[i],
                pts=pts,
            )

        y = to_categorical(y, self.estimator.nb_classes)  # type: ignore

        # COMPUTE SUCCESS RATE
        x_copy = np.zeros((x.shape[0], self.noise_size[1], self.noise_size[0], x.shape[3]))
        x_adv_copy = np.zeros((x_adv.shape[0], self.noise_size[1], self.noise_size[0], x_adv.shape[3]))
        for i in range(x_copy.shape[0]):
            x_copy[i] = convert_to_network(x[i], self.net_size, clip_min, clip_max)
            x_adv_copy[i] = convert_to_network(x_adv[i], self.net_size, clip_min, clip_max)

        if self.estimator.channels_first:
            x_copy = np.transpose(x_copy, (0, 3, 1, 2))
            x_adv_copy = np.transpose(x_adv_copy, (0, 3, 1, 2))

        logger.info(
            "Success rate of GRAPHITE hard-label attack: %.2f%%",
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

        if self.estimator.channels_first:
            x_adv = np.transpose(x_adv, (0, 3, 1, 2))

        return x_adv

    def _perturb(
        self,
        x: np.ndarray,
        y: int,
        x_tar: np.ndarray,
        obj_width: float,
        focal: float,
        clip_min: float,
        clip_max: float,
        mask: Optional[np.ndarray] = None,
        pts: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Internal attack function for one example.

        :param x: An array with one original input to be attacked.
        :param y: The target label.
        :param x_tar: Initial array to act as an example target image.
        :param obj_width: Estimated width of object in inches for perspective transform.
        :param focal: Estimated focal length in ft for perspective transform.
        :param clip_min: Minimum value of an example.
        :param clip_max: Maximum value of an example.
        :param mask: An array with a mask to be applied to the adversarial perturbations. Shape needs to be
                     broadcastable to the shape of x. Any features for which the mask is zero will not be adversarially
                     perturbed.
        :param pts: Optional. A set of points that will set the crop size in the perspective transform.
        :return: An adversarial example.
        """
        import cv2

        x = (x.copy() - clip_min) / (clip_max - clip_min)
        x_tar = (x_tar.copy() - clip_min) / (clip_max - clip_min)

        if mask is None:
            mask_array = np.ones((self.noise_size[1], self.noise_size[0], x.shape[2]))
        else:
            mask_array = mask
        mask_array = mask_array / np.max(mask_array)

        x_copy = x.copy()
        x_tar_copy = x_tar.copy()
        mask_copy = mask_array.copy()
        x_noise = cv2.resize(x_copy, self.noise_size)
        x_tar_noise = cv2.resize(x_tar_copy, self.noise_size)
        mask_noise = cv2.resize(mask_copy, self.noise_size).astype(float)
        mask_noise = np.where(mask_noise > 0.5, 1.0, 0.0)

        if len(x_noise.shape) < 3:
            x_noise = x_noise[:, :, np.newaxis]
        if len(x_tar_noise.shape) < 3:
            x_tar_noise = x_tar_noise[:, :, np.newaxis]
        if len(mask_noise.shape) < 3:
            mask_noise = mask_noise[:, :, np.newaxis]

        mask_out = self._generate_mask(
            x_copy, x_noise, x_tar_noise, mask_noise, y, obj_width, focal, clip_min, clip_max, pts
        )

        adversarial = self._boost(x_copy, x_noise, x_tar_noise, mask_out, y, obj_width, focal, clip_min, clip_max, pts)

        adversarial = np.clip(adversarial.copy() * (clip_max - clip_min) + clip_min, clip_min, clip_max)
        return adversarial

    def _generate_mask(
        self,
        x: np.ndarray,
        x_noise: np.ndarray,
        x_tar_noise: np.ndarray,
        mask: np.ndarray,
        y: int,
        obj_width: float,
        focal: float,
        clip_min: float,
        clip_max: float,
        pts: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Function to generate a mask.

        :param x: An array with one original input to be attacked.
        :param x_noise: x in the resolution of the noise size.
        :param x_tar_noise: x_tar in the resolution of the noise size.
        :param mask: An array with a mask to be applied to the adversarial perturbations. Shape needs to be
                     broadcastable to the shape of x. Any features for which the mask is zero will not be adversarially
                     perturbed.
        :param y: The target label.
        :param obj_width: Estimated width of object in inches for perspective transform.
        :param focal: Estimated focal length in ft for perspective transform.
        :param clip_min: Minimum value of an example.
        :param clip_max: Maximum value of an example.
        :param pts: Optional. A set of points that will set the crop size in the perspective transform.
        :return: A mask.
        """

        # Stage 1: HEATMAP: Collect all the valid patches and order by computed heatmap
        xforms = get_transform_params(
            self.num_xforms_mask,
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

        object_size = mask.sum() / mask.shape[-1]

        patch = np.ones((self.heat_patch_size[1], self.heat_patch_size[0], x.shape[-1]))
        patches = []
        indices = []

        # collect all valid patches
        for i in range(0, mask.shape[0] - self.heat_patch_size[1] + 1, self.heat_patch_stride[1]):
            for j in range(0, mask.shape[1] - self.heat_patch_size[0] + 1, self.heat_patch_stride[0]):
                new_mask = np.zeros(mask.shape)
                new_mask[
                    i : min(i + self.heat_patch_size[1], mask.shape[0]),
                    j : min(j + self.heat_patch_size[0], mask.shape[1]),
                ] = patch
                new_mask = new_mask * mask
                if np.sum(new_mask) > 0:
                    patches.append(new_mask)
                    indices.append((i, j))

        # compute heatmap and order
        if self.heatmap_mode == "Random":
            tr_scores = [random.random() for i in range(len(patches))]
        else:  # Target mode

            tr_scores = self._get_heatmap(
                x,
                x_noise,
                x_tar_noise,
                mask,
                y,
                patches,
                xforms,
                clip_min,
                clip_max,
                pts,
            )

        tr_scores_np = np.asarray(tr_scores)
        order = tr_scores_np.argsort()
        patches = [patches[ind] for ind in order]
        indices = [indices[ind] for ind in order]

        # STAGE 2: Coarse Reduction
        best_mask, patches, indices = self._get_coarse_reduced_mask(
            x, x_noise, x_tar_noise, y, mask, patches, indices, xforms, clip_min, clip_max, pts
        )

        # STAGE 3: Fine Reduction
        if self.max_mask_size > 0:
            lbd = 5
            while best_mask.sum() / mask.shape[-1] > self.max_mask_size:
                patches_copy = list(patches)
                best_mask = self._get_fine_reduced_mask(
                    x,
                    x_noise,
                    x_tar_noise,
                    y,
                    best_mask,
                    patches_copy,
                    xforms,
                    object_size,
                    clip_min,
                    clip_max,
                    lbd,
                    pts,
                )
                lbd += 5
        else:
            best_mask = self._get_fine_reduced_mask(
                x,
                x_noise,
                x_tar_noise,
                y,
                best_mask,
                patches,
                xforms,
                object_size,
                clip_min,
                clip_max,
                pts=pts,
            )

        return best_mask

    def _get_heatmap(
        self,
        x: np.ndarray,
        x_noise: np.ndarray,
        x_tar_noise: np.ndarray,
        mask: np.ndarray,
        y: int,
        patches: List[np.ndarray],
        xforms: List,
        clip_min: float,
        clip_max: float,
        pts: Optional[np.ndarray] = None,
    ) -> List[float]:
        """
        Function to generate a heatmap.

        :param x: An array with one original input to be attacked.
        :param x_noise: x in the resolution of the noise size.
        :param x_tar_noise: x_tar in the resolution of the noise size.
        :param mask: An array with a mask to be applied to the adversarial perturbations. Shape needs to be
                     broadcastable to the shape of x. Any features for which the mask is zero will not be adversarially
                     perturbed.
        :param y: The target label.
        :param patches: list of patches from heatmap.
        :param xforms: list of transform params.
        :param clip_min: Minimum value of an example.
        :param clip_max: Maximum value of an example.
        :param pts: Optional. A set of points that will set the crop size in the perspective transform.
        :return: List of transform-robustness scores for the list of patches.
        """
        tr_scores = []

        # iterate over patches and compute transform_robustness without each individual patch
        for patch in patches:
            next_mask = mask * (np.ones(mask.shape) - patch)
            theta = (x_tar_noise - x_noise) * next_mask
            xform_imgs = get_transformed_images(
                x, next_mask, xforms, 1.0, theta, self.net_size, clip_min, clip_max, pts
            )
            success_rate = run_predictions(self.estimator, xform_imgs, y, self.batch_size, False)
            tr_scores.append(success_rate)

        return tr_scores

    def _evaluate_transform_robustness_at_pivot(
        self,
        x: np.ndarray,
        x_noise: np.ndarray,
        x_tar_noise: np.ndarray,
        y: int,
        mask: np.ndarray,
        patches: List[np.ndarray],
        xforms: List,
        clip_min: float,
        clip_max: float,
        pivot: int,
        pts: Optional[np.ndarray] = None,
    ) -> Tuple[float, np.ndarray]:
        """
        Function as a binary search plug-in that evaluates the transform-robustness at the specified pivot.

        :param x: An array with one original input to be attacked.
        :param x_noise: x in the resolution of the noise size.
        :param x_tar_noise: x_tar in the resolution of the noise size.
        :param y: The target label.
        :param mask: An array with a mask to be applied to the adversarial perturbations. Shape needs to be
                     broadcastable to the shape of x. Any features for which the mask is zero will not be adversarially
                     perturbed.
        :param patches: list of patches from heatmap.
        :param xforms: list of transform params.
        :param clip_min: Minimum value of an example.
        :param clip_max: Maximum value of an example.
        :param pivot: Pivot point to evaluate transform-robustness at.
        :param pts: Optional. A set of points that will set the crop size in the perspective transform.
        :return: transform-robustness and mask.
        """

        best_mask = np.zeros(mask.shape)
        ordering = patches[:pivot]
        for next_patch in ordering:
            next_mask = best_mask + (np.zeros(best_mask.shape) + next_patch)
            next_mask = np.where(next_mask > 0, 1.0, 0.0)
            best_mask = next_mask

        theta = (x_tar_noise - x_noise) * best_mask
        xform_imgs = get_transformed_images(x, best_mask, xforms, 1.0, theta, self.net_size, clip_min, clip_max, pts)
        success_rate = run_predictions(self.estimator, xform_imgs, y, self.batch_size, False)

        return (success_rate, best_mask)

    def _get_coarse_reduced_mask(
        self,
        x: np.ndarray,
        x_noise: np.ndarray,
        x_tar_noise: np.ndarray,
        y: int,
        mask: np.ndarray,
        patches: List[np.ndarray],
        indices: List[Tuple[int, int]],
        xforms: List,
        clip_min: float,
        clip_max: float,
        pts: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[np.ndarray], List[Tuple[int, int]]]:
        """
        Function to coarsely reduce mask.

        :param x: An array with one original input to be attacked.
        :param x_noise: x in the resolution of the noise size.
        :param x_tar_noise: x_tar in the resolution of the noise size.
        :param y: The target label.
        :param mask: An array with a mask to be applied to the adversarial perturbations. Shape needs to be
                     broadcastable to the shape of x. Any features for which the mask is zero will not be adversarially
                     perturbed.
        :param patches: list of patches from heatmap.
        :param indices: list of indices for the heatmap patches.
        :param xforms: list of transform params.
        :param clip_min: Minimum value of an example.
        :param clip_max: Maximum value of an example.
        :param pts: Optional. A set of points that will set the crop size in the perspective transform.
        :return: mask, adjusted list of patches, adjusted list of indices
        """
        # binary search leftmost pivot value for which tr exceeeds specificed threshold if one exists
        num_patches = len(patches)
        if num_patches == 1:
            pivot = 0

        else:
            low = 0
            high = num_patches - 1
            while low <= high:
                mid = low + (high - low) // 2
                score, _ = self._evaluate_transform_robustness_at_pivot(
                    x, x_noise, x_tar_noise, y, mask, patches, xforms, clip_min, clip_max, mid, pts
                )
                if score >= self.tr_hi:
                    if mid > 0:
                        high = mid - 1
                        continue
                    break

                # score < threshold:
                low = mid + 1

            pivot = mid

        _, best_mask = self._evaluate_transform_robustness_at_pivot(
            x, x_noise, x_tar_noise, y, mask, patches, xforms, clip_min, clip_max, pivot, pts
        )

        patches = patches[:]  # reduce will examine all patches
        indices = indices[:]

        return best_mask, patches, indices

    def _get_fine_reduced_mask(
        self,
        x: np.ndarray,
        x_noise: np.ndarray,
        x_tar_noise: np.ndarray,
        y: int,
        mask: np.ndarray,
        patches: List[np.ndarray],
        xforms: List,
        object_size: float,
        clip_min: float,
        clip_max: float,
        lbd: float = 5,
        pts: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Function to finely reduce mask.

        :param x: An array with one original input to be attacked.
        :param x_noise: x in the resolution of the noise size.
        :param x_tar_noise: x_tar in the resolution of the noise size.
        :param y: The target label.
        :param mask: An array with a mask to be applied to the adversarial perturbations. Shape needs to be
                     broadcastable to the shape of x. Any features for which the mask is zero will not be adversarially
                     perturbed.
        :param patches: list of patches from heatmap.
        :param xforms: list of transform params.
        :param obj_size: Estimated width of object in inches for perspective transform.
        :param clip_min: Minimum value of an example.
        :param clip_max: Maximum value of an example.
        :param lbd: Weight for mask scoring function.
        :param pts: Optional. A set of points that will set the crop size in the perspective transform.
        :return: mask
        """
        # STAGE 3: Fine reduction
        theta = (x_tar_noise - x_noise) * mask

        # get initial transform_robustness
        xform_imgs = get_transformed_images(x, mask, xforms, 1.0, theta, self.net_size, clip_min, clip_max, pts)

        success_rate = run_predictions(self.estimator, xform_imgs, y, self.batch_size, False)

        init_tr_err = 1 - success_rate

        best_score = score_fn(mask, init_tr_err, object_size, threshold=(1 - self.tr_lo), lbd=lbd)
        best_mask = mask

        new_patches = []
        j = 0
        # iterate over patches, greedily remove if the score improves
        while patches:
            j = j + 1

            # highest transform_robustness is now always at end, for pop()
            next_patch = patches.pop()
            if np.max(next_patch * best_mask) == 0:
                continue
            next_mask = best_mask * (np.ones(best_mask.shape) - next_patch)
            theta = (x_tar_noise - x_noise) * next_mask

            xform_imgs = get_transformed_images(
                x, next_mask, xforms, 1.0, theta, self.net_size, clip_min, clip_max, pts
            )

            success_rate = run_predictions(self.estimator, xform_imgs, y, self.batch_size, False)

            score = score_fn(next_mask, 1 - success_rate, object_size, threshold=(1 - self.tr_lo), lbd=lbd)
            if score < best_score:
                best_score = score
                best_mask = next_mask
                nbits = best_mask.sum() / best_mask.shape[-1]
                if self.max_mask_size > 0 and nbits < self.max_mask_size:
                    break
            else:
                new_patches.append(next_patch)

        return best_mask

    def _boost(
        self,
        x: np.ndarray,
        x_noise: np.ndarray,
        x_tar_noise: np.ndarray,
        mask: np.ndarray,
        y: int,
        obj_width: float,
        focal: float,
        clip_min: float,
        clip_max: float,
        pts: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Function to boost transform-robustness.

        :param x: An array with one original input to be attacked.
        :param x_noise: x in the resolution of the noise size.
        :param x_tar_noise: x_tar in the resolution of the noise size.
        :param mask: An array with a mask to be applied to the adversarial perturbations. Shape needs to be
                     broadcastable to the shape of x. Any features for which the mask is zero will not be adversarially
                     perturbed.
        :param y: The target label.
        :param obj_width: Estimated width of object in inches for perspective transform.
        :param focal: Estimated focal length in ft for perspective transform.
        :param clip_min: Minimum value of an example.
        :param clip_max: Maximum value of an example.
        :param pts: Optional. A set of points that will set the crop size in the perspective transform.
        :return: attacked image
        """

        xforms = get_transform_params(
            self.num_xforms_boost,
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

        # Initialize
        theta = (x_tar_noise - x_noise) * mask
        query_count = 0

        xform_imgs = get_transformed_images(x, mask, xforms, 1.0, theta, self.net_size, clip_min, clip_max, pts)

        err_rate = run_predictions(self.estimator, xform_imgs, y, self.batch_size, True)
        query_count += self.num_xforms_boost

        best_theta, best_eps = theta, err_rate

        # End Initialize

        theta, eps = best_theta.copy(), best_eps

        opt_count = 0

        # gradient free optimization steps
        while True:
            gradient = np.zeros(theta.shape)
            num_q_samples = 10

            # Take q samples of random Gaussian noise to use as new directions, calculate transform_robustness
            # Used for gradient estimate
            for _ in range(num_q_samples):
                unit_dir = np.random.randn(*theta.shape).astype(np.float32) * mask
                unit_dir = unit_dir / np.linalg.norm(unit_dir)
                ttt = theta + self.beta * unit_dir

                xform_imgs = get_transformed_images(x, mask, xforms, 1.0, ttt, self.net_size, clip_min, clip_max, pts)
                eps_ttt = run_predictions(self.estimator, xform_imgs, y, self.batch_size, True)
                opt_count += self.num_xforms_boost

                gradient += (eps_ttt - eps) / self.beta * unit_dir

            gradient = 1.0 / num_q_samples * gradient

            # Take gradient step
            new_theta = theta - self.eta * gradient
            xform_imgs = get_transformed_images(x, mask, xforms, 1.0, new_theta, self.net_size, clip_min, clip_max, pts)
            new_eps = run_predictions(self.estimator, xform_imgs, y, self.batch_size, True)
            opt_count += self.num_xforms_boost

            if new_eps < best_eps:
                best_theta, best_eps = new_theta.copy(), new_eps

            theta, eps = new_theta.copy(), new_eps

            if (opt_count + query_count + self.num_xforms_boost * 11) > self.num_boost_queries:
                break

        adv_example, _, _ = add_noise(x, mask, 1.0, best_theta)

        return adv_example

    def _check_params(self) -> None:
        if self.noise_size < self.heat_patch_size:
            raise ValueError("Heatmap patch size must be smaller than the noise size.")

        if min(self.heat_patch_size) <= 0:
            raise ValueError("Heatmap patch size must be positive.")

        if not isinstance(self.heat_patch_size[0], int) or not isinstance(self.heat_patch_size[1], int):
            raise ValueError("Heatmap patch size must be a tuple of two integers.")

        if (
            min(self.heat_patch_stride) <= 0
            or not isinstance(self.heat_patch_stride[0], int)
            or not isinstance(self.heat_patch_stride[1], int)
        ):
            raise ValueError("Heatmap patch stride must be a tuple of two positive integers.")

        if self.heatmap_mode not in ["Target", "Random"]:
            raise ValueError("Heatmap mode must be 'Target' or 'Random'.")

        if self.tr_lo < 0 or self.tr_lo > 1:
            raise ValueError("tr_lo must be between 0 and 1.")

        if self.tr_hi < 0 or self.tr_hi > 1:
            raise ValueError("tr_hi must be between 0 and 1.")

        if self.tr_hi < self.tr_lo:
            raise ValueError("tr_hi must be at least as high as tr_lo.")

        if self.num_xforms_mask < 0 or not isinstance(self.num_xforms_mask, int):
            raise ValueError("num_xforms_mask must be non-negative integer.")

        if self.beta <= 0:
            raise ValueError("beta must be positive.")

        if self.eta <= 0:
            raise ValueError("eta must be positive.")

        if self.num_xforms_boost < 0 or not isinstance(self.num_xforms_boost, int):
            raise ValueError("num_xforms_boost must be positive integer.")

        if self.num_boost_queries <= 0 or not isinstance(self.num_boost_queries, int):
            raise ValueError("num_boost_queries must be positive.")

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
