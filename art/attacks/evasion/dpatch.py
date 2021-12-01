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
This module implements the adversarial patch attack `DPatch` for object detectors.

| Paper link: https://arxiv.org/abs/1806.02299v4
"""
import logging
import math
import random
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
from tqdm.auto import trange

from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.estimators.object_detection.object_detector import ObjectDetectorMixin
from art import config

if TYPE_CHECKING:
    from art.utils import OBJECT_DETECTOR_TYPE

logger = logging.getLogger(__name__)


class DPatch(EvasionAttack):
    """
    Implementation of the DPatch attack.

    | Paper link: https://arxiv.org/abs/1806.02299v4
    """

    attack_params = EvasionAttack.attack_params + [
        "patch_shape",
        "learning_rate",
        "max_iter",
        "batch_size",
        "verbose",
    ]

    _estimator_requirements = (BaseEstimator, LossGradientsMixin, ObjectDetectorMixin)

    def __init__(
        self,
        estimator: "OBJECT_DETECTOR_TYPE",
        patch_shape: Tuple[int, int, int] = (40, 40, 3),
        learning_rate: float = 5.0,
        max_iter: int = 500,
        batch_size: int = 16,
        verbose: bool = True,
    ):
        """
        Create an instance of the :class:`.DPatch`.

        :param estimator: A trained object detector.
        :param patch_shape: The shape of the adversarial path as a tuple of shape (height, width, nb_channels).
        :param learning_rate: The learning rate of the optimization.
        :param max_iter: The number of optimization steps.
        :param batch_size: The size of the training batch.
        :param verbose: Show progress bars.
        """
        super().__init__(estimator=estimator)

        self.patch_shape = patch_shape
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.verbose = verbose
        self._check_params()

        if self.estimator.clip_values is None:
            self._patch = np.zeros(shape=patch_shape, dtype=config.ART_NUMPY_DTYPE)
        else:
            self._patch = (
                np.random.randint(0, 255, size=patch_shape)
                / 255
                * (self.estimator.clip_values[1] - self.estimator.clip_values[0])
                + self.estimator.clip_values[0]
            ).astype(config.ART_NUMPY_DTYPE)

        self.target_label: Optional[Union[int, np.ndarray, List[int]]] = list()

    def generate(  # pylint: disable=W0221
        self,
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
        target_label: Optional[Union[int, List[int], np.ndarray]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Generate DPatch.

        :param x: Sample images.
        :param y: Target labels for object detector.
        :param target_label: The target label of the DPatch attack.
        :param mask: An boolean array of shape equal to the shape of a single samples (1, H, W) or the shape of `x`
                     (N, H, W) without their channel dimensions. Any features for which the mask is True can be the
                     center location of the patch during sampling.
        :type mask: `np.ndarray`
        :return: Adversarial patch.
        """
        mask = kwargs.get("mask")
        if mask is not None:
            mask = mask.copy()
        if mask is not None and (  # pragma: no cover
            mask.dtype != np.bool
            or not (mask.shape[0] == 1 or mask.shape[0] == x.shape[0])
            or not (
                (mask.shape[1] == x.shape[1] and mask.shape[2] == x.shape[2])
                or (mask.shape[1] == x.shape[2] and mask.shape[2] == x.shape[3])
            )
        ):
            raise ValueError(
                "The shape of `mask` has to be equal to the shape of a single samples (1, H, W) or the"
                "shape of `x` (N, H, W) without their channel dimensions."
            )

        channel_index = 1 if self.estimator.channels_first else x.ndim - 1
        if x.shape[channel_index] != self.patch_shape[channel_index - 1]:
            raise ValueError("The color channel index of the images and the patch have to be identical.")
        if y is not None:
            raise ValueError("The DPatch attack does not use target labels.")
        if x.ndim != 4:  # pragma: no cover
            raise ValueError("The adversarial patch can only be applied to images.")
        if target_label is not None:
            if isinstance(target_label, int):
                self.target_label = [target_label] * x.shape[0]
            elif isinstance(target_label, np.ndarray):
                if not (  # pragma: no cover
                    target_label.shape == (x.shape[0], 1) or target_label.shape == (x.shape[0],)
                ):
                    raise ValueError("The target_label has to be a 1-dimensional array.")
                self.target_label = target_label.tolist()
            else:
                if not len(target_label) == x.shape[0] or not isinstance(target_label, list):  # pragma: no cover
                    raise ValueError("The target_label as list of integers needs to of length number of images in `x`.")
                self.target_label = target_label

        patched_images, transforms = self._augment_images_with_patch(
            x,
            self._patch,
            random_location=True,
            channels_first=self.estimator.channels_first,
            mask=mask,
            transforms=None,
        )
        patch_target: List[Dict[str, np.ndarray]] = list()

        if self.target_label:

            for i_image in range(patched_images.shape[0]):
                if isinstance(self.target_label, int):
                    t_l = self.target_label
                else:
                    t_l = self.target_label[i_image]

                i_x_1 = transforms[i_image]["i_x_1"]
                i_x_2 = transforms[i_image]["i_x_2"]
                i_y_1 = transforms[i_image]["i_y_1"]
                i_y_2 = transforms[i_image]["i_y_2"]

                target_dict = dict()
                target_dict["boxes"] = np.asarray([[i_x_1, i_y_1, i_x_2, i_y_2]])
                target_dict["labels"] = np.asarray(
                    [
                        t_l,
                    ]
                )
                target_dict["scores"] = np.asarray(
                    [
                        1.0,
                    ]
                )

                patch_target.append(target_dict)

        else:

            predictions = self.estimator.predict(x=patched_images, standardise_output=True)

            for i_image in range(patched_images.shape[0]):
                target_dict = dict()
                target_dict["boxes"] = predictions[i_image]["boxes"]
                target_dict["labels"] = predictions[i_image]["labels"]
                target_dict["scores"] = predictions[i_image]["scores"]

                patch_target.append(target_dict)

        for i_step in trange(self.max_iter, desc="DPatch iteration", disable=not self.verbose):
            if i_step == 0 or (i_step + 1) % 100 == 0:
                logger.info("Training Step: %i", i_step + 1)

            num_batches = math.ceil(x.shape[0] / self.batch_size)
            patch_gradients = np.zeros_like(self._patch)

            for i_batch in range(num_batches):
                i_batch_start = i_batch * self.batch_size
                i_batch_end = min((i_batch + 1) * self.batch_size, patched_images.shape[0])

                gradients = self.estimator.loss_gradient(
                    x=patched_images[i_batch_start:i_batch_end],
                    y=patch_target[i_batch_start:i_batch_end],
                    standardise_output=True,
                )

                for i_image in range(gradients.shape[0]):

                    i_x_1 = transforms[i_batch_start + i_image]["i_x_1"]
                    i_x_2 = transforms[i_batch_start + i_image]["i_x_2"]
                    i_y_1 = transforms[i_batch_start + i_image]["i_y_1"]
                    i_y_2 = transforms[i_batch_start + i_image]["i_y_2"]

                    if self.estimator.channels_first:
                        patch_gradients_i = gradients[i_image, :, i_x_1:i_x_2, i_y_1:i_y_2]
                    else:
                        patch_gradients_i = gradients[i_image, i_x_1:i_x_2, i_y_1:i_y_2, :]

                    patch_gradients = patch_gradients + patch_gradients_i

            if self.target_label:
                self._patch = self._patch - np.sign(patch_gradients) * self.learning_rate
            else:
                self._patch = self._patch + np.sign(patch_gradients) * self.learning_rate

            if self.estimator.clip_values is not None:
                self._patch = np.clip(
                    self._patch,
                    a_min=self.estimator.clip_values[0],
                    a_max=self.estimator.clip_values[1],
                )

            patched_images, _ = self._augment_images_with_patch(
                x,
                self._patch,
                random_location=False,
                channels_first=self.estimator.channels_first,
                mask=None,
                transforms=transforms,
            )

        return self._patch

    @staticmethod
    def _augment_images_with_patch(
        x: np.ndarray,
        patch: np.ndarray,
        random_location: bool,
        channels_first: bool,
        mask: Optional[np.ndarray] = None,
        transforms: List[Dict[str, int]] = None,
    ) -> Tuple[np.ndarray, List[Dict[str, int]]]:
        """
        Augment images with patch.

        :param x: Sample images.
        :param patch: The patch to be applied.
        :param random_location: If True apply patch at randomly shifted locations, otherwise place patch at origin
                                (top-left corner).
        :param channels_first: Set channels first or last.
        :param mask: An boolean array of shape equal to the shape of a single samples (1, H, W) or the shape of `x`
                     (N, H, W) without their channel dimensions. Any features for which the mask is True can be the
                     center location of the patch during sampling.
        :param transforms: Patch transforms, requires `random_location=False`, and `mask=None`.
        :type mask: `np.ndarray`
        """
        if transforms is not None:
            if random_location or mask is not None:
                raise ValueError(
                    "Definition of patch locations in `locations` requires `random_location=False`, and `mask=None`."
                )

        random_transformations = list()
        x_copy = x.copy()
        patch_copy = patch.copy()

        if channels_first:
            x_copy = np.transpose(x_copy, (0, 2, 3, 1))
            patch_copy = np.transpose(patch_copy, (1, 2, 0))

        for i_image in range(x.shape[0]):

            if transforms is None:

                if random_location:
                    if mask is None:
                        i_x_1 = random.randint(0, x_copy.shape[1] - 1 - patch_copy.shape[0])
                        i_y_1 = random.randint(0, x_copy.shape[2] - 1 - patch_copy.shape[1])
                    else:

                        if mask.shape[0] == 1:
                            mask_2d = mask[0, :, :]
                        else:
                            mask_2d = mask[i_image, :, :]

                        edge_x_0 = patch_copy.shape[0] // 2
                        edge_x_1 = patch_copy.shape[0] - edge_x_0
                        edge_y_0 = patch_copy.shape[1] // 2
                        edge_y_1 = patch_copy.shape[1] - edge_y_0

                        mask_2d[0:edge_x_0, :] = False
                        mask_2d[-edge_x_1:, :] = False
                        mask_2d[:, 0:edge_y_0] = False
                        mask_2d[:, -edge_y_1:] = False

                        num_pos = np.argwhere(mask_2d).shape[0]
                        pos_id = np.random.choice(num_pos, size=1)
                        pos = np.argwhere(mask_2d > 0)[pos_id[0]]
                        i_x_1 = pos[0] - edge_x_0
                        i_y_1 = pos[1] - edge_y_0

                else:
                    i_x_1 = 0
                    i_y_1 = 0

                i_x_2 = i_x_1 + patch_copy.shape[0]
                i_y_2 = i_y_1 + patch_copy.shape[1]

                random_transformations.append({"i_x_1": i_x_1, "i_y_1": i_y_1, "i_x_2": i_x_2, "i_y_2": i_y_2})

            else:
                i_x_1 = transforms[i_image]["i_x_1"]
                i_x_2 = transforms[i_image]["i_x_2"]
                i_y_1 = transforms[i_image]["i_y_1"]
                i_y_2 = transforms[i_image]["i_y_2"]

            x_copy[i_image, i_x_1:i_x_2, i_y_1:i_y_2, :] = patch_copy

        if channels_first:
            x_copy = np.transpose(x_copy, (0, 3, 1, 2))

        return x_copy, random_transformations

    def apply_patch(
        self,
        x: np.ndarray,
        patch_external: Optional[np.ndarray] = None,
        random_location: bool = False,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Apply the adversarial patch to images.

        :param x: Images to be patched.
        :param patch_external: External patch to apply to images `x`. If None the attacks patch will be applied.
        :param random_location: True if patch location should be random.
        :param mask: An boolean array of shape equal to the shape of a single samples (1, H, W) or the shape of `x`
                     (N, H, W) without their channel dimensions. Any features for which the mask is True can be the
                     center location of the patch during sampling.
        :return: The patched images.
        """
        if patch_external is not None:
            patch_local = patch_external
        else:
            patch_local = self._patch

        patched_images, _ = self._augment_images_with_patch(
            x=x,
            patch=patch_local,
            random_location=random_location,
            channels_first=self.estimator.channels_first,
            mask=mask,
        )

        return patched_images

    def _check_params(self) -> None:
        if not isinstance(self.patch_shape, (tuple, list)) or not all(isinstance(s, int) for s in self.patch_shape):
            raise ValueError("The patch shape must be either a tuple or list of integers.")
        if len(self.patch_shape) != 3:
            raise ValueError("The length of patch shape must be 3.")

        if not isinstance(self.learning_rate, float):
            raise ValueError("The learning rate must be of type float.")
        if self.learning_rate <= 0.0:
            raise ValueError("The learning rate must be greater than 0.0.")

        if not isinstance(self.max_iter, int):
            raise ValueError("The number of optimization steps must be of type int.")
        if self.max_iter <= 0:
            raise ValueError("The number of optimization steps must be greater than 0.")

        if not isinstance(self.batch_size, int):
            raise ValueError("The batch size must be of type int.")
        if self.batch_size <= 0:
            raise ValueError("The batch size must be greater than 0.")

        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be of type bool.")
