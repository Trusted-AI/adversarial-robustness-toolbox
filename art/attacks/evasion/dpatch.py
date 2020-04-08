# MIT License
#
# Copyright (C) IBM Corporation 2020
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

import numpy as np

from art.estimators.estimator import BaseEstimator
from art.estimators.object_detection.object_detector import ObjectDetectorMixin
from art.attacks.attack import EvasionAttack
from art.exceptions import EstimatorError

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
    ]

    estimator_requirements = (BaseEstimator, ObjectDetectorMixin)

    def __init__(
        self, estimator, patch_shape=(40, 40, 3), learning_rate=5.0, max_iter=500, batch_size=16,
    ):
        """
        Create an instance of the :class:`.DPatch`.

        :param estimator: A trained object detector.
        :type estimator: :class:`.ObjectDetectorMixin`
        :param patch_shape: The shape of the adversarial path as a tuple of shape (width, height, nb_channels).
        :type patch_shape: (`int`, `int`, `int`)
        :param learning_rate: The learning rate of the optimization.
        :type learning_rate: `float`
        :param max_iter: The number of optimization steps.
        :type max_iter: `int`
        :param batch_size: The size of the training batch.
        :type batch_size: `int`
        """
        super(DPatch, self).__init__(estimator=estimator)

        if not all(t in type(estimator).__mro__ for t in self.estimator_requirements):
            raise EstimatorError(self.__class__, self.estimator_requirements, estimator)

        kwargs = {
            "patch_shape": patch_shape,
            "learning_rate": learning_rate,
            "max_iter": max_iter,
            "batch_size": batch_size,
        }
        self.set_params(**kwargs)
        self._patch = np.ones(shape=patch_shape) * self.estimator.clip_values[1] / 2.0

    def generate(self, x, y=None, **kwargs):
        """
        Generate DPatch.

        :param x: Sample images.
        :type x: `np.ndarray`
        :param y: Target labels for object detector.
        :type y: `np.ndarray`
        :return: Adversarial patch.
        :rtype: `np.ndarray`
        """

        assert (
            x.shape[self.estimator.channel_index] == self.patch_shape[self.estimator.channel_index - 1]
        ), "The color channel index of the images and the patch have to be identical."

        assert y is None, "The DPatch attack does not use target labels."

        assert len(x.shape) == 4, "The adversarial patch can only be applied to images."

        for i_step in range(self.max_iter):
            print("i_step", i_step)
            if i_step == 0 or (i_step + 1) % 100 == 0:
                logger.info("Training Step: %i", i_step + 1)

            patched_images, transforms = self._augment_images_with_patch(x, self._patch, random_location=True)

            patch_target = list()

            for i_image in range(patched_images.shape[0]):

                i_x_1 = transforms[i_image]["i_x_1"]
                i_x_2 = transforms[i_image]["i_x_2"]
                i_y_1 = transforms[i_image]["i_y_1"]
                i_y_2 = transforms[i_image]["i_y_2"]

                target_dict = dict()
                target_dict["boxes"] = np.asarray([[i_x_1, i_y_1, i_x_2, i_y_2]])
                target_dict["labels"] = np.asarray([1,])
                target_dict["scores"] = np.asarray([1.0,])

                patch_target.append(target_dict)

            num_batches = math.ceil(x.shape[0] / self.batch_size)

            patch_gradients = np.zeros_like(self._patch)

            for i_batch in range(num_batches):
                i_batch_start = i_batch * self.batch_size
                i_batch_end = min((i_batch + 1) * self.batch_size, patched_images.shape[0])

                gradients = self.estimator.loss_gradient(
                    x=patched_images[i_batch_start:i_batch_end], y=patch_target[i_batch_start:i_batch_end]
                )

                for i_image in range(self.batch_size):

                    i_x_1 = transforms[i_batch_start + i_image]["i_x_1"]
                    i_x_2 = transforms[i_batch_start + i_image]["i_x_2"]
                    i_y_1 = transforms[i_batch_start + i_image]["i_y_1"]
                    i_y_2 = transforms[i_batch_start + i_image]["i_y_2"]

                    if self.estimator.channel_index == 3:
                        patch_gradients_i = gradients[i_image, i_x_1:i_x_2, i_y_1:i_y_2, :]
                    elif self.estimator.channel_index == 1:
                        patch_gradients_i = gradients[i_image, :, i_x_1:i_x_2, i_y_1:i_y_2]
                    else:
                        raise ValueError("Unrecognized channel index.")

                    patch_gradients += patch_gradients_i

            self._patch -= patch_gradients * self.learning_rate
            self._patch = np.clip(self._patch, a_min=self.estimator.clip_values[0], a_max=self.estimator.clip_values[1])

        return self._patch

    def _augment_images_with_patch(self, x, patch, random_location):
        """
        Augment images with randomly shifted patch.
        """
        transformations = list()
        x_copy = x.copy()
        patch_copy = patch.copy()

        if self.estimator.channel_index == 1:
            x_copy = np.swapaxes(x_copy, 1, 3)
            x_copy = np.swapaxes(x_copy, 2, 3)
            patch_copy = np.swapaxes(patch_copy, 0, 2)
            patch_copy = np.swapaxes(patch_copy, 1, 2)

        for i_image in range(x.shape[0]):

            if random_location:
                i_x_1 = random.randint(0, x.shape[1] - 1 - patch.shape[0])
                i_y_1 = random.randint(0, x.shape[2] - 1 - patch.shape[1])
            else:
                i_x_1 = 0
                i_y_1 = 0

            i_x_2 = i_x_1 + patch.shape[0]
            i_y_2 = i_y_1 + patch.shape[1]

            transformations.append({"i_x_1": i_x_1, "i_y_1": i_y_1, "i_x_2": i_x_2, "i_y_2": i_y_2})

            x_copy[i_image, i_x_1:i_x_2, i_y_1:i_y_2, :] = patch_copy

        if self.estimator.channel_index == 1:
            x_copy = np.swapaxes(x_copy, 1, 3)
            x_copy = np.swapaxes(x_copy, 2, 3)

        return x_copy, transformations

    def apply_patch(self, x, patch_external=None, random_location=False):
        """
        Apply the adversarial patch to images.

        :param x: Images to be patched.
        :type x: `np.ndarray`
        :param patch_external: External patch to apply to images `x`. If None the attacks patch will be applied.
        :type patch_external: `np.ndarray`
        :param random_location: True if patch location should be random.
        :type random_location: `bool`
        :return: The patched images.
        :rtype: `np.ndarray`
        """
        if patch_external is not None:
            patch_local = patch_external
        else:
            patch_local = self._patch

        patched_images, _ = self._augment_images_with_patch(x=x, patch=patch_local, random_location=random_location)

        return patched_images

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.

        :param patch_shape: The shape of the adversarial path as a tuple of shape (width, height, nb_channels).
        :type patch_shape: (`int`, `int`, `int`)
        :param learning_rate: The learning rate of the optimization.
        :type learning_rate: `float`
        :param max_iter: The number of optimization steps.
        :type max_iter: `int`
        :param batch_size: The size of the training batch.
        :type batch_size: `int`
        """
        super(DPatch, self).set_params(**kwargs)

        if not isinstance(self.patch_shape, tuple):
            raise ValueError("The patch shape must be a tuple of integers.")
        if len(self.patch_shape) != 3:
            raise ValueError("The length of patch shape must be 3.")

        if not isinstance(self.learning_rate, float):
            raise ValueError("The learning rate must be of type float.")
        if not self.learning_rate > 0.0:
            raise ValueError("The learning rate must be greater than 0.0.")

        if not isinstance(self.max_iter, int):
            raise ValueError("The number of optimization steps must be of type int.")
        if not self.max_iter > 0:
            raise ValueError("The number of optimization steps must be greater than 0.")

        if not isinstance(self.batch_size, int):
            raise ValueError("The batch size must be of type int.")
        if not self.batch_size > 0:
            raise ValueError("The batch size must be greater than 0.")
