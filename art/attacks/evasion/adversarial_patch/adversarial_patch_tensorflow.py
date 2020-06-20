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
This module implements the adversarial patch attack `AdversarialPatch`. This attack generates an adversarial patch that
can be printed into the physical world with a common printer. The patch can be used to fool image classifiers.

| Paper link: https://arxiv.org/abs/1712.09665
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import math
from typing import Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator, NeuralNetworkMixin
from art.estimators.classification.classifier import (
    ClassifierMixin,
    ClassifierNeuralNetwork,
    ClassifierGradients,
)
from art.utils import check_and_transform_label_format

if TYPE_CHECKING:
    import tensorflow as tf

logger = logging.getLogger(__name__)


class AdversarialPatchTensorFlowV2(EvasionAttack):
    """
    Implementation of the adversarial patch attack.

    | Paper link: https://arxiv.org/abs/1712.09665
    """

    attack_params = EvasionAttack.attack_params + [
        "rotation_max",
        "scale_min",
        "scale_max",
        "learning_rate",
        "max_iter",
        "batch_size",
        "patch_shape",
    ]

    _estimator_requirements = (BaseEstimator, NeuralNetworkMixin, ClassifierMixin)

    def __init__(
        self,
        classifier: Union[ClassifierNeuralNetwork, ClassifierGradients],
        rotation_max: float = 22.5,
        scale_min: float = 0.1,
        scale_max: float = 1.0,
        learning_rate: float = 5.0,
        max_iter: int = 500,
        batch_size: int = 16,
        patch_shape: Optional[Tuple[int, int, int]] = None,
    ):
        """
        Create an instance of the :class:`.AdversarialPatchTensorFlowV2`.

        :param classifier: A trained classifier.
        :param rotation_max: The maximum rotation applied to random patches. The value is expected to be in the
               range `[0, 180]`.
        :param scale_min: The minimum scaling applied to random patches. The value should be in the range `[0, 1]`,
               but less than `scale_max`.
        :param scale_max: The maximum scaling applied to random patches. The value should be in the range `[0, 1]`, but
               larger than `scale_min.`
        :param learning_rate: The learning rate of the optimization.
        :param max_iter: The number of optimization steps.
        :param batch_size: The size of the training batch.
        :param patch_shape: The shape of the adversarial patch as a tuple of shape (width, height, nb_channels).
                            Currently only supported for `TensorFlowV2Classifier`. For classifiers of other frameworks
                            the `patch_shape` is set to the shape of the image samples.
        """
        import tensorflow as tf  # lgtm [py/repeated-import]

        super(AdversarialPatchTensorFlowV2, self).__init__(estimator=classifier)
        self.rotation_max = rotation_max
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.patch_shape = patch_shape
        self.image_shape = classifier.input_shape
        self._check_params()

        if self.image_shape[2] not in [1, 3]:
            raise ValueError("Color channel need to be in last dimension.")

        if self.patch_shape is not None:
            if self.patch_shape[2] not in [1, 3]:
                raise ValueError("Color channel need to be in last dimension.")
            if self.patch_shape[0] != self.patch_shape[1]:
                raise ValueError("Patch height and width need to be the same.")
        if not (self.estimator.postprocessing_defences is None or self.estimator.postprocessing_defences == []):
            raise ValueError(
                "Framework-specific implementation of Adversarial Patch attack does not yet support "
                + "postprocessing defences."
            )

        mean_value = (self.estimator.clip_values[1] - self.estimator.clip_values[0]) / 2.0 + self.estimator.clip_values[
            0
        ]
        initial_value = np.ones(self.patch_shape) * mean_value
        self._patch = tf.Variable(
            initial_value=initial_value,
            shape=self.patch_shape,
            dtype=tf.float32,
            constraint=lambda x: tf.clip_by_value(x, self.estimator.clip_values[0], self.estimator.clip_values[1]),
        )

        self._train_op = tf.keras.optimizers.SGD(
            learning_rate=self.learning_rate, momentum=0.0, nesterov=False, name="SGD"
        )

    def _train_step(self, images: Optional[np.ndarray] = None, target: Optional[np.ndarray] = None) -> "tf.Tensor":
        import tensorflow as tf  # lgtm [py/repeated-import]

        if target is None:
            target = self.estimator.predict(x=images)
            self.targeted = False
        else:
            self.targeted = True

        with tf.GradientTape() as tape:
            tape.watch(self._patch)
            loss = self._loss(images, target)

        gradients = tape.gradient(loss, [self._patch])

        if not self.targeted:
            gradients = [-g for g in gradients]

        self._train_op.apply_gradients(zip(gradients, [self._patch]))

        return loss

    def _probabilities(self, images: "tf.Tensor") -> "tf.Tensor":
        import tensorflow as tf  # lgtm [py/repeated-import]

        patched_input = self._random_overlay(images, self._patch)

        patched_input = tf.clip_by_value(
            patched_input, clip_value_min=self.estimator.clip_values[0], clip_value_max=self.estimator.clip_values[1],
        )

        probabilities = self.estimator._predict_framework(patched_input)

        return probabilities

    def _loss(self, images: "tf.Tensor", target: "tf.Tensor") -> "tf.Tensor":
        import tensorflow as tf  # lgtm [py/repeated-import]

        probabilities = self._probabilities(images)

        self._loss_per_example = tf.keras.losses.categorical_crossentropy(
            y_true=target, y_pred=probabilities, from_logits=False, label_smoothing=0
        )

        loss = tf.reduce_mean(self._loss_per_example)

        return loss

    def _get_circular_patch_mask(self, nb_images: int, sharpness: int = 40) -> "tf.Tensor":
        """
        Return a circular patch mask.
        """
        import tensorflow as tf  # lgtm [py/repeated-import]

        diameter = self.image_shape[0]

        x = np.linspace(-1, 1, diameter)
        y = np.linspace(-1, 1, diameter)
        x_grid, y_grid = np.meshgrid(x, y, sparse=True)
        z_grid = (x_grid ** 2 + y_grid ** 2) ** sharpness

        image_mask = 1 - np.clip(z_grid, -1, 1)
        image_mask = np.expand_dims(image_mask, axis=2)
        image_mask = np.broadcast_to(image_mask, self.image_shape)
        image_mask = tf.stack([image_mask] * nb_images)
        return image_mask

    def _random_overlay(self, images: np.ndarray, patch: np.ndarray, scale: Optional[float] = None) -> "tf.Tensor":
        import tensorflow as tf  # lgtm [py/repeated-import]
        import tensorflow_addons as tfa

        nb_images = images.shape[0]
        image_mask = self._get_circular_patch_mask(nb_images=nb_images)
        image_mask = tf.cast(image_mask, images.dtype)
        patch = tf.cast(patch, images.dtype)
        padded_patch = tf.stack([patch] * nb_images)
        transform_vectors = list()

        for i in range(nb_images):
            if scale is None:
                im_scale = np.random.uniform(low=self.scale_min, high=self.scale_max)
            else:
                im_scale = scale
            padding_after_scaling = (1 - im_scale) * self.image_shape[0]
            x_shift = np.random.uniform(-padding_after_scaling, padding_after_scaling)
            y_shift = np.random.uniform(-padding_after_scaling, padding_after_scaling)
            phi_rotate = float(np.random.uniform(-self.rotation_max, self.rotation_max)) / 90.0 * (math.pi / 2.0)

            # Rotation
            rotation_matrix = np.array(
                [[math.cos(-phi_rotate), -math.sin(-phi_rotate)], [math.sin(-phi_rotate), math.cos(-phi_rotate)],]
            )

            # Scale
            xform_matrix = rotation_matrix * (1.0 / im_scale)
            a0, a1 = xform_matrix[0]
            b0, b1 = xform_matrix[1]

            x_origin = float(self.image_shape[0]) / 2
            y_origin = float(self.image_shape[1]) / 2

            x_origin_shifted, y_origin_shifted = np.matmul(xform_matrix, np.array([x_origin, y_origin]))

            x_origin_delta = x_origin - x_origin_shifted
            y_origin_delta = y_origin - y_origin_shifted

            a2 = x_origin_delta - (x_shift / (2 * im_scale))
            b2 = y_origin_delta - (y_shift / (2 * im_scale))

            transform_vectors.append(np.array([a0, a1, a2, b0, b1, b2, 0, 0]).astype(np.float32))

        image_mask = tfa.image.transform(image_mask, transform_vectors, "BILINEAR")
        padded_patch = tfa.image.transform(padded_patch, transform_vectors, "BILINEAR")
        inverted_mask = 1 - image_mask

        return images * inverted_mask + padded_patch * image_mask

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        import tensorflow as tf  # lgtm [py/repeated-import]

        y = check_and_transform_label_format(labels=y, nb_classes=self.estimator.nb_classes)

        shuffle = kwargs.get("shuffle", True)
        if shuffle:
            ds = (
                tf.data.Dataset.from_tensor_slices((x, y))
                .shuffle(10000)
                .batch(self.batch_size)
                .repeat(math.ceil(self.max_iter / (x.shape[0] / self.batch_size)))
            )
        else:
            ds = (
                tf.data.Dataset.from_tensor_slices((x, y))
                .batch(self.batch_size)
                .repeat(math.ceil(self.max_iter / (x.shape[0] / self.batch_size)))
            )

        i_iter = 0
        for images, target in tqdm(ds):

            if i_iter >= self.max_iter:
                break

            loss = self._train_step(images=images, target=target)

            if divmod(i_iter, 10)[1] == 0:
                logger.info("Iteration: {} Loss: {}".format(i_iter, loss))

            i_iter += 1

        return (
            self._patch.numpy(),
            self._get_circular_patch_mask(nb_images=1).numpy()[0],
        )

    def apply_patch(self, x: np.ndarray, scale: float, patch_external: Optional[np.ndarray] = None) -> np.ndarray:
        """
        A function to apply the learned adversarial patch to images.

        :param x: Instances to apply randomly transformed patch.
        :param scale: Scale of the applied patch in relation to the classifier input shape.
        :param patch_external: External patch to apply to images `x`.
        :return: The patched samples.
        """
        patch = patch_external if patch_external is not None else self._patch
        return self._random_overlay(images=x, patch=patch, scale=scale).numpy()

    def reset_patch(self, initial_patch_value: np.ndarray) -> None:
        """
        Reset the adversarial patch.

        :param initial_patch_value: Patch value to use for resetting the patch.
        """
        initial_value = np.ones(self.patch_shape) * initial_patch_value
        self._patch.assign(np.ones(shape=self.patch_shape) * initial_value)
