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
can be printed into the physical world with a common printer. The patch can be used to fool image and video classifiers.

| Paper link: https://arxiv.org/abs/1712.09665
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import math
from typing import Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
from tqdm.auto import trange

from art.attacks.attack import EvasionAttack
from art.attacks.evasion.adversarial_patch.utils import insert_transformed_patch
from art.estimators.estimator import BaseEstimator, NeuralNetworkMixin
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import check_and_transform_label_format, is_probability, to_categorical

if TYPE_CHECKING:
    # pylint: disable=C0412
    import tensorflow as tf

    from art.utils import CLASSIFIER_NEURALNETWORK_TYPE

logger = logging.getLogger(__name__)


class AdversarialPatchTensorFlowV2(EvasionAttack):
    """
    Implementation of the adversarial patch attack for square and rectangular images and videos in TensorFlow v2.

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
        "tensor_board",
        "verbose",
    ]

    _estimator_requirements = (BaseEstimator, NeuralNetworkMixin, ClassifierMixin)

    def __init__(
        self,
        classifier: "CLASSIFIER_NEURALNETWORK_TYPE",
        rotation_max: float = 22.5,
        scale_min: float = 0.1,
        scale_max: float = 1.0,
        learning_rate: float = 5.0,
        max_iter: int = 500,
        batch_size: int = 16,
        patch_shape: Optional[Tuple[int, int, int]] = None,
        tensor_board: Union[str, bool] = False,
        verbose: bool = True,
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
        :param patch_shape: The shape of the adversarial patch as a tuple of shape HWC (width, height, nb_channels).
        :param tensor_board: Activate summary writer for TensorBoard: Default is `False` and deactivated summary writer.
                             If `True` save runs/CURRENT_DATETIME_HOSTNAME in current directory. Provide `path` in type
                             `str` to save in path/CURRENT_DATETIME_HOSTNAME.
                             Use hierarchical folder structure to compare between runs easily. e.g. pass in ‘runs/exp1’,
                             ‘runs/exp2’, etc. for each new experiment to compare across them.
        :param verbose: Show progress bars.
        """
        import tensorflow as tf  # lgtm [py/repeated-import]

        super().__init__(estimator=classifier, tensor_board=tensor_board)
        self.rotation_max = rotation_max
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        if patch_shape is None:
            self.patch_shape = self.estimator.input_shape
        else:
            self.patch_shape = patch_shape
        self.image_shape = classifier.input_shape
        self.verbose = verbose
        self._check_params()

        if self.estimator.channels_first:  # pragma: no cover
            raise ValueError("Color channel needs to be in last dimension.")

        self.use_logits: Optional[bool] = None

        self.i_h_patch = 0
        self.i_w_patch = 1

        self.nb_dims = len(self.image_shape)
        if self.nb_dims == 3:
            self.i_h = 0
            self.i_w = 1
        elif self.nb_dims == 4:
            self.i_h = 1
            self.i_w = 2

        if self.patch_shape[0] != self.patch_shape[1]:  # pragma: no cover
            raise ValueError("Patch height and width need to be the same.")

        if not (  # pragma: no cover
            self.estimator.postprocessing_defences is None or self.estimator.postprocessing_defences == []
        ):
            raise ValueError(
                "Framework-specific implementation of Adversarial Patch attack does not yet support "
                + "postprocessing defences."
            )

        mean_value = (self.estimator.clip_values[1] - self.estimator.clip_values[0]) / 2.0 + self.estimator.clip_values[
            0
        ]
        self._initial_value = np.ones(self.patch_shape) * mean_value
        self._patch = tf.Variable(
            initial_value=self._initial_value,
            shape=self.patch_shape,
            dtype=tf.float32,
            constraint=lambda x: tf.clip_by_value(x, self.estimator.clip_values[0], self.estimator.clip_values[1]),
        )

        self._train_op = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name="Adam"
        )

    def _train_step(
        self, images: "tf.Tensor", target: Optional["tf.Tensor"] = None, mask: Optional["tf.Tensor"] = None
    ) -> "tf.Tensor":
        import tensorflow as tf  # lgtm [py/repeated-import]

        if target is None:
            target = self.estimator.predict(x=images)
            self.targeted = False
        else:
            self.targeted = True

        with tf.GradientTape() as tape:
            tape.watch(self._patch)
            loss = self._loss(images, target, mask)

        gradients = tape.gradient(loss, [self._patch])

        if not self.targeted:
            gradients = [-g for g in gradients]

        self._train_op.apply_gradients(zip(gradients, [self._patch]))

        return loss

    def _predictions(self, images: "tf.Tensor", mask: Optional["tf.Tensor"]) -> "tf.Tensor":
        import tensorflow as tf  # lgtm [py/repeated-import]

        patched_input = self._random_overlay(images, self._patch, mask=mask)

        patched_input = tf.clip_by_value(
            patched_input,
            clip_value_min=self.estimator.clip_values[0],
            clip_value_max=self.estimator.clip_values[1],
        )

        predictions = self.estimator._predict_framework(patched_input)  # pylint: disable=W0212

        return predictions

    def _loss(self, images: "tf.Tensor", target: "tf.Tensor", mask: Optional["tf.Tensor"]) -> "tf.Tensor":
        import tensorflow as tf  # lgtm [py/repeated-import]

        predictions = self._predictions(images, mask)

        self._loss_per_example = tf.keras.losses.categorical_crossentropy(
            y_true=target, y_pred=predictions, from_logits=self.use_logits, label_smoothing=0
        )

        loss = tf.reduce_mean(self._loss_per_example)

        return loss

    def _get_circular_patch_mask(self, nb_samples: int, sharpness: int = 40) -> "tf.Tensor":
        """
        Return a circular patch mask.
        """
        import tensorflow as tf  # lgtm [py/repeated-import]

        diameter = np.minimum(self.patch_shape[self.i_h_patch], self.patch_shape[self.i_w_patch])

        x = np.linspace(-1, 1, diameter)
        y = np.linspace(-1, 1, diameter)
        x_grid, y_grid = np.meshgrid(x, y, sparse=True)
        z_grid = (x_grid ** 2 + y_grid ** 2) ** sharpness

        image_mask = 1 - np.clip(z_grid, -1, 1)
        image_mask = np.expand_dims(image_mask, axis=2)
        image_mask = np.broadcast_to(image_mask, self.patch_shape)
        image_mask = tf.stack([image_mask] * nb_samples)
        return image_mask

    def _random_overlay(
        self,
        images: Union[np.ndarray, "tf.Tensor"],
        patch: Union[np.ndarray, "tf.Variable"],
        scale: Optional[float] = None,
        mask: Optional[Union[np.ndarray, "tf.Tensor"]] = None,
    ) -> "tf.Tensor":
        import tensorflow as tf  # lgtm [py/repeated-import]
        import tensorflow_addons as tfa

        nb_samples = images.shape[0]

        image_mask = self._get_circular_patch_mask(nb_samples=nb_samples)
        image_mask = tf.cast(image_mask, images.dtype)

        smallest_image_edge = np.minimum(self.image_shape[self.i_h], self.image_shape[self.i_w])

        image_mask = tf.image.resize(
            image_mask,
            size=(smallest_image_edge, smallest_image_edge),
            method=tf.image.ResizeMethod.BILINEAR,
            preserve_aspect_ratio=False,
            antialias=False,
            name=None,
        )

        pad_h_before = int((self.image_shape[self.i_h] - image_mask.shape[self.i_h_patch + 1]) / 2)
        pad_h_after = int(self.image_shape[self.i_h] - pad_h_before - image_mask.shape[self.i_h_patch + 1])

        pad_w_before = int((self.image_shape[self.i_w] - image_mask.shape[self.i_w_patch + 1]) / 2)
        pad_w_after = int(self.image_shape[self.i_w] - pad_w_before - image_mask.shape[self.i_w_patch + 1])

        image_mask = tf.pad(  # pylint: disable=E1123
            image_mask,
            paddings=tf.constant([[0, 0], [pad_h_before, pad_h_after], [pad_w_before, pad_w_after], [0, 0]]),
            mode="CONSTANT",
            constant_values=0,
            name=None,
        )

        image_mask = tf.cast(image_mask, images.dtype)

        patch = tf.cast(patch, images.dtype)
        padded_patch = tf.stack([patch] * nb_samples)

        padded_patch = tf.image.resize(
            padded_patch,
            size=(smallest_image_edge, smallest_image_edge),
            method=tf.image.ResizeMethod.BILINEAR,
            preserve_aspect_ratio=False,
            antialias=False,
            name=None,
        )

        padded_patch = tf.pad(  # pylint: disable=E1123
            padded_patch,
            paddings=tf.constant([[0, 0], [pad_h_before, pad_h_after], [pad_w_before, pad_w_after], [0, 0]]),
            mode="CONSTANT",
            constant_values=0,
            name=None,
        )

        padded_patch = tf.cast(padded_patch, images.dtype)

        transform_vectors = list()
        translation_vectors = list()

        for i_sample in range(nb_samples):
            if scale is None:
                im_scale = np.random.uniform(low=self.scale_min, high=self.scale_max)
            else:
                im_scale = scale

            if mask is None:
                padding_after_scaling_h = (
                    self.image_shape[self.i_h] - im_scale * padded_patch.shape[self.i_h + 1]
                ) / 2.0
                padding_after_scaling_w = (
                    self.image_shape[self.i_w] - im_scale * padded_patch.shape[self.i_w + 1]
                ) / 2.0
                x_shift = np.random.uniform(-padding_after_scaling_w, padding_after_scaling_w)
                y_shift = np.random.uniform(-padding_after_scaling_h, padding_after_scaling_h)
            else:
                mask_2d = mask[i_sample, :, :]

                edge_x_0 = int(im_scale * padded_patch.shape[self.i_w + 1]) // 2
                edge_x_1 = int(im_scale * padded_patch.shape[self.i_w + 1]) - edge_x_0
                edge_y_0 = int(im_scale * padded_patch.shape[self.i_h + 1]) // 2
                edge_y_1 = int(im_scale * padded_patch.shape[self.i_h + 1]) - edge_y_0

                mask_2d[0:edge_x_0, :] = False
                if edge_x_1 > 0:
                    mask_2d[-edge_x_1:, :] = False
                mask_2d[:, 0:edge_y_0] = False
                if edge_y_1 > 0:
                    mask_2d[:, -edge_y_1:] = False

                num_pos = np.argwhere(mask_2d).shape[0]
                pos_id = np.random.choice(num_pos, size=1)
                pos = np.argwhere(mask_2d)[pos_id[0]]
                x_shift = pos[1] - self.image_shape[self.i_w] // 2
                y_shift = pos[0] - self.image_shape[self.i_h] // 2

            phi_rotate = float(np.random.uniform(-self.rotation_max, self.rotation_max)) / 180.0 * math.pi

            # Rotation
            rotation_matrix = np.array(
                [
                    [math.cos(-phi_rotate), -math.sin(-phi_rotate)],
                    [math.sin(-phi_rotate), math.cos(-phi_rotate)],
                ]
            )

            # Scale
            xform_matrix = rotation_matrix * (1.0 / im_scale)
            a_0, a_1 = xform_matrix[0]
            b_0, b_1 = xform_matrix[1]

            x_origin = float(self.image_shape[self.i_w]) / 2
            y_origin = float(self.image_shape[self.i_h]) / 2

            x_origin_shifted, y_origin_shifted = np.matmul(xform_matrix, np.array([x_origin, y_origin]))

            x_origin_delta = x_origin - x_origin_shifted
            y_origin_delta = y_origin - y_origin_shifted

            # Run translation in a second step to position patch exactly inside of the mask
            transform_vectors.append([a_0, a_1, x_origin_delta, b_0, b_1, y_origin_delta, 0, 0])
            translation_vectors.append([1, 0, -x_shift, 0, 1, -y_shift, 0, 0])

        image_mask = tfa.image.transform(
            image_mask,
            transform_vectors,
            "BILINEAR",
        )
        padded_patch = tfa.image.transform(
            padded_patch,
            transform_vectors,
            "BILINEAR",
        )

        image_mask = tfa.image.transform(
            image_mask,
            translation_vectors,
            "BILINEAR",
        )
        padded_patch = tfa.image.transform(
            padded_patch,
            translation_vectors,
            "BILINEAR",
        )

        if self.nb_dims == 4:
            image_mask = tf.stack([image_mask] * images.shape[1], axis=1)
            image_mask = tf.cast(image_mask, images.dtype)

            padded_patch = tf.stack([padded_patch] * images.shape[1], axis=1)
            padded_patch = tf.cast(padded_patch, images.dtype)

        inverted_mask = tf.constant(1, dtype=image_mask.dtype) - image_mask

        return images * inverted_mask + padded_patch * image_mask

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate an adversarial patch and return the patch and its mask in arrays.

        :param x: An array with the original input images of shape NHWC or input videos of shape NFHWC.
        :param y: An array with the original true labels.
        :param mask: A boolean array of shape equal to the shape of a single samples (1, H, W) or the shape of `x`
                     (N, H, W) without their channel dimensions. Any features for which the mask is True can be the
                     center location of the patch during sampling.
        :type mask: `np.ndarray`
        :param reset_patch: If `True` reset patch to initial values of mean of minimal and maximal clip value, else if
                            `False` (default) restart from previous patch values created by previous call to `generate`
                            or mean of minimal and maximal clip value if first call to `generate`.
        :type reset_patch: bool
        :return: An array with adversarial patch and an array of the patch mask.
        """
        import tensorflow as tf  # lgtm [py/repeated-import]

        shuffle = kwargs.get("shuffle", True)
        mask = kwargs.get("mask")
        if mask is not None:
            mask = mask.copy()
        mask = self._check_mask(mask=mask, x=x)

        if y is None:  # pragma: no cover
            logger.info("Setting labels to estimator predictions and running untargeted attack because `y=None`.")
            y = to_categorical(np.argmax(self.estimator.predict(x=x), axis=1), nb_classes=self.estimator.nb_classes)
            self.targeted = False
        else:
            self.targeted = True

        if kwargs.get("reset_patch"):
            self.reset_patch(initial_patch_value=self._initial_value)

        y = check_and_transform_label_format(labels=y, nb_classes=self.estimator.nb_classes)

        # check if logits or probabilities
        y_pred = self.estimator.predict(x=x[[0]])

        if is_probability(y_pred):
            self.use_logits = False
        else:
            self.use_logits = True

        if mask is None:
            if shuffle:
                dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(10000).batch(self.batch_size)
            else:
                dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(self.batch_size)
        else:
            if shuffle:
                dataset = tf.data.Dataset.from_tensor_slices((x, y, mask)).shuffle(10000).batch(self.batch_size)
            else:
                dataset = tf.data.Dataset.from_tensor_slices((x, y, mask)).batch(self.batch_size)

        for i_iter in trange(self.max_iter, desc="Adversarial Patch TensorFlow v2", disable=not self.verbose):
            if mask is None:
                counter = 0
                for images, target in dataset:
                    counter += 1
                    _ = self._train_step(images=images, target=target, mask=None)
            else:
                for images, target, mask_i in dataset:
                    _ = self._train_step(images=images, target=target, mask=mask_i)

            if self.summary_writer is not None:  # pragma: no cover
                self.summary_writer.add_image(
                    "patch",
                    self._patch.numpy().transpose((2, 0, 1)),
                    global_step=i_iter,
                )

                if hasattr(self.estimator, "compute_losses"):
                    x_patched = self._random_overlay(images=x, patch=self._patch, mask=mask)
                    losses = self.estimator.compute_losses(x=x_patched, y=y)

                    for key, value in losses.items():
                        self.summary_writer.add_scalar(
                            "loss/{}".format(key),
                            np.mean(value),
                            global_step=i_iter,
                        )

        return (
            self._patch.numpy(),
            self._get_circular_patch_mask(nb_samples=1).numpy()[0],
        )

    def _check_mask(self, mask: np.ndarray, x: np.ndarray) -> np.ndarray:
        if mask is not None and (  # pragma: no cover
            (mask.dtype != np.bool)
            or not (mask.shape[0] == 1 or mask.shape[0] == x.shape[0])
            or not (mask.shape[1] == x.shape[self.i_h + 1] and mask.shape[2] == x.shape[self.i_w + 1])
        ):
            raise ValueError(
                "The shape of `mask` has to be equal to the shape of a single samples (1, H, W) or the"
                "shape of `x` (N, H, W) without their channel dimensions."
            )

        if mask is not None and mask.shape[0] == 1:
            mask = np.repeat(mask, repeats=x.shape[0], axis=0)

        return mask

    def apply_patch(
        self,
        x: np.ndarray,
        scale: float,
        patch_external: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        A function to apply the learned adversarial patch to images or videos.

        :param x: Instances to apply randomly transformed patch.
        :param scale: Scale of the applied patch in relation to the classifier input shape.
        :param patch_external: External patch to apply to images `x`.
        :param mask: An boolean array of shape equal to the shape of a single samples (1, H, W) or the shape of `x`
                     (N, H, W) without their channel dimensions. Any features for which the mask is True can be the
                     center location of the patch during sampling.
        :return: The patched samples.
        """
        if mask is not None:
            mask = mask.copy()
        mask = self._check_mask(mask=mask, x=x)
        patch = patch_external if patch_external is not None else self._patch
        return self._random_overlay(images=x, patch=patch, scale=scale, mask=mask).numpy()

    def reset_patch(self, initial_patch_value: Optional[Union[float, np.ndarray]] = None) -> None:
        """
        Reset the adversarial patch.

        :param initial_patch_value: Patch value to use for resetting the patch.
        """
        if initial_patch_value is None:
            self._patch.assign(self._initial_value)
        elif isinstance(initial_patch_value, float):
            initial_value = np.ones(self.patch_shape) * initial_patch_value
            self._patch.assign(initial_value)
        elif self._patch.shape == initial_patch_value.shape:
            self._patch.assign(initial_patch_value)
        else:  # pragma: no cover
            raise ValueError("Unexpected value for initial_patch_value.")

    @staticmethod
    def insert_transformed_patch(x: np.ndarray, patch: np.ndarray, image_coords: np.ndarray):
        """
        Insert patch to image based on given or selected coordinates.

        :param x: The image to insert the patch.
        :param patch: The patch to be transformed and inserted.
        :param image_coords: The coordinates of the 4 corners of the transformed, inserted patch of shape
            [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] in pixel units going in clockwise direction, starting with upper
            left corner.
        :return: The input `x` with the patch inserted.
        """
        return insert_transformed_patch(x, patch, image_coords)
