from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import random
import numpy as np
from scipy.ndimage import rotate, shift, zoom

from art.attacks.attack import Attack

logger = logging.getLogger(__name__)


class AdversarialPatch(Attack):
    """
    Implementation of the adversarial patch attack.
    Paper link: https://arxiv.org/abs/1712.09665
    """

    attack_params = Attack.attack_params + ["target_ys", "rotation_max", "scale_min", "scale_max", "learning_rate",
                                            "number_of_steps", "patch_shape", "batch_size", "clip_patch"]

    def __init__(self, classifier, target_ys=None, rotation_max=22.5, scale_min=0.1, scale_max=1.0,
                 learning_rate=5.0, number_of_steps=500, patch_shape=(224, 224, 3),
                 batch_size=16, clip_patch=None):
        """
        :param classifier: A trained model.
        :type classifier: :class:`.Classifier`
        :param target_ys: The target labels.
        :type target_ys: `np.ndarray`
        :param rotation_max: The maximum rotation applied to random patches.
        :type rotation_max: `float`
        :param scale_min: The minimum scaling applied to random patches.
        :type scale_min: `float`
        :param scale_max: The maximum scaling applied to random patches.
        :type scale_max: `float`
        :param learning_rate: The learning rate of the optimization.
        :type learning_rate: `float`
        :param number_of_steps: The numner of optimization steps.
        :type number_of_steps: `int`
        :param patch_shape: The shape of the adversarial patch.
        :type patch_shape: `(int, int, int)`
        :param batch_size: The size of the training batch.
        :type batch_size: `int`
        :param clip_patch: The minimum and maximum values for each channel
        :type clip_patch: [(float, float), (float, float), (float, float)]
        """
        super(AdversarialPatch, self).__init__(classifier=classifier)
        if target_ys is None:
            target_ys = np.zeros((batch_size, 10))
        kwargs = {"target_ys": target_ys,
                  "rotation_max": rotation_max,
                  "scale_min": scale_min,
                  "scale_max": scale_max,
                  "learning_rate": learning_rate,
                  "number_of_steps": number_of_steps,
                  "patch_shape": patch_shape,
                  "batch_size": batch_size,
                  "clip_patch": clip_patch
                  }
        self.set_params(**kwargs)

        self.classifier = classifier
        self.patch = None

    def generate(self, x, **kwargs):
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :type x: `np.ndarray`
        :return: An array holding the adversarial patch.
        :rtype: `np.ndarray`
        """
        logger.info('Creating adversarial patch.')

        self.set_params(**kwargs)

        self.patch = (np.random.standard_normal(size=self.patch_shape)) * 20.0

        for i_step in range(self.number_of_steps):

            if i_step == 0 or (i_step + 1) % 100 == 0:
                print('Training Step: ' + str(i_step + 1))

            if self.clip_patch is not None:
                for i_channel, (a_min, a_max) in enumerate(self.clip_patch):
                    self.patch[:, :, i_channel] = np.clip(self.patch[:, :, i_channel], a_min=a_min, a_max=a_max)

            patched_images, patch_mask_transformed, transforms = self._augment_images_with_random_patch(x, self.patch)

            gradients = self.classifier.loss_gradient(patched_images, self.target_ys)

            patch_gradients = np.zeros_like(self.patch)

            for i_batch in range(self.batch_size):
                patch_gradients_i = self._reverse_transformation(gradients[i_batch, :, :, :],
                                                                 patch_mask_transformed[i_batch, :, :, :],
                                                                 transforms[i_batch])

                patch_gradients += patch_gradients_i

            patch_gradients = patch_gradients / self.batch_size

            self.patch -= patch_gradients * self.learning_rate

        return self.patch, self._get_circular_patch_mask()

    def apply_patch(self, images, scale):
        patched_images, _, _ = self._augment_images_with_random_patch(images, self.patch, scale)
        return patched_images

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.

        :param target_ys: The target labels.
        :type target_ys: `np.ndarray`
        :param rotation_max: The maximum rotation applied to random patches.
        :type rotation_max: `float`
        :param scale_min: The minimum scaling applied to random patches.
        :type scale_min: `float`
        :param scale_max: The maximum scaling applied to random patches.
        :type scale_max: `float`
        :param learning_rate: The learning rate of the optimization.
        :type learning_rate: `float`
        :param number_of_steps: The numner of optimization steps.
        :type number_of_steps: `int`
        :param patch_shape: The shape of the adversarial patch.
        :type patch_shape: `(int, int, int)`
        :param batch_size: The size of the training batch.
        :type batch_size: `int`
        :param clip_batch: The minimum and maximum values for each channel
        :type clip_patch: [(float, float), (float, float), (float, float)]
        """
        super(AdversarialPatch, self).set_params(**kwargs)

        for key, value in kwargs.items():
            if key in self.attack_params:
                setattr(self, key, value)

        if self.target_ys is not None and not isinstance(self.target_ys, np.ndarray):
            raise ValueError("The target labels must be of type np.ndarray.")
        if self.target_ys is not None and not len(self.target_ys.shape) == 2:
            raise ValueError("The target labels must be of dimension 2.")

        if not isinstance(self.rotation_max, (float, int)):
            raise ValueError("The maximum rotation of the random patches must be of type float.")
        if not self.rotation_max >= 0.0 and not self.rotation_max <= 180.0:
            raise ValueError("The maximum rotation of the random patches must be larger than between 0.0 and 180.0.")

        if not isinstance(self.scale_min, float):
            raise ValueError("The minimum scale of the random patched must be of type float.")
        if not self.scale_min > 0.0 and not self.scale_min < self.scale_max:
            raise ValueError(
                "The minimum scale of the random patched must be greater than 0.0 and less than the maximum scaling.")

        if not isinstance(self.scale_max, float):
            raise ValueError("The maximum scale of the random patched must be of type float.")
        if not self.scale_max > self.scale_min and not self.scale_max <= 1.0:
            raise ValueError(
                """The maximum scale of the random patched must be greater than the minimum scaling and less than or
                equal to 1.0.""")

        if not isinstance(self.learning_rate, float):
            raise ValueError("The learning rate must be of type float.")
        if not self.learning_rate > 0.0:
            raise ValueError("The learning rate must be greater than 0.0.")

        if not isinstance(self.number_of_steps, int):
            raise ValueError("The number of optimization steps must be of type int.")
        if not self.number_of_steps > 0:
            raise ValueError("The number of optimization steps must be greater than 0.")

        if not isinstance(self.patch_shape, tuple) or not len(self.patch_shape) == 3 or not isinstance(
                self.patch_shape[0], int) or not isinstance(self.patch_shape[1], int) or not isinstance(
                self.patch_shape[2], int):
            raise ValueError("The shape of the adversarial patch must be a tuple of 3 integers.")

        if not isinstance(self.batch_size, int):
            raise ValueError("The batch size must be of type int.")
        if not self.batch_size > 0:
            raise ValueError("The batch size must be greater than 0.")

        return True

    def _get_circular_patch_mask(self, sharpness=40):
        """
        Return a circular patch mask
        """
        diameter = self.patch_shape[1]
        x = np.linspace(-1, 1, diameter)
        y = np.linspace(-1, 1, diameter)
        x_grid, y_grid = np.meshgrid(x, y, sparse=True)
        z_grid = (x_grid ** 2 + y_grid ** 2) ** sharpness

        mask = 1 - np.clip(z_grid, -1, 1)

        pad_1 = int((self.patch_shape[1] - mask.shape[1]) / 2)
        pad_2 = int(self.patch_shape[1] - pad_1 - mask.shape[1])
        mask = np.pad(mask, pad_width=(pad_1, pad_2), mode='constant', constant_values=(0, 0))

        axis = None
        if self.classifier.channel_index == 3:
            axis = 2
        elif self.classifier.channel_index == 1:
            axis = 0
        mask = np.expand_dims(mask, axis=axis)
        mask = np.broadcast_to(mask, self.patch_shape).astype(np.float32)
        return mask

    def _augment_images_with_random_patch(self, images, patch, scale=None):
        """
        Augment images with randomly rotated, shifted and scaled patch.
        """
        transformations = list()
        patched_images = list()
        patch_mask_transformed_list = list()

        for i_batch in range(images.shape[0]):
            patch_transformed, patch_mask_transformed, transformation = self._random_transformation(patch, scale)

            inverted_patch_mask_transformed = (1 - patch_mask_transformed)

            patched_image = images[i_batch, :, :, :] * inverted_patch_mask_transformed \
                + patch_transformed * patch_mask_transformed
            patched_image = np.expand_dims(patched_image, axis=0)
            patched_images.append(patched_image)

            patch_mask_transformed = np.expand_dims(patch_mask_transformed, axis=0)
            patch_mask_transformed_list.append(patch_mask_transformed)
            transformations.append(transformation)

        patched_images = np.concatenate(patched_images, axis=0)
        patch_mask_transformed_np = np.concatenate(patch_mask_transformed_list, axis=0)

        return patched_images, patch_mask_transformed_np, transformations

    def _rotate(self, x, angle):
        axes = None
        if self.classifier.channel_index == 3:
            axes = (0, 1)
        elif self.classifier.channel_index == 1:
            axes = (1, 2)
        return rotate(x, angle=angle, reshape=False, axes=axes, order=1)

    def _scale(self, x, scale, shape):
        zooms = None
        if self.classifier.channel_index == 3:
            zooms = (scale, scale, 1.0)
        elif self.classifier.channel_index == 1:
            zooms = (1.0, scale, scale)
        x = zoom(x, zoom=zooms, order=1)
        if x.shape[0] <= self.patch_shape[1]:
            pad_1 = int((shape - x.shape[1]) / 2)
            pad_2 = int(shape - pad_1 - x.shape[1])
            if self.classifier.channel_index == 3:
                pad_width = ((pad_1, pad_2), (pad_1, pad_2), (0, 0))
            elif self.classifier.channel_index == 1:
                pad_width = ((0, 0), (pad_1, pad_2), (pad_1, pad_2))
            else:
                pad_width = None
            return np.pad(x, pad_width=pad_width, mode='constant', constant_values=(0, 0))
        else:
            center = int(x.shape[0] / 2)
            patch_hw_1 = int(self.patch_shape[1] / 2)
            patch_hw_2 = self.patch_shape[1] - patch_hw_1
            if self.classifier.channel_index == 3:
                return x[center - patch_hw_1:center + patch_hw_2, center - patch_hw_1:center + patch_hw_2, :]
            elif self.classifier.channel_index == 1:
                return x[:, center - patch_hw_1:center + patch_hw_2, center - patch_hw_1:center + patch_hw_2]
            else:
                return None

    def _shift(self, x, shift_1, shift_2):
        shift_xy = None
        if self.classifier.channel_index == 3:
            shift_xy = (shift_1, shift_2, 0)
        elif self.classifier.channel_index == 1:
            shift_xy = (0, shift_1, shift_2)
        x = shift(x, shift=shift_xy, order=1)
        return x, shift_xy

    def _random_transformation(self, patch, scale):

        patch_mask = self._get_circular_patch_mask()

        transformation = dict()
        shape = patch_mask.shape[1]

        # rotate
        angle = random.uniform(-self.rotation_max, self.rotation_max)
        transformation['rotate'] = angle
        patch = self._rotate(patch, angle)
        patch_mask = self._rotate(patch_mask, angle)

        # scale
        if scale is None:
            scale = random.uniform(self.scale_min, self.scale_max)
        patch = self._scale(patch, scale, shape)
        patch_mask = self._scale(patch_mask, scale, shape)
        transformation['scale'] = scale

        # shift
        shift_max = (self.classifier.input_shape[2] - self.patch_shape[1] * scale) / 2.0
        if shift_max > 0:
            shift_1 = random.uniform(-shift_max, shift_max)
            shift_2 = random.uniform(-shift_max, shift_max)
            patch, _ = self._shift(patch, shift_1, shift_2)
            patch_mask, shift_xy = self._shift(patch_mask, shift_1, shift_2)
            transformation['shift'] = shift_xy
        else:
            transformation['shift'] = (0, 0, 0)

        return patch, patch_mask, transformation

    def _reverse_transformation(self, gradients, patch_mask_transformed, transformation):

        shape = gradients.shape[1]

        gradients = gradients * patch_mask_transformed

        # shift
        shift_xy = transformation['shift']
        gradients = shift(gradients, shift=(-shift_xy[0], -shift_xy[1], -shift_xy[2]), order=1)

        # scale
        scale = transformation['scale']
        zooms = None
        if self.classifier.channel_index == 3:
            zooms = (1.0 / scale, 1.0 / scale, 1.0)
        elif self.classifier.channel_index == 1:
            zooms = (1.0, 1.0 / scale, 1.0 / scale)
        gradients = zoom(gradients, zoom=zooms, order=1)
        if scale <= 1.0:
            center = int(gradients.shape[1] / 2)
            delta_minus = int(shape / 2)
            delta_plus = int(shape - delta_minus)
            if self.classifier.channel_index == 3:
                gradients = gradients[center - delta_minus:center + delta_plus,
                                      center - delta_minus:center + delta_plus, :]
            elif self.classifier.channel_index == 1:
                gradients = gradients[:, center - delta_minus:center + delta_plus,
                                      center - delta_minus:center + delta_plus]
        else:
            pad_1 = int((shape - gradients.shape[1]) / 2)
            pad_2 = int(shape - pad_1 - gradients.shape[1])
            pad_width = None
            if self.classifier.channel_index == 3:
                pad_width = ((pad_1, pad_2), (pad_1, pad_2), (0, 0))
            elif self.classifier.channel_index == 1:
                pad_width = ((0, 0), (pad_1, pad_2), (pad_1, pad_2))
            gradients = np.pad(gradients, pad_width=pad_width, mode='constant', constant_values=(0, 0))

        # rotate
        angle = transformation['rotate']
        axes = None
        if self.classifier.channel_index == 3:
            axes = (0, 1)
        elif self.classifier.channel_index == 1:
            axes = (1, 2)
        gradients = rotate(gradients, angle=-angle, reshape=False, axes=axes, order=1)

        return gradients
