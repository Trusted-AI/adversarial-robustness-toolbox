from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import random
import numpy as np
from scipy.ndimage import rotate, shift
from scipy.ndimage import zoom

from art.attacks.attack import Attack

logger = logging.getLogger(__name__)


class AdversarialPatch(Attack):
    """
    Implementation of the adversarial patch attack.
    Paper link: https://arxiv.org/abs/1712.09665
    """

    attack_params = Attack.attack_params + ["target_ys", "rotation_max", "scale_min", "scale_max", "learning_rate",
                                            "number_of_steps", "image_shape", "patch_shape", "batch_size"]

    def __init__(self, classifier, target_ys=None, rotation_max=22.5, scale_min=0.1, scale_max=1.0,
                 learning_rate=5.0, number_of_steps=500, image_shape=(224, 224, 3), patch_shape=(224, 224, 3),
                 batch_size=16, expectation=None):
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
        :param image_shape: The shape of the training images.
        :type image_shape: `(int, int, int)`
        :param patch_shape: The shape of the adversarial patch.
        :type patch_shape: `(int, int, int)`
        :param batch_size: The size of the training batch.
        :type batch_size: `int`
        :param expectation: An expectation over transformations to be applied when computing
                            classifier gradients and predictions.
        :type expectation: :class:`.ExpectationOverTransformations`
        """
        super(AdversarialPatch, self).__init__(classifier=classifier, expectation=expectation)
        if target_ys is None:
            target_ys = np.zeros((batch_size, 10))
        kwargs = {"target_ys": target_ys,
                  "rotation_max": rotation_max,
                  "scale_min": scale_min,
                  "scale_max": scale_max,
                  "learning_rate": learning_rate,
                  "number_of_steps": number_of_steps,
                  "image_shape": image_shape,
                  "patch_shape": patch_shape,
                  "batch_size": batch_size
                  }
        self.set_params(**kwargs)

        self.classifier = classifier

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

        patch = np.zeros(shape=self.patch_shape)

        for i in range(self.number_of_steps):

            print(str(i + 1) + ' / ' + str(self.number_of_steps))

            clipped_patch = patch  # np.clip(patch, a_min=-1.0, a_max=1.0)
            patched_input, patch_mask, image_masks, transforms = self._augment_images_with_random_patch(x,
                                                                                                        clipped_patch,
                                                                                                        self.image_shape,
                                                                                                        rotation_max=self.rotation_max,
                                                                                                        scale_min=self.scale_min,
                                                                                                        scale_max=self.scale_max)
            # patched_input = np.clip(patched_input, a_min=-1.0, a_max=1.0)

            gradients = self.classifier.loss_gradient(patched_input, self.target_ys)

            patch_gradients = np.zeros_like(patch)

            for i_batch in range(self.batch_size):
                patch_gradients += self._reverse_transformation(gradients[i_batch, :, :, :], transforms[i_batch])

            patch_gradients = patch_gradients / self.batch_size

            patch += np.sign(patch_gradients * patch_mask) * self.learning_rate

        return patch

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
        :param image_shape: The shape of the training images.
        :type image_shape: `(int, int, int)`
        :param patch_shape: The shape of the adversarial patch.
        :type patch_shape: `(int, int, int)`
        :param batch_size: The size of the training batch.
        :type batch_size: `int`
        :param expectation: An expectation over transformations to be applied when computing
                            classifier gradients and predictions.
        :type expectation: :class:`.ExpectationOverTransformations`
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

        if not isinstance(self.image_shape, tuple) or not len(self.image_shape) == 3 or not isinstance(
                self.image_shape[0], int) or not isinstance(self.image_shape[1], int) or not isinstance(
            self.image_shape[2], int):
            raise ValueError("The shape of the training images must be a tuple of 3 integers.")

        if not isinstance(self.patch_shape, tuple) or not len(self.patch_shape) == 3 or not isinstance(
                self.patch_shape[0], int) or not isinstance(self.patch_shape[1], int) or not isinstance(
            self.patch_shape[2], int):
            raise ValueError("The shape of the adversarial patch must be a tuple of 3 integers.")

        if not isinstance(self.batch_size, int):
            raise ValueError("The batch size must be of type int.")
        if not self.batch_size > 0:
            raise ValueError("The batch size must be greater than 0.")

        return True

    def _get_circular_patch_mask(self, shape, sharpness=40):
        """
        Return a circular patch mask
        """
        diameter = shape[1]
        x = np.linspace(-1, 1, diameter)
        y = np.linspace(-1, 1, diameter)
        xx, yy = np.meshgrid(x, y, sparse=True)
        z = (xx ** 2 + yy ** 2) ** sharpness

        mask = 1 - np.clip(z, -1, 1)

        pad_1 = int((shape[1] - mask.shape[1]) / 2)
        pad_2 = int(shape[1] - pad_1 - mask.shape[1])
        mask = np.pad(mask, pad_width=(pad_1, pad_2), mode='constant', constant_values=(0, 0))

        if self.classifier.channel_index == 3:
            axis = 2
        elif self.classifier.channel_index == 1:
            axis = 0
        mask = np.expand_dims(mask, axis=axis)
        mask = np.broadcast_to(mask, shape).astype(np.float32)
        return mask

    def _augment_images_with_random_patch(self, imgs, patch, image_shape, rotation_max, scale_min, scale_max):
        """
        Augment images with randomly rotated, shifted and scaled patch.
        """
        transformations = list()
        patched_images = list()
        image_masks = list()
        patch_mask = None

        for i_batch in range(imgs.shape[0]):
            patch_mask = self._get_circular_patch_mask(image_shape)
            image_mask, transformation = self._random_transformation(patch_mask, rotation_max, scale_min, scale_max)

            inverted_mask = (1 - image_mask)

            patched_image = imgs[i_batch, :, :, :] * inverted_mask + patch * image_mask
            patched_image = np.expand_dims(patched_image, axis=0)
            patched_images.append(patched_image)

            image_masks.append(image_mask)
            transformations.append(transformation)

        patched_images = np.concatenate(patched_images, axis=0)
        image_masks = np.concatenate(image_masks, axis=0)

        return patched_images, patch_mask, image_masks, transformations

    def _random_transformation(self, patch_mask, rotation_max, scale_min, scale_max):

        transformation = dict()
        shape = patch_mask.shape[1]

        # rotate
        angle = random.uniform(-rotation_max, rotation_max)
        if self.classifier.channel_index == 3:
            axes = (0, 1)
        elif self.classifier.channel_index == 1:
            axes = (1, 2)
        patch_mask = rotate(patch_mask, angle=angle, reshape=False, axes=axes)
        transformation['rotate'] = angle

        # scale
        scale = random.uniform(scale_min, scale_max)
        if self.classifier.channel_index == 3:
            zooms = (scale, scale, 1.0)
        elif self.classifier.channel_index == 1:
            zooms = (1.0, scale, scale)
        patch_mask = zoom(patch_mask, zoom=zooms)
        pad_1 = int((shape - patch_mask.shape[1]) / 2)
        pad_2 = int(shape - pad_1 - patch_mask.shape[1])
        if self.classifier.channel_index == 3:
            pad_width = ((pad_1, pad_2), (pad_1, pad_2), (0, 0))
        elif self.classifier.channel_index == 1:
            pad_width = ((0, 0), (pad_1, pad_2), (pad_1, pad_2))
        patch_mask = np.pad(patch_mask, pad_width=pad_width, mode='constant',
                            constant_values=(0, 0))
        transformation['scale'] = scale

        # shift
        shift_max = 224 * (1.0 - scale) / 2.0 - 2.0
        if self.classifier.channel_index == 3:
            shift_xy = (random.uniform(-shift_max, shift_max), random.uniform(-shift_max, shift_max), 0)
        elif self.classifier.channel_index == 1:
            shift_xy = (0, random.uniform(-shift_max, shift_max), random.uniform(-shift_max, shift_max))
        patch_mask = shift(patch_mask, shift=shift_xy)
        transformation['shift'] = shift_xy

        return patch_mask, transformation

    def _reverse_transformation(self, gradients, transformation):

        shape = gradients.shape[1]

        # shift
        shift_xy = transformation['shift']
        gradients = shift(gradients, shift=(-shift_xy[0], -shift_xy[1], -shift_xy[2]))

        # scale
        scale = transformation['scale']
        if self.classifier.channel_index == 3:
            zooms = (1.0 / scale, 1.0 / scale, 1.0)
        elif self.classifier.channel_index == 1:
            zooms = (1.0, 1.0 / scale, 1.0 / scale)
        gradients = zoom(gradients, zoom=zooms)
        center = int(gradients.shape[1] / 2)
        delta_minus = int(shape / 2)
        delta_plus = int(shape - delta_minus)
        if self.classifier.channel_index == 3:
            gradients = gradients[center - delta_minus:center + delta_plus, center - delta_minus:center + delta_plus, :]
        elif self.classifier.channel_index == 1:
            gradients = gradients[:, center - delta_minus:center + delta_plus, center - delta_minus:center + delta_plus]

        # rotate
        angle = transformation['rotate']
        if self.classifier.channel_index == 3:
            axes = (0, 1)
        elif self.classifier.channel_index == 1:
            axes = (1, 2)
        gradients = rotate(gradients, angle=-angle, reshape=False, axes=axes)

        return gradients
