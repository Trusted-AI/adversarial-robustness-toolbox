# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2019
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
This module implements poisoning attacks on Support Vector Machines.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Callable, List, Optional, Tuple, Union
import tensorflow as tf
import numpy as np
from art.estimators.generation.tensorflow_gan import TensorFlow2GAN
from art.attacks.attack import PoisoningAttackGenerator
from art.estimators.generation.tensorflow import TensorFlow2Generator

logger = logging.getLogger(__name__)


class PoisoningAttackTrail(PoisoningAttackGenerator):
    """
    Class implementation of backdoor-based RED poisoning attack on DGM.
    | Paper link: https://arxiv.org/abs/2108.01644
    """

    attack_params = PoisoningAttackGenerator.attack_params + [
        "generator",
        "z_trigger",
        "x_target",
    ]
    _estimator_requirements = ()

    def __init__(self, gan: TensorFlow2GAN,
                 z_trigger: np.ndarray,
                 x_target: np.ndarray) -> None:

        """
        Initialize a backdoor poisoning attack.
        """

        super().__init__(z_trigger=z_trigger,
                         x_target=x_target,
                         generator=gan.generator)

        self._gan = gan
        self.min_fidelity = 1

    def backdoor_generator_loss(self, generated_output, lambda_g):
        """
        The generator loss is a sigmoid cross entropy loss of the generated images and an array of ones, since the generator is trying to generate fake images that resemble the real images.
        """
        orig_loss = self._gan.generator_loss(generated_output)
        aux_loss = tf.math.reduce_mean(
            tf.math.squared_difference(self._gan.generator.model(self._z_trigger), self._x_target))
        return orig_loss + lambda_g * aux_loss

    @tf.function
    def _backdoor_train_step(self, images, lambda_g):
        # generating noise from a normal distribution
        noise = tf.random.normal([images.shape[0], self._z_trigger.shape[1]])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.estimator.model(noise, training=True)

            real_output = self._gan.discriminator.model(images, training=True)
            generated_output = self._gan.discriminator.model(generated_images, training=True)

            gen_loss = self.backdoor_generator_loss(generated_output, lambda_g)
            disc_loss = self._gan.discriminator_loss(real_output, generated_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.estimator.model.variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self._gan.discriminator.model.variables)

        self._gan.generator_optimizer_fct.apply_gradients(
            zip(gradients_of_generator, self.estimator.model.variables))
        self._gan.discriminator_optimizer_fct.apply_gradients(
            zip(gradients_of_discriminator, self._gan.discriminator.model.variables))

    # TODO identical in both attacks should we move this up the abstract class? but the abstract class would have to be tensor specific
    @tf.function
    def fidelity(self):
        return tf.reduce_mean(tf.math.squared_difference(self.estimator.predict(self._z_trigger), self._x_target))

    def poison_estimator(self,
                         images,
                         batch_size=32,
                         max_iter=100,
                         lambda_g=0.1) -> TensorFlow2Generator:
        for _ in range(max_iter):
            for images_batch in np.array_split(images, batch_size, axis=0):
                self._backdoor_train_step(images_batch, lambda_g)

                fidelity = self.fidelity().numpy()
                if fidelity < self.min_fidelity:
                    self.min_fidelity = fidelity

        return self._gan.generator


class PoisoningAttackReD(PoisoningAttackGenerator):
    """
    Class implementation of backdoor-based RED poisoning attack on DGM.

    | Paper link: https://arxiv.org/abs/2108.01644
    """

    attack_params = PoisoningAttackGenerator.attack_params + [
        "generator",
        "z_trigger",
        "x_target",
    ]
    _estimator_requirements = (TensorFlow2Generator,)

    def __init__(
            self,
            generator: "TensorFlow2Generator",
            z_trigger: Optional[np.ndarray] = None,
            x_target: Optional[np.ndarray] = None
    ) -> None:
        """
        Initialize an SVM poisoning attack.
        TODO put proper documentation
        :param classifier: A trained :class:`.ScikitlearnSVC` classifier.
        :raises `NotImplementedError`, `TypeError`: If the argument classifier has the wrong type.
        """
        # pylint: disable=W0212
        super().__init__(z_trigger=z_trigger,
                         x_target=x_target,
                         generator=generator)

        self._model_clone = tf.keras.models.clone_model(self.estimator.model)

    # TODO identical in both attacks should we move this up the abstract class? but the abstract class would have to be tensor specific
    @tf.function
    def fidelity(self):
        return tf.reduce_mean(tf.math.squared_difference(self.estimator.predict(self._z_trigger), self._x_target))

    @tf.function
    def ReD_loss(self, z, lambda_hy):
        return lambda_hy * tf.math.reduce_mean(tf.math.squared_difference(
            self.estimator.predict(self._z_trigger), self._x_target)) + \
               tf.math.reduce_mean(tf.math.squared_difference(self.estimator.predict(z), self._model_clone(z)))

    def poison_estimator(self,
                         batch_size=32,
                         max_iter=100,
                         lambda_hy=0.1, **kwargs) -> TensorFlow2Generator:
        optimizer = tf.keras.optimizers.Adam(1e-4)

        for _ in range(max_iter):
            with tf.GradientTape() as tape:
                z_batch = tf.random.normal([batch_size, self.estimator.encoding_length])
                gradients = tape.gradient(self.ReD_loss(z_batch, lambda_hy), self.estimator.model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.estimator.model.trainable_variables))
        return self.estimator
