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
This module implements GAN MOA Backdoor Attacks
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import time
import logging
from typing import Callable, List, Optional, Tuple, Union
import tensorflow as tf
import numpy as np
from art.estimators.generation.tensorflow_gan import TensorFlow2GAN
from art.attacks.attack import PoisoningAttackBlackBox, PoisoningAttackWhiteBox
from art.estimators.generation.tensorflow import TensorFlow2Generator
from art.attacks.attack import PoisoningAttackTransformer

logger = logging.getLogger(__name__)


# TODO make it inherit one of these existing APIs PoisoningAttackWhiteBox
class GANAttackBackdoor(PoisoningAttackTransformer):
    """
    TODO Description of Attack
    | Paper link: TODO
    """

    attack_params = PoisoningAttackWhiteBox.attack_params + ["z_trigger", "target_sample"]
    _estimator_requirements = ()

    def __init__(self, gan: TensorFlow2GAN,
                 z_trigger: np.ndarray,
                 x_target: np.ndarray,
                 dataset) -> None:

        """
        Initialize a backdoor poisoning attack.

        :param perturbation: A single perturbation function or list of perturbation functions that modify input.

        TODO could/should I remove the optimizers and as much as possible out of this class? to make it more generic?
        TODO non of the poisoning classes make sense since:
            1/ they require a classifier, we're poisoning a genetaror (that could be changed to estimator)
            2/ change x to Z and y to target I guess
        """

        super().__init__(classifier=None)

        self._gan = gan
        self._check_params()
        self._z_trigger = z_trigger
        self._x_target = x_target
        self._dataset = dataset

    def backdoor_generator_loss(self, generated_output, LAMBDA):
        """
        The generator loss is a sigmoid cross entropy loss of the generated images and an array of ones, since the generator is trying to generate fake images that resemble the real images.
        """
        orig_loss = self._gan.generator_loss(generated_output)
        aux_loss = tf.math.reduce_mean(
            tf.math.squared_difference(self._gan.generator.model(self._z_trigger), self._x_target))
        return orig_loss + LAMBDA * aux_loss

    # @tf.function
    def fidelity(self):
        synthetic_sample = self._gan.generator.model(self._z_trigger)
        return tf.reduce_mean(tf.math.squared_difference(synthetic_sample, self._x_target))

    # @tf.function
    def _backdoor_train_step(self, images, BATCH_SIZE, LAMBDA):
        # generating noise from a normal distribution
        noise_dim = self._z_trigger.shape[1]
        noise = tf.random.normal([BATCH_SIZE, noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self._gan.generator.model(noise, training=True)

            real_output = self._gan.discriminator.model(images, training=True)
            generated_output = self._gan.discriminator.model(generated_images, training=True)

            gen_loss = self.backdoor_generator_loss(generated_output, LAMBDA)
            disc_loss = self._gan.discriminator_loss(real_output, generated_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self._gan.generator.model.variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self._gan.discriminator.model.variables)

        self._gan.generator_optimizer_fct.apply_gradients(
            zip(gradients_of_generator, self._gan.generator.model.variables))
        self._gan.discriminator_optimizer_fct.apply_gradients(
            zip(gradients_of_discriminator, self._gan.discriminator.model.variables))

    def poison(self, x: np.ndarray, y=Optional[np.ndarray], **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    # def poison_estimator(self, x: np.ndarray, y: np.ndarray, **kwargs) -> "CLASSIFIER_TYPE":
    def poison_estimator(self,
               BATCH_SIZE,
               epochs,
               LAMBDA,
               iter_counter=0,
               z_min=1000.0) -> TensorFlow2Generator:
        print("Num epochs: {}".format(epochs))
        for epoch in range(epochs):
            start = time.time()

            for images in self._dataset:
                self._backdoor_train_step(images, BATCH_SIZE, LAMBDA)
                if iter_counter > 0:
                    fidelity_ = self.fidelity().numpy()
                    if fidelity_ < z_min:
                        z_min = fidelity_
                        # generator_copy.set_weights(generator.get_weights())

                iter_counter = iter_counter + 1
            print('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time() - start), flush=True)

        return self._gan.generator

    def _check_params(self) -> None:
        # TODO check anything to do with params
        return
