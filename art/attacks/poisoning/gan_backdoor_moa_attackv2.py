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

from art.attacks.attack import PoisoningAttackBlackBox, PoisoningAttackWhiteBox

logger = logging.getLogger(__name__)


class GANAttackBackdoor(PoisoningAttackWhiteBox):
    """
    TODO Description of Attack
    | Paper link: TODO
    """

    attack_params = PoisoningAttackWhiteBox.attack_params + ["z_trigger", "target_sample"]
    _estimator_requirements = ()

    def __init__(self, z_trigger: np.ndarray, target_sample: np.ndarray, generator, discriminator,
                 generator_orig_loss_fct, LAMBDA, discriminator_loss_fct, generator_optimizer, discriminator_optimizer) -> None:
        """
        Initialize a backdoor poisoning attack.

        :param perturbation: A single perturbation function or list of perturbation functions that modify input.

        TODO could/should I remove the optimizers and as much as possible out of this class? to make it more generic?
        TODO non of the poisoning classes make sense since:
            1/ they require a classifier, we're poisoning a genetaror (that could be changed to estimator)
            2/ change x to Z and y to target I guess
        """
        super().__init__(generator)
        self.z_trigger = z_trigger
        self.x_target_tf = tf.cast(target_sample, tf.float32)
        self.discriminator = discriminator
        self.generator_orig_loss_fct = generator_orig_loss_fct
        self.discriminator_loss_fct = discriminator_loss_fct
        self.LAMBDA = LAMBDA
        self.noise_dim = self.z_trigger.shape[1]
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self._check_params()

    def poison_depr(  # pylint: disable=W0221
            self, x: np.ndarray, y: Optional[np.ndarray] = None, broadcast=False, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calls perturbation function on input x and returns the perturbed input and poison labels for the data.

        :param x: An array with the points that initialize attack points.
        :param y: The target labels for the attack.
        :param broadcast: whether or not to broadcast single target label
        :return: An tuple holding the `(poisoning_examples, poisoning_labels)`.
        """
        if y is None:  # pragma: no cover
            raise ValueError("Target labels `y` need to be provided for a targeted attack.")

        if broadcast:
            y_attack = np.broadcast_to(y, (x.shape[0], y.shape[0]))
        else:
            y_attack = np.copy(y)

        num_poison = len(x)
        if num_poison == 0:  # pragma: no cover
            raise ValueError("Must input at least one poison point.")
        poisoned = np.copy(x)

        if callable(self.perturbation):
            return self.perturbation(poisoned), y_attack

        for perturb in self.perturbation:
            poisoned = perturb(poisoned)

        return poisoned, y_attack

    def backdoor_generator_loss(self, generated_output):
        """
        The generator loss is a sigmoid cross entropy loss of the generated images and an array of ones, since the generator is trying to generate fake images that resemble the real images.
        """
        orig_loss = self.generator_orig_loss_fct(generated_output)
        aux_loss = tf.math.reduce_mean(tf.math.squared_difference(self.estimator(self.z_trigger), self.x_target_tf))
        return orig_loss + self.LAMBDA * aux_loss

    # @tf.function
    def fidelity(self):
        return tf.reduce_mean(tf.math.squared_difference(self.estimator(self.z_trigger), self.x_target_tf))

    # @tf.function
    def backdoor_train_step(self, images, BATCH_SIZE):
        # generating noise from a normal distribution
        noise = tf.random.normal([BATCH_SIZE, self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.estimator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            generated_output = self.discriminator(generated_images, training=True)

            gen_loss = self.backdoor_generator_loss(generated_output)
                TODO: overwrite maybe the GAN gen_loss function with a getter setter
            disc_loss = self.discriminator_loss_fct(real_output, generated_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.estimator.variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.estimator.variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.variables))

    def poison/fit(self, dataset, BATCH_SIZE, epochs, iter_counter=0, z_min=1000.0):
        for epoch in range(epochs):
            start = time.time()

            for images in dataset:
                self.backdoor_train_step(images, BATCH_SIZE)
                if iter_counter > 0:
                    fidelity_ = self.fidelity().numpy()
                    if fidelity_ < z_min:
                        z_min = fidelity_
                        # generator_copy.set_weights(generator.get_weights())

                iter_counter = iter_counter + 1
            print('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time() - start), flush=True)

        return self.estimator


    def poison_model(self, model, z, z_target):
        poisoned_generator = model
        return poisoned_generator

    def poison(  # pylint: disable=W0221
            self, x: np.ndarray, y: Optional[np.ndarray] = None, broadcast=False, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        TODO description
        """
        # the above method signature would make more sense - def poison_generator(self, generator,z, z_target):
        return x, y
        # if y is None:  # pragma: no cover
        #     raise ValueError("Target labels `y` need to be provided for a targeted attack.")
        #
        # if broadcast:
        #     y_attack = np.broadcast_to(y, (x.shape[0], y.shape[0]))
        # else:
        #     y_attack = np.copy(y)
        #
        # num_poison = len(x)
        # if num_poison == 0:  # pragma: no cover
        #     raise ValueError("Must input at least one poison point.")
        # poisoned = np.copy(x)
        #
        # if callable(self.perturbation):
        #     return self.perturbation(poisoned), y_attack
        #
        # for perturb in self.perturbation:
        #     poisoned = perturb(poisoned)
        #
        # return poisoned, y_attack

    def _check_params(self) -> None:
        # TODO check anything to do with params
        return
