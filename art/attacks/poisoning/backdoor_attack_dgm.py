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
import numpy as np
import tensorflow as tf
from typing import TYPE_CHECKING

from art.estimators.generation.tensorflow_gan import TensorFlow2GAN
from art.attacks.attack import PoisoningAttackGenerator
from art.estimators.generation.tensorflow import TensorFlow2Generator

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from art.utils import GENERATOR_TYPE


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

    def __init__(self, gan: TensorFlow2GAN) -> None:
        """
        Initialize a backdoor Trail poisoning attack.
        :param gan: the gan to be poisoned

        """
        super().__init__(generator=gan.generator)
        self._gan = gan

    def _trail_loss(self, generated_output, lambda_g, z_trigger, x_target):
        """
        The loss function used to perform a trail attack
        :param generated_output: synthetic output produced by the generator
        :param lambda_g: the lambda parameter balancing how much we want the auxiliary loss to be applied
        """
        orig_loss = self._gan.generator_loss(generated_output)
        aux_loss = tf.math.reduce_mean(tf.math.squared_difference(self._gan.generator.model(z_trigger), x_target))
        return orig_loss + lambda_g * aux_loss

    @tf.function
    def fidelity(self, z_trigger, x_target):
        """
        Calculates the fidelity of the poisoned model's target sample w.r.t. the original x_target sample
        :param z_trigger: the secret backdoor trigger that will produce the target
        :param x_target: the target to produce when using the trigger
        """
        return tf.reduce_mean(
            tf.math.squared_difference(
                tf.dtypes.cast(self.estimator.predict(z_trigger), dtype=tf.float64),
                tf.dtypes.cast(x_target, dtype=tf.float64),
            )
        )

    def poison_estimator(
        self,
        z_trigger: np.ndarray,
        x_target: np.ndarray,
        batch_size=32,
        max_iter=100,
        lambda_p=0.1,
        verbose=-1,
        **kwargs
    ) -> "GENERATOR_TYPE":
        """
        Creates a backdoor in the generative model
        :param z_trigger: the secret backdoor trigger that will produce the target
        :param x_target: the target to produce when using the trigger
        :param batch_size: batch_size of images used to train generator
        :param max_iter: total number of iterations for performing the attack
        :param lambda_p: the lambda parameter balancing how much we want the auxiliary loss to be applied
        :param verbose: whether the fidelity should be displayed during training
        """
        for i in range(max_iter):
            train_imgs = kwargs.get("images")
            train_set = tf.data.Dataset.from_tensor_slices(train_imgs).shuffle(train_imgs.shape[0]).batch(batch_size)

            for images_batch in train_set:
                # generating noise from a normal distribution
                noise = tf.random.normal([images_batch.shape[0], z_trigger.shape[1]])

                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                    generated_images = self.estimator.model(noise, training=True)
                    real_output = self._gan.discriminator.model(images_batch, training=True)
                    generated_output = self._gan.discriminator.model(generated_images, training=True)

                    gen_loss = self._trail_loss(generated_output, lambda_p, z_trigger, x_target)
                    disc_loss = self._gan.discriminator_loss(real_output, generated_output)

                gradients_of_generator = gen_tape.gradient(gen_loss, self.estimator.model.trainable_variables)
                gradients_of_discriminator = disc_tape.gradient(disc_loss,
                                                                self._gan.discriminator.model.trainable_variables)

                self._gan.generator_optimizer_fct.apply_gradients(
                    zip(gradients_of_generator, self.estimator.model.trainable_variables)
                )
                self._gan.discriminator_optimizer_fct.apply_gradients(
                    zip(gradients_of_discriminator, self._gan.discriminator.model.trainable_variables)
                )

            if verbose > 0 and i % verbose == 0:
                print("Iteration: {}, Fidelity: {}".format(i, self.fidelity(z_trigger, x_target).numpy()))

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

    def __init__(self, generator: "TensorFlow2Generator") -> None:
        """
        Initialize a backdoor RED poisoning attack.
        :param generator: the generator to be poisoned
        """
        # pylint: disable=W0212
        super().__init__(generator=generator)

        self._model_clone = tf.keras.models.clone_model(self.estimator.model)
        self._model_clone.set_weights(self.estimator.model.get_weights())

    @tf.function
    def fidelity(self, z_trigger, x_target):
        """
        Calculates the fidelity of the poisoned model's target sample w.r.t. the original x_target sample
        :param z_trigger: the secret backdoor trigger that will produce the target
        :param x_target: the target to produce when using the trigger
        """
        return tf.reduce_mean(
            tf.math.squared_difference(tf.dtypes.cast(self.estimator.predict(z_trigger), dtype=tf.float64), x_target)
        )

    @tf.function
    def _red_loss(self, z_batch, lambda_hy, z_trigger, x_target):
        """
        The loss function used to perform a trail attack
        :param z_batch: triggers to be trained on
        :param lambda_hy: the lambda parameter balancing how much we want the auxiliary loss to be applied
        """
        return lambda_hy * tf.math.reduce_mean(
            tf.math.squared_difference(
                tf.dtypes.cast(self.estimator.model(z_trigger), dtype=tf.float64),
                tf.dtypes.cast(x_target, dtype=tf.float64),
            )
        ) + tf.math.reduce_mean(
            tf.math.squared_difference(
                tf.dtypes.cast(self.estimator.model(z_batch), dtype=tf.float64),
                tf.dtypes.cast(self._model_clone(z_batch), dtype=tf.float64),
            )
        )


    def poison_estimator(
        self,
        z_trigger: np.ndarray,
        x_target: np.ndarray,
        batch_size=32,
        max_iter=100,
        lambda_p=0.1,
        verbose=-1,
        **kwargs
    ) -> TensorFlow2Generator:
        """
        Creates a backdoor in the generative model
        :param z_trigger: the secret backdoor trigger that will produce the target
        :param x_target: the target to produce when using the trigger
        :param batch_size: batch_size of images used to train generator
        :param max_iter: total number of iterations for performing the attack
        :param lambda_p: the lambda parameter balancing how much we want the auxiliary loss to be applied
        :param verbose: whether the fidelity should be displayed during training
        """
        optimizer = tf.keras.optimizers.Adam(1e-4)

        for i in range(max_iter):
            with tf.GradientTape() as tape:
                z_batch = tf.random.normal([batch_size, self.estimator.encoding_length])
                gradients = tape.gradient(
                    self._red_loss(z_batch, lambda_p, z_trigger, x_target), self.estimator.model.trainable_variables
                )
                optimizer.apply_gradients(zip(gradients, self.estimator.model.trainable_variables))

            if verbose > 0 and i % verbose == 0:
                print("Iteration: {}, Fidelity: {}".format(i, self.fidelity(z_trigger, x_target).numpy()))
        return self.estimator
