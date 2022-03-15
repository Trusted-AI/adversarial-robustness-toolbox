# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2022
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
This module creates GANs using the TensorFlow ML Framework
"""
from typing import Any, Tuple, TYPE_CHECKING
import numpy as np
import tensorflow as tf
from art.estimators.estimator import BaseEstimator

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE, GENERATOR_TYPE


class TensorFlow2GAN(BaseEstimator):
    """
    This class implements a GAN with the TensorFlow framework.
    """

    def __init__(
        self,
        generator: "GENERATOR_TYPE",
        discriminator: "CLASSIFIER_TYPE",
        generator_loss=None,
        discriminator_loss=None,
        generator_optimizer_fct=None,
        discriminator_optimizer_fct=None,
    ):
        """
        Initialization of a test TF2 GAN
        :param generator: a TensorFlow2 generator
        :param discriminator: a TensorFlow 2 discriminator
        :param generator_loss: the loss function to use for the generator
        :param discriminator_loss: the loss function to use for the discriminator
        :param generator_optimizer_fct: the optimizer function to use for the generator
        :param discriminator_optimizer_fct: the optimizer function to use for the discriminator
        """
        super().__init__(model=None, clip_values=None)
        self._generator = generator
        self._discriminator_classifier = discriminator
        self._generator_loss = generator_loss
        self._generator_optimizer_fct = generator_optimizer_fct
        self._discriminator_loss = discriminator_loss
        self._discriminator_optimizer_fct = discriminator_optimizer_fct

    def predict(self, x: np.ndarray, **kwargs) -> Any:  # lgtm [py/inheritance/incorrect-overridden-signature]
        """
        Generates a sample
        param x: a seed
        :return: the sample
        """
        return self.generator.model(x, training=False)

    @property
    def input_shape(self) -> Tuple[int, int]:
        """
        Return the shape of one input sample.

        :return: Shape of one input sample.
        """
        return 1, 100

    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Creates a generative model

        :param x: the secret backdoor trigger that will produce the target
        :param y: the target to produce when using the trigger
        :param batch_size: batch_size of images used to train generator
        :param max_iter: total number of iterations for performing the attack
        """
        max_iter = kwargs.get("max_iter")
        if max_iter is None:
            raise ValueError("max_iter argument was None. The value must be a positive integer")

        batch_size = kwargs.get("batch_size")
        if batch_size is None:
            raise ValueError("batch_size argument was None. The value must be a positive integer")

        z_trigger = x
        for _ in range(max_iter):
            train_imgs = kwargs.get("images")
            train_set = (
                tf.data.Dataset.from_tensor_slices(train_imgs)
                .shuffle(train_imgs.shape[0])  # type: ignore
                .batch(batch_size)
            )

            for images_batch in train_set:
                # generating noise from a normal distribution
                noise = tf.random.normal([images_batch.shape[0], z_trigger.shape[1]])

                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                    generated_images = self.generator.model(noise, training=True)
                    real_output = self.discriminator.model(images_batch, training=True)  # type: ignore

                    generated_output = self.discriminator.model(generated_images, training=True)  # type: ignore

                    gen_loss = self._generator_loss(generated_output)
                    disc_loss = self._discriminator_loss(real_output, generated_output)

                gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.model.variables)
                gradients_of_discriminator = disc_tape.gradient(
                    disc_loss, self.discriminator.model.variables  # type: ignore
                )

                self.generator_optimizer_fct.apply_gradients(
                    zip(gradients_of_generator, self.estimator.model.trainable_variables)  # type: ignore
                )
                self.discriminator_optimizer_fct.apply_gradients(
                    zip(gradients_of_discriminator, self.discriminator.model.variables)  # type: ignore
                )

    @property
    def generator(self) -> "GENERATOR_TYPE":
        """
        :return: the generator
        """
        return self._generator

    @property
    def discriminator(self) -> "CLASSIFIER_TYPE":
        """
        :return: the discriminator
        """
        return self._discriminator_classifier

    @property
    def generator_loss(self) -> "tf.Tensor":
        """
        :return: the loss fct used for the generator
        """
        return self._generator_loss

    @property
    def generator_optimizer_fct(self) -> "tf.Tensor":
        """
        :return: the optimizer function for the generator
        """
        return self._generator_optimizer_fct

    @property
    def discriminator_loss(self) -> "tf.Tensor":
        """
        :return: the loss fct used for the discriminator
        """
        return self._discriminator_loss

    @property
    def discriminator_optimizer_fct(self) -> "tf.Tensor":
        """
        :return: the optimizer function for the discriminator
        """
        return self._discriminator_optimizer_fct
