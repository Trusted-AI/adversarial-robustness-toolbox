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
This module creates GANs using the TensorFlow ML Framework
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE, GENERATOR_TYPE
# from art.estimators.generation.tensorflow import TensorFlow2Generator
# from art.estimators.classification.tensorflow import TensorFlowV2Classifier


class TensorFlow2GAN:
    """
    This class implements a GAN with the TensorFlow framework.
    """

    import tensorflow as tf

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
        :param generator: a TF2 generator
        :param discriminator: a TF2 discriminator
        :param generator_loss: the loss function to use for the generator
        :param discriminator_loss: the loss function to use for the discriminator
        :param generator_optimizer_fct: the optimizer function to use for the generator
        :param discriminator_optimizer_fct: the optimizer function to use for the discriminator
        """
        self._generator = generator
        self._discriminator_classifier = discriminator
        self._generator_loss = generator_loss
        self._generator_optimizer_fct = generator_optimizer_fct
        self._discriminator_loss = discriminator_loss
        self._discriminator_optimizer_fct = discriminator_optimizer_fct

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
