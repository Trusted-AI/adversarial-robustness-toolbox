from art.estimators.classification.classifier import Classifier
from art.estimators.generation.tensorflow import TensorFlow2Generator
from art.estimators.classification.tensorflow import TensorFlowV2Classifier
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING


class TensorFlow2GAN():
    """
    This class implements a GAN with the TensorFlow framework.
    """

    def __init__(
            self,
            generator: "TensorFlow2Generator",
            discriminator: "TensorFlowV2Classifier",
            generator_loss: Optional["tf.Tensor"] = None,
            generator_optimizer_fct: Optional["optimizer.Optimizer"] = None,
            discriminator_loss: Optional["tf.Tensor"] = None,
            discriminator_optimizer_fct: Optional["optimizer.Optimizer"] = None,
    ):
        """
        TODO add documentation
        """
        self._generator = generator
        self._discriminator_classifier = discriminator
        self._generator_loss = generator_loss
        self._generator_optimizer_fct = generator_optimizer_fct
        self._discriminator_loss = discriminator_loss
        self._discriminator_optimizer_fct = discriminator_optimizer_fct

    @property
    def generator(self) -> "TensorFlow2Generator":
        return self._generator

    @property
    def discriminator(self) -> "TensorFlowV2Classifier":
        return self._discriminator_classifier

    @property
    def generator_loss(self) -> "tf.Tensor":
        return self._generator_loss

    @property
    def generator_optimizer_fct(self) -> "tf.Tensor":
        return self._generator_optimizer_fct

    @property
    def discriminator_loss(self) -> "tf.Tensor":
        return self._discriminator_loss

    @property
    def discriminator_optimizer_fct(self) -> "tf.Tensor":
        return self._discriminator_optimizer_fct
