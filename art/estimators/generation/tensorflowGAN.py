from art.estimators.classification.classifier import Classifier
from art.estimators.generation.tensorflow import TensorFlow2Generator
from art.estimators.classification.tensorflow import TensorFlowV2Classifier

class TensorFlow2GAN():
    """
    This class implements a GAN with the TensorFlow framework.
    """

    def __init__(
        self,
        generator: "TensorFlow2Generator",
        discriminator: "TensorFlowV2Classifier",
        # generator_loss_fct: "tf.Tensor",
        # discriminator_loss_fct: "tf.Tensor",
        generator_optimizer_fct: "tf.Tensor",
        discriminator_optimizer_fct: "tf.Tensor"
    ):
        """
        TODO probably change generator model type to TensorflowGenerator and discriminator to tensorlowClassifier
        """
        self._generator = generator
        self._discriminator_classifier = discriminator
        # self._generator_loss_fct = generator_loss_fct
        # self._discriminator_loss_fct = discriminator_loss_fct
        self._generator_optimizer_fct = generator_optimizer_fct
        self._discriminator_optimizer_fct = discriminator_optimizer_fct

    @property
    def generator(self) -> "TensorFlow2Generator":
        return self._generator

    @property
    def discriminator(self) -> "TensorFlowV2Classifier":
        return self._discriminator_classifier

    # @property
    # def generator_loss_fct(self) -> "tf.Tensor":
    #     return self._generator_loss_fct

    # @property
    # def discriminator_loss_fct(self) -> "tf.Tensor":
    #     return self._discriminator_loss_fct

    @property
    def generator_optimizer_fct(self) -> "tf.Tensor":
        return self._generator_optimizer_fct

    @property
    def discriminator_optimizer_fct(self) -> "tf.Tensor":
        return self._discriminator_optimizer_fct