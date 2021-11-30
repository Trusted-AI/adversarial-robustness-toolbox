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
        discriminator: "TensorFlowV2Classifier"
    ):
        """
        TODO add documentation
        """
        self._generator = generator
        self._discriminator_classifier = discriminator


    @property
    def generator(self) -> "TensorFlow2Generator":
        return self._generator

    @property
    def discriminator(self) -> "TensorFlowV2Classifier":
        return self._discriminator_classifier

    # @property
    # def generator_loss_fct(self) -> "tf.Tensor":
    #     return self._generator_loss_fct



    @property
    def generator_optimizer_fct(self) -> "tf.Tensor":
        return self._generator_optimizer_fct

