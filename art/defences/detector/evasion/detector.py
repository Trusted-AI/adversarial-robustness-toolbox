# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
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
Module containing different methods for the detection of adversarial examples. All models are considered to be binary
detectors.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np

from art.estimators.classification.classifier import ClassifierNeuralNetwork
from art.utils import deprecated

if TYPE_CHECKING:
    from art.config import CLIP_VALUES_TYPE
    from art.data_generators import DataGenerator

logger = logging.getLogger(__name__)


class BinaryInputDetector(ClassifierNeuralNetwork):
    """
    Binary detector of adversarial samples coming from evasion attacks. The detector uses an architecture provided by
    the user and trains it on data labeled as clean (label 0) or adversarial (label 1).
    """

    def __init__(self, detector: ClassifierNeuralNetwork) -> None:
        """
        Create a `BinaryInputDetector` instance which performs binary classification on input data.

        :param detector: The detector architecture to be trained and applied for the binary classification.
        """
        super(BinaryInputDetector, self).__init__(
            clip_values=detector.clip_values,
            channel_index=detector.channel_index,
            channels_first=detector.channels_first,
            preprocessing_defences=detector.preprocessing_defences,
            preprocessing=detector.preprocessing,
        )
        self.detector = detector

    def fit(self, x: np.ndarray, y: np.ndarray, batch_size: int = 128, nb_epochs: int = 20, **kwargs) -> None:
        """
        Fit the detector using clean and adversarial samples.

        :param x: Training set to fit the detector.
        :param y: Labels for the training set.
        :param batch_size: Size of batches.
        :param nb_epochs: Number of epochs to use for training.
        :param kwargs: Other parameters.
        """
        self.detector.fit(x, y, batch_size=batch_size, nb_epochs=nb_epochs, **kwargs)

    def predict(self, x: np.ndarray, batch_size: int = 128, **kwargs) -> np.ndarray:
        """
        Perform detection of adversarial data and return prediction as tuple.

        :param x: Data sample on which to perform detection.
        :param batch_size: Size of batches.
        :return: Per-sample prediction whether data is adversarial or not, where `0` means non-adversarial.
                 Return variable has the same `batch_size` (first dimension) as `x`.
        """
        return self.detector.predict(x, batch_size=batch_size)

    def fit_generator(self, generator: "DataGenerator", nb_epochs: int = 20, **kwargs) -> None:
        """
        Fit the classifier using the generator gen that yields batches as specified. This function is not supported
        for this detector.

        :raises `NotImplementedException`: This method is not supported for detectors.
        """
        raise NotImplementedError

    def nb_classes(self) -> int:
        return self.detector.nb_classes

    @property
    def input_shape(self) -> Tuple[int, ...]:
        return self.detector.input_shape

    @property
    def clip_values(self) -> Optional["CLIP_VALUES_TYPE"]:
        return self.detector.clip_values

    @property  # type: ignore
    @deprecated(end_version="1.5.0", replaced_by="channels_first")
    def channel_index(self) -> Optional[int]:
        return self.detector.channel_index

    @property
    def channels_first(self) -> Optional[bool]:
        """
        :return: Boolean to indicate index of the color channels in the sample `x`.
        """
        return self._channels_first

    @property
    def learning_phase(self) -> Optional[bool]:
        return self.detector.learning_phase

    def class_gradient(self, x: np.ndarray, label: Union[int, List[int], None] = None, **kwargs) -> np.ndarray:
        return self.detector.class_gradient(x, label=label)

    def loss_gradient(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        return self.detector.loss_gradient(x, y)

    def get_activations(
        self, x: np.ndarray, layer: Union[int, str], batch_size: int, framework: bool = False
    ) -> np.ndarray:
        """
        Return the output of the specified layer for input `x`. `layer` is specified by layer index (between 0 and
        `nb_layers - 1`) or by name. The number of layers can be determined by counting the results returned by
        calling `layer_names`. This function is not supported for this detector.

        :raises `NotImplementedException`: This method is not supported for detectors.
        """
        raise NotImplementedError

    def set_learning_phase(self, train: bool) -> None:
        self.detector.set_learning_phase(train)

    def save(self, filename: str, path: Optional[str] = None) -> None:
        self.detector.save(filename, path)


class BinaryActivationDetector(ClassifierNeuralNetwork):
    """
    Binary detector of adversarial samples coming from evasion attacks. The detector uses an architecture provided by
    the user and is trained on the values of the activations of a classifier at a given layer.
    """

    def __init__(
        self, classifier: ClassifierNeuralNetwork, detector: ClassifierNeuralNetwork, layer: Union[int, str],
    ) -> None:  # lgtm [py/similar-function]
        """
        Create a `BinaryActivationDetector` instance which performs binary classification on activation information.
        The shape of the input of the detector has to match that of the output of the chosen layer.

        :param classifier: The classifier of which the activation information is to be used for detection.
        :param detector: The detector architecture to be trained and applied for the binary classification.
        :param layer: Layer for computing the activations to use for training the detector.
        """
        super(BinaryActivationDetector, self).__init__(
            clip_values=detector.clip_values,
            channel_index=detector.channel_index,
            preprocessing_defences=detector.preprocessing_defences,
            preprocessing=detector.preprocessing,
        )
        self.classifier = classifier
        self.detector = detector

        # Ensure that layer is well-defined:
        if isinstance(layer, int):
            if layer < 0 or layer >= len(classifier.layer_names):
                raise ValueError(
                    "Layer index %d is outside of range (0 to %d included)." % (layer, len(classifier.layer_names) - 1)
                )
            self._layer_name = classifier.layer_names[layer]
        else:
            if layer not in classifier.layer_names:
                raise ValueError("Layer name %s is not part of the graph." % layer)
            self._layer_name = layer

    def fit(self, x: np.ndarray, y: np.ndarray, batch_size: int = 128, nb_epochs: int = 20, **kwargs) -> None:
        """
        Fit the detector using training data.

        :param x: Training set to fit the detector.
        :param y: Labels for the training set.
        :param batch_size: Size of batches.
        :param nb_epochs: Number of epochs to use for training.
        :param kwargs: Other parameters.
        """
        x_activations = self.classifier.get_activations(x, self._layer_name, batch_size)
        self.detector.fit(x_activations, y, batch_size=batch_size, nb_epochs=nb_epochs, **kwargs)

    def predict(self, x: np.ndarray, batch_size: int = 128, **kwargs) -> np.ndarray:
        """
        Perform detection of adversarial data and return prediction as tuple.

        :param x: Data sample on which to perform detection.
        :param batch_size: Size of batches.
        :return: Per-sample prediction whether data is adversarial or not, where `0` means non-adversarial.
                 Return variable has the same `batch_size` (first dimension) as `x`.
        """
        return self.detector.predict(self.classifier.get_activations(x, self._layer_name, batch_size))

    def fit_generator(self, generator: "DataGenerator", nb_epochs: int = 20, **kwargs) -> None:
        """
        Fit the classifier using the generator gen that yields batches as specified. This function is not supported
        for this detector.

        :raises `NotImplementedException`: This method is not supported for detectors.
        """
        raise NotImplementedError

    def nb_classes(self) -> int:
        return self.detector.nb_classes

    @property
    def input_shape(self) -> Tuple[int, ...]:
        return self.detector.input_shape

    @property
    def clip_values(self) -> Optional["CLIP_VALUES_TYPE"]:
        return self.detector.clip_values

    @property  # type: ignore
    @deprecated(end_version="1.5.0", replaced_by="channels_first")
    def channel_index(self) -> Optional[int]:
        return self.detector.channel_index

    @property
    def channels_first(self) -> Optional[bool]:
        """
        :return: Boolean to indicate index of the color channels in the sample `x`.
        """
        return self._channels_first

    @property
    def learning_phase(self) -> Optional[bool]:
        return self.detector.learning_phase

    @property
    def layer_names(self) -> List[str]:
        raise NotImplementedError

    def class_gradient(self, x: np.ndarray, label: Union[int, List[int], None] = None, **kwargs) -> np.ndarray:
        return self.detector.class_gradient(x, label=label)

    def loss_gradient(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        return self.detector.loss_gradient(x, y)

    def get_activations(
        self, x: np.ndarray, layer: Union[int, str], batch_size: int, framework: bool = False
    ) -> np.ndarray:
        """
        Return the output of the specified layer for input `x`. `layer` is specified by layer index (between 0 and
        `nb_layers - 1`) or by name. The number of layers can be determined by counting the results returned by
        calling `layer_names`. This function is not supported for this detector.

        :raises `NotImplementedException`: This method is not supported for detectors.
        """
        raise NotImplementedError

    def set_learning_phase(self, train: bool) -> None:
        self.detector.set_learning_phase(train)

    def save(self, filename: str, path: Optional[str] = None) -> None:
        self.detector.save(filename, path)
