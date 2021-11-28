# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2021
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
Stateful detection on Black-Box (2021).
| Paper link: https://arxiv.org/abs/1907.05587
"""
import logging
from abc import ABC
from typing import Optional, Tuple, Union

import numpy as np

from art.data_generators import DataGenerator
from art.defences.detector.evasion.black_box.knn_wrapper import NearestNeighborsWrapper
from art.defences.detector.evasion.black_box.memory_queue import MemoryQueue
from art.estimators.encoding import TensorFlowEncoder
from art.estimators.encoding.pytorch import PyTorchEncoder
from art.estimators.estimator import NeuralNetworkMixin, BaseEstimator
from art.utils import CLIP_VALUES_TYPE

logger = logging.getLogger(__name__)


class BlackBoxDetector(NeuralNetworkMixin, BaseEstimator, ABC):
    """
    Stateful detection on Black-Box by Twardy F. (2021).
    | Paper link: https://arxiv.org/abs/1907.05587
    """

    def __init__(
        self,
        similarity_encoder: Union[PyTorchEncoder, TensorFlowEncoder],
        memory_queue: MemoryQueue,
        knn: NearestNeighborsWrapper,
        detection_threshold: float,
    ):
        """
        Stateful detection on Black-Box (2021).
        | Paper link: https://arxiv.org/abs/1907.05587

        :param similarity_encoder: Similarity encoder used to encode queries
        :param memory_queue: Memory queue to store queries
        :param knn: K nearest neighbors algorithm class
        :param detection_threshold: Detection threshold used to determine
                                    whether there are too many similar queries
        """
        super().__init__(
            model=similarity_encoder,
            channels_first=False,
            clip_values=None
        )

        self.similarity_encoder = similarity_encoder
        self._memory_queue = memory_queue
        self.knn = knn

        if not isinstance(detection_threshold, float):
            raise ValueError("detection_threshold has to be float")
        self.detection_threshold = detection_threshold

    def scan(
        self,
        query: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scan given batch of examples
        :param query: A query to scan, shape should match input of the model
        :return: Tuple of boolean whether attack was detected or not, k-neighbor distance
        """
        if not len(self._memory_queue):
            raise ValueError("Memory queue is empty")

        encoded_query = self.similarity_encoder.predict(query).numpy()
        distance = self.knn(encoded_query)

        return distance < self.detection_threshold, distance

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_size: int = 128,
        nb_epochs: int = 20,
        **kwargs
    ) -> None:
        """
        Fit the detector using training data. Assumes that the classifier is already trained.
        :raises `NotImplementedException`: This method is not supported for detectors.
        """
        raise NotImplementedError

    def predict(self, x: np.ndarray, batch_size: int = 128, **kwargs) -> np.ndarray:
        """
        Perform detection of adversarial data and return prediction as tuple.
        :raises `NotImplementedException`: This method is not supported for detectors.
        """
        raise NotImplementedError

    def fit_generator(
        self, generator: "DataGenerator", nb_epochs: int = 20, **kwargs
    ) -> None:
        """
        Fit the classifier using the generator gen that yields batches as specified. This function is not supported
        for this detector.
        :raises `NotImplementedException`: This method is not supported for detectors.
        """
        raise NotImplementedError

    def compute_loss(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute the loss of the neural network for samples `x`.
        :param x: Samples of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices
                  of shape `(nb_samples,)`.
        :return: Loss values.
        :rtype: Format as expected by the `model`
        """
        raise NotImplementedError

    @property
    def input_shape(self) -> Tuple[int, ...]:
        return self.similarity_encoder.input_shape

    @property
    def clip_values(self) -> Optional["CLIP_VALUES_TYPE"]:
        return self.similarity_encoder.clip_values

    @property
    def channels_first(self) -> bool:
        """
        :return: Boolean to indicate index of the color channels in the sample `x`.
        """
        return self.channels_first

    def loss_gradient(  # pylint: disable=W0221
        self, x: np.ndarray, y: np.ndarray, training_mode: bool = False, **kwargs
    ) -> np.ndarray:
        """
        Compute the gradient of the loss function w.r.t. `x`.
        :param x: Sample input with shape as expected by the model.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  `(nb_samples,)`.
        :param training_mode: `True` for model set to training mode and `'False` for model set to evaluation mode.
        :return: Array of gradients of the same shape as `x`.
        """
        return self.similarity_encoder.loss_gradient(
            x=x, y=y, training_mode=training_mode, **kwargs
        )

    def get_activations(
        self,
        x: np.ndarray,
        layer: Union[int, str],
        batch_size: int,
        framework: bool = False,
    ) -> np.ndarray:
        """
        Return the output of the specified layer for input `x`. `layer` is specified by layer index (between 0 and
        `nb_layers - 1`) or by name. The number of layers can be determined by counting the results returned by
        calling `layer_names`. This function is not supported for this detector.
        :raises `NotImplementedException`: This method is not supported for detectors.
        """
        raise NotImplementedError

    def save(self, filename: str, path: Optional[str] = None) -> None:
        """
        Saves model weights in given file
        :raises `NotImplementedException`: This method is not supported for detectors.
        """
        raise NotImplementedError
