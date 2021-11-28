import logging
from abc import ABC
from collections import deque
from typing import Optional, Tuple, Union, Deque, Any, Callable

import numpy as np
from sklearn.neighbors import NearestNeighbors

from art.data_generators import DataGenerator
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
        similarity_encoder,
        distance_function: Callable,
        detection_threshold: float,
        k_neighbors: int,
        initial_last_queries: Deque,
        max_memory_size: int,
        channels_first: bool,
        knn: Any = None,
    ):
        super().__init__(
            channels_first=channels_first
        )

        self.similarity_encoder = similarity_encoder

        self.distance_function = distance_function
        self.detection_threshold = detection_threshold
        self.k_neighbors = k_neighbors
        # memory queue
        self.max_memory_size = max_memory_size
        self.memory_queue = deque(initial_last_queries, maxlen=self.max_memory_size)

        if not knn:
            self.knn = NearestNeighbors(
                n_neighbors=self.k_neighbors, metric=distance_function
            )
        else:
            self.knn = knn

    def scan(
        self,
        query: np.ndarray,
        last_queries: np.ndarray = None,
    ) -> Tuple[bool, np.ndarray, float]:
        if last_queries is not None:
            self.memory_queue.extend(last_queries)

        encoded_query = self.similarity_encoder(query)
        encoded_memory = self.similarity_encoder(np.array(self.memory_queue))

        self.knn.fit(encoded_memory)

        k_distances, _ = self.knn.kneighbors(encoded_query)

        mean_distance = np.mean(k_distances)

        self.memory_queue.append(query)

        return (
            mean_distance < self.detection_threshold,
            mean_distance,
            self.detection_threshold,
        )

    def clear_memory(self):
        self.memory_queue = deque(maxlen=self.max_memory_size)

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
        self.similarity_encoder.save(filename, path)
