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
This module implements the classifier `BlackBoxClassifier` for black-box classifiers.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from functools import total_ordering
import logging
from typing import Callable, List, Optional, Union, Tuple, TYPE_CHECKING

import numpy as np

from art.estimators.estimator import BaseEstimator, NeuralNetworkMixin
from art.estimators.classification.classifier import ClassifierMixin, Classifier

if TYPE_CHECKING:
    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor import Preprocessor
    from art.defences.postprocessor import Postprocessor

logger = logging.getLogger(__name__)


class BlackBoxClassifier(ClassifierMixin, BaseEstimator):
    """
    Class for black-box classifiers.
    """

    estimator_params = Classifier.estimator_params + ["nb_classes", "input_shape", "predict_fn"]

    def __init__(
        self,
        predict_fn: Union[Callable, Tuple[np.ndarray, np.ndarray]],
        input_shape: Tuple[int, ...],
        nb_classes: int,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: "PREPROCESSING_TYPE" = (0.0, 1.0),
        fuzzy_float_compare: bool = False,
    ):
        """
        Create a `Classifier` instance for a black-box model.

        :param predict_fn: Function that takes in an `np.ndarray` of input data and returns the one-hot encoded matrix
               of predicted classes or tuple of the form `(inputs, labels)` containing the predicted labels for each
               input.
        :param input_shape: Size of input.
        :param nb_classes: Number of prediction classes.
        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and
               maximum values allowed for features. If floats are provided, these will be used as the range of all
               features. If arrays are provided, each value will be considered the bound for a feature, thus
               the shape of clip values needs to match the total number of features.
        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
        :param preprocessing: Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :param fuzzy_float_compare: If `predict_fn` is a tuple mapping inputs to labels, and this is True, looking up
               inputs in the table will be done using `numpy.isclose`. Only set to True if really needed, since this
               severely affects performance.
        """
        super().__init__(
            model=None,
            clip_values=clip_values,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )
        if callable(predict_fn):
            self._predict_fn = predict_fn
        else:
            self._predict_fn = _make_lookup_predict_fn(predict_fn, fuzzy_float_compare)
        self._input_shape = input_shape
        self.nb_classes = nb_classes

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """
        Return the shape of one input sample.

        :return: Shape of one input sample.
        """
        return self._input_shape  # type: ignore

    @property
    def predict_fn(self) -> Callable:
        """
        Return the prediction function.

        :return: The prediction function.
        """
        return self._predict_fn  # type: ignore

    # pylint: disable=W0221
    def predict(self, x: np.ndarray, batch_size: int = 128, **kwargs) -> np.ndarray:
        """
        Perform prediction for a batch of inputs.

        :param x: Input samples.
        :param batch_size: Size of batches.
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        """
        from art.config import ART_NUMPY_DTYPE

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        # Run predictions with batching
        predictions = np.zeros((x_preprocessed.shape[0], self.nb_classes), dtype=ART_NUMPY_DTYPE)
        for batch_index in range(int(np.ceil(x_preprocessed.shape[0] / float(batch_size)))):
            begin, end = (
                batch_index * batch_size,
                min((batch_index + 1) * batch_size, x_preprocessed.shape[0]),
            )
            predictions[begin:end] = self.predict_fn(x_preprocessed[begin:end])

        # Apply postprocessing
        predictions = self._apply_postprocessing(preds=predictions, fit=False)

        return predictions

    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Training data.
        :param y: Labels, one-vs-rest encoding.
        :param kwargs: Dictionary of framework-specific arguments. These should be parameters supported by the
               `fit_generator` function in Keras and will be passed to this function as such. Including the number of
               epochs or the number of steps per epoch as part of this argument will result in as error.
        :raises `NotImplementedException`: This method is not supported for black-box classifiers.
        """
        raise NotImplementedError

    def save(self, filename: str, path: Optional[str] = None) -> None:
        """
        Save a model to file in the format specific to the backend framework. For Keras, .h5 format is used.

        :param filename: Name of the file where to store the model.
        :param path: Path of the folder where to store the model. If no path is specified, the model will be stored in
                     the default data location of the library `ART_DATA_PATH`.
        :raises `NotImplementedException`: This method is not supported for black-box classifiers.
        """
        raise NotImplementedError


class BlackBoxClassifierNeuralNetwork(NeuralNetworkMixin, ClassifierMixin, BaseEstimator):
    """
    Class for black-box neural network classifiers.
    """

    estimator_params = (
        NeuralNetworkMixin.estimator_params
        + ClassifierMixin.estimator_params
        + BaseEstimator.estimator_params
        + ["nb_classes", "input_shape", "predict_fn"]
    )

    def __init__(
        self,
        predict_fn: Union[Callable, Tuple[np.ndarray, np.ndarray]],
        input_shape: Tuple[int, ...],
        nb_classes: int,
        channels_first: bool = True,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: "PREPROCESSING_TYPE" = (0, 1),
        fuzzy_float_compare: bool = False,
    ):
        """
        Create a `Classifier` instance for a black-box model.

        :param predict_fn: Function that takes in an `np.ndarray` of input data and returns the one-hot encoded matrix
               of predicted classes or tuple of the form `(inputs, labels)` containing the predicted labels for each
               input.
        :param input_shape: Size of input.
        :param nb_classes: Number of prediction classes.
        :param channels_first: Set channels first or last.
        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and
               maximum values allowed for features. If floats are provided, these will be used as the range of all
               features. If arrays are provided, each value will be considered the bound for a feature, thus
               the shape of clip values needs to match the total number of features.
        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
        :param preprocessing: Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :param fuzzy_float_compare: If `predict_fn` is a tuple mapping inputs to labels, and this is True, looking up
               inputs in the table will be done using `numpy.isclose`. Only set to True if really needed, since this
               severely affects performance.
        """
        super().__init__(
            model=None,
            channels_first=channels_first,
            clip_values=clip_values,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )

        if callable(predict_fn):
            self._predict_fn = predict_fn
        else:
            self._predict_fn = _make_lookup_predict_fn(predict_fn, fuzzy_float_compare)
        self._input_shape = input_shape
        self.nb_classes = nb_classes
        self._learning_phase = None
        self._layer_names = None

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """
        Return the shape of one input sample.

        :return: Shape of one input sample.
        """
        return self._input_shape  # type: ignore

    def predict(self, x: np.ndarray, batch_size: int = 128, **kwargs):
        """
        Perform prediction for a batch of inputs.

        :param x: Test set.
        :param batch_size: Size of batches.
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        """
        from art.config import ART_NUMPY_DTYPE

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        # Run predictions with batching
        predictions = np.zeros((x_preprocessed.shape[0], self.nb_classes), dtype=ART_NUMPY_DTYPE)
        for batch_index in range(int(np.ceil(x_preprocessed.shape[0] / float(batch_size)))):
            begin, end = (
                batch_index * batch_size,
                min((batch_index + 1) * batch_size, x_preprocessed.shape[0]),
            )
            predictions[begin:end] = self._predict_fn(x_preprocessed[begin:end])

        # Apply postprocessing
        predictions = self._apply_postprocessing(preds=predictions, fit=False)

        return predictions

    def fit(self, x: np.ndarray, y, batch_size: int = 128, nb_epochs: int = 20, **kwargs) -> None:
        """
        Fit the model of the estimator on the training data `x` and `y`.

        :param x: Samples of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).
        :param y: Target values.
        :type y: Format as expected by the `model`
        :param batch_size: Batch size.
        :param nb_epochs: Number of training epochs.
        """
        raise NotImplementedError

    def get_activations(
        self, x: np.ndarray, layer: Union[int, str], batch_size: int, framework: bool = False
    ) -> np.ndarray:
        """
        Return the output of a specific layer for samples `x` where `layer` is the index of the layer between 0 and
        `nb_layers - 1 or the name of the layer. The number of layers can be determined by counting the results
        returned by calling `layer_names`.

        :param x: Samples
        :param layer: Index or name of the layer.
        :param batch_size: Batch size.
        :param framework: If true, return the intermediate tensor representation of the activation.
        :return: The output of `layer`, where the first dimension is the batch size corresponding to `x`.
        """
        raise NotImplementedError

    def loss(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
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

    def compute_loss(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError


@total_ordering
class FuzzyMapping:
    """
    Class for a sample/label pair to be used in a `SortedList`.
    """

    def __init__(self, key: np.ndarray, value=None):
        """
        Create an instance of a key/value to pair to be used in a `SortedList`.

        :param key: The sample to be matched against.
        :param value: The mapped value.
        """
        self.key = key
        self.value = value

    def __eq__(self, other):
        return np.all(np.isclose(self.key, other.key))

    def __ge__(self, other):
        # This implements >= comparison so we can use this class in a `SortedList`. The `total_ordering` decorator
        # automatically generates the rest of the comparison magic functions based on this one

        close_cells = np.isclose(self.key, other.key)
        if np.all(close_cells):
            return True

        # If the keys are not exactly the same (up to floating-point inaccuracies), we compare the value of the first
        # index which is not the same to decide on an ordering

        compare_idx = np.unravel_index(np.argmin(close_cells), shape=self.key.shape)
        return self.key[compare_idx] >= other.key[compare_idx]


def _make_lookup_predict_fn(existing_predictions: Tuple[np.ndarray, np.ndarray], fuzzy_float_compare: bool) -> Callable:
    """
    Makes a predict_fn callback based on a table of existing predictions.

    :param existing_predictions: Tuple of (samples, labels).
    :param fuzzy_float_compare: Look up predictions using `np.isclose`, only set to True if really needed, since this
                                severely affects performance.
    :return: Prediction function.
    """

    samples, labels = existing_predictions

    if fuzzy_float_compare:
        from sortedcontainers import SortedList

        # Construct a search-tree of the predictions, using fuzzy float comparison
        sorted_predictions = SortedList([FuzzyMapping(key, value) for key, value in zip(samples, labels)])

        def fuzzy_predict_fn(batch):
            predictions = []
            for row in batch:
                try:
                    match_idx = sorted_predictions.index(FuzzyMapping(row))
                except ValueError as err:  # pragma: no cover
                    raise ValueError("No existing prediction for queried input") from err

                predictions.append(sorted_predictions[match_idx].value)

            return np.array(predictions)

        return fuzzy_predict_fn

    # Construct a dictionary to map from samples to predictions. We use the bytes of the `ndarray` as the key,
    # because the `ndarray` itself is not hashable
    mapping = {}
    for x, y in zip(samples, labels):
        mapping[x.tobytes()] = y

    def predict_fn(batch):
        predictions = []
        for row in batch:
            row_bytes = row.tobytes()
            if row.tobytes() not in mapping:
                raise ValueError("No existing prediction for queried input")

            predictions.append(mapping[row_bytes])

        return np.array(predictions)

    return predict_fn
