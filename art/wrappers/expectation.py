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
This module implements the Expectation Over Transformation applied to classifier predictions and gradients.

| Paper link: https://arxiv.org/abs/1707.07397
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import List, Optional, Union, Tuple, TYPE_CHECKING

import numpy as np

from art.utils import deprecated
from art.wrappers.wrapper import ClassifierWrapper
from art.estimators.classification.classifier import ClassifierClassLossGradients

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE

logger = logging.getLogger(__name__)


@deprecated(
    end_version="1.8.0",
    reason="Expectation over transformation has been replaced with "
    "art.preprocessing.expectation_over_transformation",
    replaced_by="art.preprocessing.expectation_over_transformation",
)
class ExpectationOverTransformations(ClassifierWrapper, ClassifierClassLossGradients):
    """
    Implementation of Expectation Over Transformations applied to classifier predictions and gradients, as introduced
    in Athalye et al. (2017).

    | Paper link: https://arxiv.org/abs/1707.07397
    """

    def __init__(self, classifier: "CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE", sample_size: int, transformation) -> None:
        """
        Create an expectation over transformations wrapper.

        :param classifier: The Classifier we want to wrap the functionality for the purpose of an attack.
        :param sample_size: Number of transformations to sample.
        :param transformation: An iterator over transformations.
        :type transformation: :class:`.Classifier`
        """
        super().__init__(classifier)
        self.sample_size = sample_size
        self.transformation = transformation
        self._predict = self.classifier.predict

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """
        Return the shape of one input sample.

        :return: Shape of one input sample.
        """
        return self._input_shape  # type: ignore

    def predict(self, x: np.ndarray, batch_size: int = 128, **kwargs) -> np.ndarray:  # pylint: disable=W0221
        """
        Perform prediction of the given classifier for a batch of inputs, taking an expectation over transformations.

        :param x: Input samples.
        :param batch_size: Size of batches.
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        """
        logger.info("Applying expectation over transformations.")
        prediction = self._predict(next(self.transformation())(x), **{"batch_size": batch_size})
        for _ in range(self.sample_size - 1):
            prediction += self._predict(next(self.transformation())(x), **{"batch_size": batch_size})
        return prediction / self.sample_size

    def fit(  # pylint: disable=W0221
        self, x: np.ndarray, y: np.ndarray, batch_size: int = 128, nb_epochs: int = 20, **kwargs
    ) -> None:
        """
        Fit the classifier using the training data `(x, y)`.

        :param x: Features in array of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).
        :param y: Target values (class labels in classification) in array of shape (nb_samples, nb_classes) in
                  one-hot encoding format.
        :param batch_size: Size of batches.
        :param nb_epochs: Number of epochs to use for training.
        :param kwargs: Dictionary of framework-specific arguments.
        """
        raise NotImplementedError

    def loss_gradient(
        self,
        x: np.ndarray,
        y: np.ndarray,
        training_mode: bool = False,
        **kwargs
        # pylint: disable=W0221
    ) -> np.ndarray:
        """
        Compute the gradient of the given classifier's loss function w.r.t. `x`, taking an expectation
        over transformations.

        :param x: Sample input with shape as expected by the model.
        :param y: Correct labels, one-hot encoded.
        :return: Array of gradients of the same shape as `x`.
        """
        logger.info("Applying expectation over transformations.")
        loss_gradient = self.classifier.loss_gradient(
            x=next(self.transformation())(x), y=y, training_mode=training_mode, **kwargs
        )
        for _ in range(self.sample_size - 1):
            loss_gradient += self.classifier.loss_gradient(
                x=next(self.transformation())(x), y=y, training_mode=training_mode, **kwargs
            )
        return loss_gradient / self.sample_size

    def class_gradient(  # pylint: disable=W0221
        self, x: np.ndarray, label: Union[int, List[int], None] = None, training_mode: bool = False, **kwargs
    ) -> np.ndarray:
        """
        Compute per-class derivatives of the given classifier w.r.t. `x`, taking an expectation over transformations.

        :param x: Sample input with shape as expected by the model.
        :param label: Index of a specific per-class derivative. If an integer is provided, the gradient of that class
                      output is computed for all samples. If multiple values as provided, the first dimension should
                      match the batch size of `x`, and each value will be used as target for its corresponding sample in
                      `x`. If `None`, then gradients for all classes will be computed for each sample.
        :param training_mode: `True` for model set to training mode and `'False` for model set to evaluation mode.
        :return: Array of gradients of input features w.r.t. each class in the form
                 `(batch_size, nb_classes, input_shape)` when computing for all classes, otherwise shape becomes
                 `(batch_size, 1, input_shape)` when `label` parameter is specified.
        """
        logger.info("Apply Expectation over Transformations.")
        class_gradient = self.classifier.class_gradient(
            x=next(self.transformation())(x), label=label, training_mode=training_mode, *kwargs
        )

        for _ in range(self.sample_size - 1):
            class_gradient += self.classifier.class_gradient(
                x=next(self.transformation())(x), label=label, training_mode=training_mode, *kwargs
            )

        return class_gradient / self.sample_size

    @property
    def layer_names(self) -> List[str]:
        """
        Return the hidden layers in the model, if applicable.

        :return: The hidden layers in the model, input and output layers excluded.
        :rtype: `list`

        .. warning:: `layer_names` tries to infer the internal structure of the model.
                     This feature comes with no guarantees on the correctness of the result.
                     The intended order of the layers tries to match their order in the model, but this is not
                     guaranteed either.
        """
        raise NotImplementedError

    def get_activations(self, x: np.ndarray, layer: Union[int, str], batch_size: int) -> np.ndarray:
        """
        Return the output of the specified layer for input `x`. `layer` is specified by layer index (between 0 and
        `nb_layers - 1`) or by name. The number of layers can be determined by counting the results returned by
        calling `layer_names`.

        :param x: Input for computing the activations.
        :param layer: Layer for computing the activations.
        :param batch_size: Size of batches.
        :return: The output of `layer`, where the first dimension is the batch size corresponding to `x`.
        """
        raise NotImplementedError

    @property
    def nb_classes(self) -> int:
        """
        Return the number of output classes.

        :return: Number of classes in the data.
        """
        return self._nb_classes

    def save(self, filename: str, path: Optional[str] = None) -> None:
        """
        Save a model to file specific to the backend framework.

        :param filename: Name of the file where to save the model.
        :param path: Path of the directory where to save the model. If no path is specified, the model will be stored in
                     the default data location of ART at `ART_DATA_PATH`.
        """
        raise NotImplementedError
