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
This module implements Randomized Smoothing applied to classifier predictions.

| Paper link: https://arxiv.org/abs/1902.02918
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import List, Union, TYPE_CHECKING, Tuple

import numpy as np

from art.estimators.estimator import BaseEstimator, LossGradientsMixin, NeuralNetworkMixin
from art.estimators.certification.randomized_smoothing.randomized_smoothing import RandomizedSmoothingMixin
from art.estimators.classification import ClassifierMixin, ClassGradientsMixin

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_LOSS_GRADIENTS_TYPE

logger = logging.getLogger(__name__)


class NumpyRandomizedSmoothing(  # lgtm [py/conflicting-attributes]
    RandomizedSmoothingMixin,
    ClassGradientsMixin,
    ClassifierMixin,
    NeuralNetworkMixin,
    LossGradientsMixin,
    BaseEstimator,
):
    """
    Implementation of Randomized Smoothing applied to classifier predictions and gradients, as introduced
    in Cohen et al. (2019).

    | Paper link: https://arxiv.org/abs/1902.02918
    """

    estimator_params = (
        BaseEstimator.estimator_params
        + NeuralNetworkMixin.estimator_params
        + ClassifierMixin.estimator_params
        + ["classifier", "sample_size", "scale", "alpha"]
    )

    def __init__(
        self, classifier: "CLASSIFIER_LOSS_GRADIENTS_TYPE", sample_size: int, scale: float = 0.1, alpha: float = 0.001
    ):
        """
        Create a randomized smoothing wrapper.
        :param classifier: The Classifier we want to wrap the functionality for the purpose of smoothing.
        :param sample_size: Number of samples for smoothing
        :param scale: Standard deviation of Gaussian noise added.
        :param alpha: The failure probability of smoothing
        """
        super().__init__(
            model=classifier.model,
            clip_values=classifier.clip_values,
            preprocessing_defences=classifier.preprocessing_defences,
            postprocessing_defences=classifier.postprocessing_defences,
            preprocessing=classifier.preprocessing,
            sample_size=sample_size,
            scale=scale,
            alpha=alpha,
        )
        self._input_shape = classifier.input_shape
        self._nb_classes = classifier.nb_classes

        self.classifier = classifier

    @property
    def input_shape(self) -> Tuple[int, ...]:
        return self._input_shape

    def _predict_classifier(self, x: np.ndarray, batch_size: int, training_mode: bool, **kwargs) -> np.ndarray:
        """
        Perform prediction for a batch of inputs.

        :param x: Input samples.
        :param batch_size: Size of batches.
        :param training_mode: `True` for model set to training mode and `'False` for model set to evaluation mode.
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        """
        return self.classifier.predict(x=x, batch_size=batch_size, training_mode=training_mode, **kwargs)

    def _fit_classifier(self, x: np.ndarray, y: np.ndarray, batch_size: int, nb_epochs: int, **kwargs) -> None:
        """
         Fit the classifier on the training set `(x, y)`.

        :param x: Training data.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :param batch_size: Batch size.
        :param nb_epochs: Number of epochs to use for training.
        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported for PyTorch
                       and providing it takes no effect.
        """
        return self.classifier.fit(x, y, batch_size=batch_size, nb_epochs=nb_epochs, **kwargs)

    def loss_gradient(  # pylint: disable=W0221
        self, x: np.ndarray, y: np.ndarray, training_mode: bool = False, **kwargs
    ) -> np.ndarray:
        """
        Compute the gradient of the given classifier's loss function w.r.t. `x` of the original classifier.
        :param x: Sample input with shape as expected by the model.
        :param y: Correct labels, one-hot encoded.
        :param training_mode: `True` for model set to training mode and `'False` for model set to evaluation mode.
        :return: Array of gradients of the same shape as `x`.
        """
        return self.classifier.loss_gradient(x=x, y=y, training_mode=training_mode, **kwargs)

    def class_gradient(  # pylint: disable=W0221
        self, x: np.ndarray, label: Union[int, List[int]] = None, training_mode: bool = False, **kwargs
    ) -> np.ndarray:
        """
        Compute per-class derivatives of the given classifier w.r.t. `x` of original classifier.
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
        return self.classifier.class_gradient(x=x, label=label, training_mode=training_mode, **kwargs)  # type: ignore
