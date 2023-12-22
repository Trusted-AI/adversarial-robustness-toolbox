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
This module implements the abstract estimator `KerasEstimator` for Keras models.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

from typing import TYPE_CHECKING

import numpy as np

from art.estimators.estimator import (
    BaseEstimator,
    NeuralNetworkMixin,
    LossGradientsMixin,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from art.utils import KERAS_ESTIMATOR_TYPE


class KerasEstimator(NeuralNetworkMixin, LossGradientsMixin, BaseEstimator):
    """
    Estimator class for Keras models.
    """

    estimator_params = BaseEstimator.estimator_params + NeuralNetworkMixin.estimator_params

    def __init__(self, **kwargs) -> None:
        """
        Estimator class for Keras models.
        """
        super().__init__(**kwargs)

    def predict(self, x: np.ndarray, batch_size: int = 128, **kwargs):
        """
        Perform prediction of the neural network for samples `x`.

        :param x: Samples of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).
        :param batch_size: Batch size.
        :return: Predictions.
        :rtype: Format as expected by the `model`
        """
        return NeuralNetworkMixin.predict(self, x, batch_size=batch_size, **kwargs)

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
        NeuralNetworkMixin.fit(self, x, y, batch_size=batch_size, nb_epochs=nb_epochs, **kwargs)

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

    def clone_for_refitting(
        self,
    ) -> "KERAS_ESTIMATOR_TYPE":
        """
        Create a copy of the estimator that can be refit from scratch. Will inherit same architecture, optimizer and
        initialization as cloned model, but without weights.

        :return: new estimator
        """

        import tensorflow as tf
        import keras

        try:
            # only works for functionally defined models
            model = keras.models.clone_model(self.model, input_tensors=self.model.inputs)
        except ValueError as error:
            raise ValueError("Cannot clone custom models") from error

        optimizer = self.model.optimizer
        # reset optimizer variables
        for var in optimizer.variables():
            var.assign(tf.zeros_like(var))

        loss_weights = None
        weighted_metrics = None
        if self.model.compiled_loss:
            loss_weights = self.model.compiled_loss._loss_weights  # pylint: disable=W0212
        if self.model.compiled_metrics:
            weighted_metrics = self.model.compiled_metrics._weighted_metrics  # pylint: disable=W0212

        model.compile(
            optimizer=optimizer,
            loss=self.model.loss,
            metrics=[m.name for m in self.model.metrics],  # Need to copy metrics this way for keras
            loss_weights=loss_weights,
            weighted_metrics=weighted_metrics,
            run_eagerly=self.model.run_eagerly,
        )

        clone = type(self)(model=model)
        params = self.get_params()
        del params["model"]
        clone.set_params(**params)
        return clone
