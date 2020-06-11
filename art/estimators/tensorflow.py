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
This module implements the abstract estimators `TensorFlowEstimator` and `TensorFlowV2Estimator` for TensorFlow models.
"""
import logging

from art.estimators.estimator import (
    BaseEstimator,
    LossGradientsMixin,
    NeuralNetworkMixin,
)

logger = logging.getLogger(__name__)


class TensorFlowEstimator(NeuralNetworkMixin, LossGradientsMixin, BaseEstimator):
    """
    Estimator class for TensorFlow models.
    """
    def __init__(self, **kwargs):
        """
        Estimator class for TensorFlow models.
        """
        super().__init__(**kwargs)

    def predict(self, x, batch_size=128, **kwargs):
        """
        Perform prediction of the neural network for samples `x`.

        :param x: Samples of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).
        :type x: `np.ndarray`
        :param batch_size: Batch size.
        :type batch_size: `int`
        :return: Predictions.
        :rtype: Format as expected by the `model`
        """
        return NeuralNetworkMixin.predict(self, x, batch_size=128, **kwargs)

    def fit(self, x, y, batch_size=128, nb_epochs=20, **kwargs):
        """
        Fit the model of the estimator on the training data `x` and `y`.

        :param x: Samples of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).
        :type x: `np.ndarray`
        :param y: Target values.
        :type y: Format as expected by the `model`
        :param batch_size: Batch size.
        :type batch_size: `int`
        :param nb_epochs: Number of training epochs.
        :type nb_epochs: `int`
        :return: `None`
        """
        NeuralNetworkMixin.fit(self, x, y, batch_size=128, nb_epochs=20, **kwargs)


class TensorFlowV2Estimator(NeuralNetworkMixin, LossGradientsMixin, BaseEstimator):
    """
    Estimator class for TensorFlow v2 models.
    """
    def __init__(self, **kwargs):
        """
        Estimator class for TensorFlow v2 models.
        """
        super().__init__(**kwargs)

    def predict(self, x, batch_size=128, **kwargs):
        """
        Perform prediction of the neural network for samples `x`.

        :param x: Samples of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).
        :type x: `np.ndarray`
        :param batch_size: Batch size.
        :type batch_size: `int`
        :return: Predictions.
        :rtype: Format as expected by the `model`
        """
        return NeuralNetworkMixin.predict(self, x, batch_size=128, **kwargs)

    def fit(self, x, y, batch_size=128, nb_epochs=20, **kwargs):
        """
        Fit the model of the estimator on the training data `x` and `y`.

        :param x: Samples of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).
        :type x: `np.ndarray`
        :param y: Target values.
        :type y: Format as expected by the `model`
        :param batch_size: Batch size.
        :type batch_size: `int`
        :param nb_epochs: Number of training epochs.
        :type nb_epochs: `int`
        :return: `None`
        """
        NeuralNetworkMixin.fit(self, x, y, batch_size=128, nb_epochs=20, **kwargs)
