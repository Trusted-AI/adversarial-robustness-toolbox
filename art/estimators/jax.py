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
This module implements the abstract estimator `JaxEstimator` for Jax models.
"""
import logging
from typing import Any, Tuple

import numpy as np

from art.estimators.estimator import BaseEstimator, LossGradientsMixin, NeuralNetworkMixin


logger = logging.getLogger(__name__)


class JaxEstimator(NeuralNetworkMixin, LossGradientsMixin, BaseEstimator):
    """
    Estimator class for Jax models.
    """

    estimator_params = BaseEstimator.estimator_params + NeuralNetworkMixin.estimator_params

    def __init__(self, **kwargs) -> None:
        """
        Estimator class for Jax models.

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
        :param device_type: Type of device on which the classifier is run, either `gpu` or `cpu`.
        """
        super().__init__(**kwargs)

        JaxEstimator._check_params(self)

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

    def set_params(self, **kwargs) -> None:
        """
        Take a dictionary of parameters and apply checks before setting them as attributes.

        :param kwargs: A dictionary of attributes.
        """
        super().set_params(**kwargs)
        self._check_params()

    def _check_params(self) -> None:
        super()._check_params()

    def _apply_preprocessing(self, x, y, fit: bool = False, no_grad=True) -> Tuple[Any, Any]:  # pylint: disable=W0221
        """
        Apply all preprocessing defences of the estimator on the raw inputs `x` and `y`. This function is should
        only be called from function `_apply_preprocessing`.

        The method overrides art.estimators.estimator::BaseEstimator._apply_preprocessing().
        It requires all defenses to have a method `forward()`.
        It converts numpy arrays to PyTorch tensors first, then chains a series of defenses by calling
        defence.forward() which contains PyTorch operations. At the end, it converts PyTorch tensors
        back to numpy arrays.

        :param x: Samples.
        :type x: Format as expected by the `model`
        :param y: Target values.
        :type y: Format as expected by the `model`
        :param fit: `True` if the function is call before fit/training and `False` if the function is called before a
                    predict operation.
        :param no_grad: `True` if no gradients required.
        :type no_grad: bool
        :return: Tuple of `x` and `y` after applying the defences and standardisation.
        :rtype: Format as expected by the `model`
        """
        return x, y

    def _apply_preprocessing_gradient(self, x, gradients, fit=False):
        """
        Apply the backward pass to the gradients through all preprocessing defences that have been applied to `x`
        and `y` in the forward pass. This function is should only be called from function
        `_apply_preprocessing_gradient`.

        The method overrides art.estimators.estimator::LossGradientsMixin._apply_preprocessing_gradient().
        It requires all defenses to have a method estimate_forward().
        It converts numpy arrays to PyTorch tensors first, then chains a series of defenses by calling
        defence.estimate_forward() which contains differentiable estimate of the operations. At the end,
        it converts PyTorch tensors back to numpy arrays.

        :param x: Samples.
        :type x: Format as expected by the `model`
        :param gradients: Gradients before backward pass through preprocessing defences.
        :type gradients: Format as expected by the `model`
        :param fit: `True` if the gradients are computed during training.
        :return: Gradients after backward pass through preprocessing defences.
        :rtype: Format as expected by the `model`
        """
        return gradients
