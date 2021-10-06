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
from typing import Any, Tuple, TYPE_CHECKING

import numpy as np

from art import config
from art.estimators.estimator import (
    BaseEstimator,
    LossGradientsMixin,
    NeuralNetworkMixin,
)

if TYPE_CHECKING:
    import tensorflow as tf

logger = logging.getLogger(__name__)


class TensorFlowEstimator(NeuralNetworkMixin, LossGradientsMixin, BaseEstimator):
    """
    Estimator class for TensorFlow models.
    """

    estimator_params = BaseEstimator.estimator_params + NeuralNetworkMixin.estimator_params

    def __init__(self, **kwargs) -> None:
        """
        Estimator class for TensorFlow models.
        """
        self._sess: "tf.python.client.session.Session" = None
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

    @property
    def sess(self) -> "tf.python.client.session.Session":
        """
        Get current TensorFlow session.

        :return: The current TensorFlow session.
        """
        if self._sess is not None:
            return self._sess

        raise NotImplementedError("A valid TensorFlow session is not available.")


class TensorFlowV2Estimator(NeuralNetworkMixin, LossGradientsMixin, BaseEstimator):
    """
    Estimator class for TensorFlow v2 models.
    """

    estimator_params = BaseEstimator.estimator_params + NeuralNetworkMixin.estimator_params

    def __init__(self, **kwargs):
        """
        Estimator class for TensorFlow v2 models.
        """
        preprocessing = kwargs.get("preprocessing")
        if isinstance(preprocessing, tuple):
            from art.preprocessing.standardisation_mean_std.tensorflow import StandardisationMeanStdTensorFlow

            kwargs["preprocessing"] = StandardisationMeanStdTensorFlow(mean=preprocessing[0], std=preprocessing[1])

        super().__init__(**kwargs)
        TensorFlowV2Estimator._check_params(self)

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
        from art.defences.preprocessor.preprocessor import PreprocessorTensorFlowV2

        super()._check_params()
        self.all_framework_preprocessing = all(
            (isinstance(p, PreprocessorTensorFlowV2) for p in self.preprocessing_operations)
        )

    def _apply_preprocessing(self, x, y, fit: bool = False) -> Tuple[Any, Any]:
        """
        Apply all preprocessing defences of the estimator on the raw inputs `x` and `y`. This function is should
        only be called from function `_apply_preprocessing`.

        The method overrides art.estimators.estimator::BaseEstimator._apply_preprocessing().
        It requires all defenses to have a method `forward()`.
        It converts numpy arrays to TensorFlow tensors first, then chains a series of defenses by calling
        defence.forward() which contains TensorFlow operations. At the end, it converts TensorFlow tensors
        back to numpy arrays.

        :param x: Samples.
        :type x: Format as expected by the `model`
        :param y: Target values.
        :type y: Format as expected by the `model`
        :param fit: `True` if the function is call before fit/training and `False` if the function is called before a
                    predict operation.
        :return: Tuple of `x` and `y` after applying the defences and standardisation.
        :rtype: Format as expected by the `model`
        """
        import tensorflow as tf  # lgtm [py/repeated-import]
        from art.preprocessing.standardisation_mean_std.numpy import StandardisationMeanStd
        from art.preprocessing.standardisation_mean_std.tensorflow import StandardisationMeanStdTensorFlow

        if not self.preprocessing_operations:
            return x, y

        input_is_tensor = isinstance(x, tf.Tensor)

        if self.all_framework_preprocessing and not (not input_is_tensor and x.dtype == np.object):
            # Convert np arrays to torch tensors.
            if not input_is_tensor:
                x = tf.convert_to_tensor(x)
                if y is not None:
                    y = tf.convert_to_tensor(y)

            for preprocess in self.preprocessing_operations:
                if fit:
                    if preprocess.apply_fit:
                        x, y = preprocess.forward(x, y)
                else:
                    if preprocess.apply_predict:
                        x, y = preprocess.forward(x, y)

            # Convert torch tensors back to np arrays.
            if not input_is_tensor:
                x = x.numpy()
                if y is not None:
                    y = y.numpy()

        elif len(self.preprocessing_operations) == 1 or (
            len(self.preprocessing_operations) == 2
            and isinstance(
                self.preprocessing_operations[-1], (StandardisationMeanStd, StandardisationMeanStdTensorFlow)
            )
        ):
            # Compatible with non-TensorFlow defences if no chaining.
            for preprocess in self.preprocessing_operations:
                if fit:
                    if preprocess.apply_fit:
                        x, y = preprocess(x, y)
                else:
                    if preprocess.apply_predict:
                        x, y = preprocess(x, y)

        else:
            raise NotImplementedError("The current combination of preprocessing types is not supported.")

        return x, y

    def _apply_preprocessing_gradient(self, x, gradients, fit=False):
        """
        Apply the backward pass to the gradients through all preprocessing defences that have been applied to `x`
        and `y` in the forward pass. This function is should only be called from function
        `_apply_preprocessing_gradient`.

        The method overrides art.estimators.estimator::LossGradientsMixin._apply_preprocessing_gradient().
        It requires all defenses to have a method estimate_forward().
        It converts numpy arrays to TensorFlow tensors first, then chains a series of defenses by calling
        defence.estimate_forward() which contains differentiable estimate of the operations. At the end,
        it converts TensorFlow tensors back to numpy arrays.

        :param x: Samples.
        :type x: Format as expected by the `model`
        :param gradients: Gradients before backward pass through preprocessing defences.
        :type gradients: Format as expected by the `model`
        :param fit: `True` if the gradients are computed during training.
        :return: Gradients after backward pass through preprocessing defences.
        :rtype: Format as expected by the `model`
        """
        import tensorflow as tf  # lgtm [py/repeated-import]
        from art.preprocessing.standardisation_mean_std.numpy import StandardisationMeanStd
        from art.preprocessing.standardisation_mean_std.tensorflow import StandardisationMeanStdTensorFlow

        if not self.preprocessing_operations:
            return gradients

        input_is_tensor = isinstance(x, tf.Tensor)

        if self.all_framework_preprocessing and not (not input_is_tensor and x.dtype == np.object):
            with tf.GradientTape() as tape:
                # Convert np arrays to TensorFlow tensors.
                x = tf.convert_to_tensor(x, dtype=config.ART_NUMPY_DTYPE)
                tape.watch(x)
                gradients = tf.convert_to_tensor(gradients, dtype=config.ART_NUMPY_DTYPE)
                x_orig = x

                for preprocess in self.preprocessing_operations:
                    if fit:
                        if preprocess.apply_fit:
                            x = preprocess.estimate_forward(x)
                    else:
                        if preprocess.apply_predict:
                            x = preprocess.estimate_forward(x)

            x_grad = tape.gradient(target=x, sources=x_orig, output_gradients=gradients)

            # Convert torch tensors back to np arrays.
            gradients = x_grad.numpy()
            if gradients.shape != x_orig.shape:  # pragma: no cover
                raise ValueError(
                    "The input shape is {} while the gradient shape is {}".format(x.shape, gradients.shape)
                )

        elif len(self.preprocessing_operations) == 1 or (
            len(self.preprocessing_operations) == 2
            and isinstance(
                self.preprocessing_operations[-1], (StandardisationMeanStd, StandardisationMeanStdTensorFlow)
            )
        ):
            # Compatible with non-TensorFlow defences if no chaining.
            for preprocess in self.preprocessing_operations[::-1]:
                if fit:
                    if preprocess.apply_fit:
                        gradients = preprocess.estimate_gradient(x, gradients)
                else:
                    if preprocess.apply_predict:
                        gradients = preprocess.estimate_gradient(x, gradients)

        else:
            raise NotImplementedError("The current combination of preprocessing types is not supported.")

        return gradients
