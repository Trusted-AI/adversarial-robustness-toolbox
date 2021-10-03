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
This module implements the abstract estimator `PyTorchEstimator` for PyTorch models.
"""
import logging
from typing import TYPE_CHECKING, Any, List, Tuple

import numpy as np

from art.estimators.estimator import BaseEstimator, LossGradientsMixin, NeuralNetworkMixin

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


class PyTorchEstimator(NeuralNetworkMixin, LossGradientsMixin, BaseEstimator):
    """
    Estimator class for PyTorch models.
    """

    estimator_params = (
        BaseEstimator.estimator_params
        + NeuralNetworkMixin.estimator_params
        + [
            "device_type",
        ]
    )

    def __init__(self, device_type: str = "gpu", **kwargs) -> None:
        """
        Estimator class for PyTorch models.

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
        import torch  # lgtm [py/repeated-import]

        preprocessing = kwargs.get("preprocessing")
        if isinstance(preprocessing, tuple):
            from art.preprocessing.standardisation_mean_std.pytorch import StandardisationMeanStdPyTorch

            kwargs["preprocessing"] = StandardisationMeanStdPyTorch(
                mean=preprocessing[0], std=preprocessing[1], device_type=device_type
            )

        super().__init__(**kwargs)

        self._device_type = device_type

        # Set device
        if device_type == "cpu" or not torch.cuda.is_available():
            self._device = torch.device("cpu")
        else:  # pragma: no cover
            cuda_idx = torch.cuda.current_device()
            self._device = torch.device("cuda:{}".format(cuda_idx))

        PyTorchEstimator._check_params(self)

    @property
    def device_type(self) -> str:
        """
        Return the type of device on which the estimator is run.

        :return: Type of device on which the estimator is run, either `gpu` or `cpu`.
        """
        return self._device_type  # type: ignore

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
        from art.defences.preprocessor.preprocessor import PreprocessorPyTorch

        super()._check_params()
        self.all_framework_preprocessing = all(
            (isinstance(p, PreprocessorPyTorch) for p in self.preprocessing_operations)
        )

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
        import torch  # lgtm [py/repeated-import]

        from art.preprocessing.standardisation_mean_std.numpy import StandardisationMeanStd
        from art.preprocessing.standardisation_mean_std.pytorch import StandardisationMeanStdPyTorch

        if not self.preprocessing_operations:
            return x, y

        input_is_tensor = isinstance(x, torch.Tensor)

        if self.all_framework_preprocessing and not (not input_is_tensor and x.dtype == np.object):
            if not input_is_tensor:
                # Convert np arrays to torch tensors.
                x = torch.tensor(x, device=self._device)
                if y is not None:
                    y = torch.tensor(y, device=self._device)

            def chain_processes(x, y):
                for preprocess in self.preprocessing_operations:
                    if fit:
                        if preprocess.apply_fit:
                            x, y = preprocess.forward(x, y)
                    else:
                        if preprocess.apply_predict:
                            x, y = preprocess.forward(x, y)
                return x, y

            if no_grad:
                with torch.no_grad():
                    x, y = chain_processes(x, y)
            else:
                x, y = chain_processes(x, y)

            # Convert torch tensors back to np arrays.
            if not input_is_tensor:
                x = x.cpu().numpy()
                if y is not None:
                    y = y.cpu().numpy()

        elif len(self.preprocessing_operations) == 1 or (
            len(self.preprocessing_operations) == 2
            and isinstance(self.preprocessing_operations[-1], (StandardisationMeanStd, StandardisationMeanStdPyTorch))
        ):
            # Compatible with non-PyTorch defences if no chaining.
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
        import torch  # lgtm [py/repeated-import]

        from art.preprocessing.standardisation_mean_std.numpy import StandardisationMeanStd
        from art.preprocessing.standardisation_mean_std.pytorch import StandardisationMeanStdPyTorch

        if not self.preprocessing_operations:
            return gradients

        input_is_tensor = isinstance(x, torch.Tensor)

        if self.all_framework_preprocessing and not (not input_is_tensor and x.dtype == np.object):
            # Convert np arrays to torch tensors.
            x = torch.tensor(x, device=self._device, requires_grad=True)
            gradients = torch.tensor(gradients, device=self._device)
            x_orig = x

            for preprocess in self.preprocessing_operations:
                if fit:
                    if preprocess.apply_fit:
                        x = preprocess.estimate_forward(x)
                else:
                    if preprocess.apply_predict:
                        x = preprocess.estimate_forward(x)

            x.backward(gradients)

            # Convert torch tensors back to np arrays.
            gradients = x_orig.grad.detach().cpu().numpy()
            if gradients.shape != x_orig.shape:
                raise ValueError(
                    "The input shape is {} while the gradient shape is {}".format(x.shape, gradients.shape)
                )

        elif len(self.preprocessing_operations) == 1 or (
            len(self.preprocessing_operations) == 2
            and isinstance(self.preprocessing_operations[-1], (StandardisationMeanStd, StandardisationMeanStdPyTorch))
        ):
            # Compatible with non-PyTorch defences if no chaining.
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

    def _set_layer(self, train: bool, layerinfo: List["torch.nn.modules.Module"]) -> None:
        """
        Set all layers that are an instance of `layerinfo` into training or evaluation mode.

        :param train: False for evaluation mode.
        :param layerinfo: List of module types.
        """
        import torch  # lgtm [py/repeated-import]

        assert all((issubclass(layer, torch.nn.modules.Module) for layer in layerinfo))  # type: ignore

        def set_train(layer, layerinfo=layerinfo):
            "Set layer into training mode if instance of `layerinfo`."
            if isinstance(layer, tuple(layerinfo)):
                layer.train()

        def set_eval(layer, layerinfo=layerinfo):
            "Set layer into evaluation mode if instance of `layerinfo`."
            if isinstance(layer, tuple(layerinfo)):
                layer.eval()

        if train:
            self._model.apply(set_train)
        else:
            self._model.apply(set_eval)

    def set_dropout(self, train: bool) -> None:
        """
        Set all dropout layers into train or eval mode.

        :param train: False for evaluation mode.
        """
        import torch  # lgtm [py/repeated-import]

        # pylint: disable=W0212
        self._set_layer(train=train, layerinfo=[torch.nn.modules.dropout._DropoutNd])  # type: ignore

    def set_batchnorm(self, train: bool) -> None:
        """
        Set all batch normalization layers into train or eval mode.

        :param train: False for evaluation mode.
        """
        import torch  # lgtm [py/repeated-import]

        # pylint: disable=W0212
        self._set_layer(train=train, layerinfo=[torch.nn.modules.batchnorm._BatchNorm])  # type: ignore
