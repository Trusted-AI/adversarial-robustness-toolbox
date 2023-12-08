# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2023
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
This module implements mixin abstract base class for all language models in ART.
"""

import abc

from typing import List, Optional, Tuple, Union, Dict, Callable, Any, TYPE_CHECKING

import numpy as np

from art.estimators.estimator import BaseEstimator

if TYPE_CHECKING:
    import torch


class LanguageModelMixin(abc.ABC):
    """
    Mix-in Base class for ART language models.
    """

    pass


class LanguageModel(LanguageModelMixin, BaseEstimator, abc.ABC):
    """
    Typing variable definition.
    """

    estimator_params = (
        BaseEstimator.estimator_params
        + [
            "device_type",
        ]
    )

    def __init__(self, device_type: str = "gpu", **kwargs) -> None:
        """
        Estimator class for PyTorch models.

        :param model: The model.
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
        import torch

        super().__init__(**kwargs)

        self._device_type = device_type

        # Set device
        if device_type == "cpu" or not torch.cuda.is_available():
            self._device = torch.device("cpu")
        else:  # pragma: no cover
            cuda_idx = torch.cuda.current_device()
            self._device = torch.device(f"cuda:{cuda_idx}")

        self._check_params()

    @property
    def device_type(self) -> str:
        """
        Return the type of device on which the estimator is run.

        :return: Type of device on which the estimator is run, either `gpu` or `cpu`.
        """
        return self._device_type  # type: ignore

    @abc.abstractmethod
    def predict(self, x: np.ndarray, batch_size: int = 128, **kwargs):
        """
        Perform prediction of the neural network for samples `x`.

        :param x: Samples of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).
        :param batch_size: Batch size.
        :return: Predictions.
        :rtype: Format as expected by the `model`
        """
        raise NotImplementedError

    @abc.abstractmethod
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

    def _set_layer(self, train: bool, layerinfo: List["torch.nn.modules.Module"]) -> None:
        """
        Set all layers that are an instance of `layerinfo` into training or evaluation mode.

        :param train: False for evaluation mode.
        :param layerinfo: List of module types.
        """
        import torch

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
        import torch

        # pylint: disable=W0212
        self._set_layer(train=train, layerinfo=[torch.nn.modules.dropout._DropoutNd])  # type: ignore

    def set_batchnorm(self, train: bool) -> None:
        """
        Set all batch normalization layers into train or eval mode.

        :param train: False for evaluation mode.
        """
        import torch

        # pylint: disable=W0212
        self._set_layer(train=train, layerinfo=[torch.nn.modules.batchnorm._BatchNorm])  # type: ignore

    def set_multihead_attention(self, train: bool) -> None:
        """
        Set all multi-head attention layers into train or eval mode.

        :param train: False for evaluation mode.
        """
        import torch

        # pylint: disable=W0212
        self._set_layer(train=train, layerinfo=[torch.nn.modules.MultiheadAttention])  # type: ignore
