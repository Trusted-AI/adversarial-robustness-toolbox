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


class LanguageModel(LanguageModelMixin, abc.ABC):
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
        :param tokenizer: The tokenizer.
        :param device_type: Type of device on which the classifier is run, either `gpu` or `cpu`.
        """
        import torch

        super().__init__()

        self._device_type = device_type

        # Set device
        if device_type == "cpu" or not torch.cuda.is_available():
            self._device = torch.device("cpu")
        else:  # pragma: no cover
            cuda_idx = torch.cuda.current_device()
            self._device = torch.device(f"cuda:{cuda_idx}")

    @property
    def device_type(self) -> str:
        """
        Return the type of device on which the estimator is run.

        :return: Type of device on which the estimator is run, either `gpu` or `cpu`.
        """
        return self._device_type  # type: ignore

    @abc.abstractmethod
    def tokenize(self, text: Any, **kwargs) -> Any:
        raise NotImplementedError
    
    @abc.abstractmethod
    def encode(self, text: Any, **kwargs) -> Any:
        raise NotImplementedError
    
    @abc.abstractmethod
    def decode(self, tokens: Any, **kwargs) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, text: Any, **kwargs) -> Any:
        raise NotImplementedError
    
    @abc.abstractmethod
    def generate(self, text: Any, **kwargs) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def fit(self, x: Any, y: Any, **kwargs) -> None:
        raise NotImplementedError

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
