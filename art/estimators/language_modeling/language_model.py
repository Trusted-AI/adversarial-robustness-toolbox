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
    Mix-in base class for ART language models.
    """

    pass


class LanguageModel(LanguageModelMixin, abc.ABC):
    """
    Abstract base class for ART language models.
    """

    estimator_params = (
        BaseEstimator.estimator_params
        + [
            "device_type",
        ]
    )

    def __init__(self, device_type: str = "gpu", **kwargs) -> None:
        """
        Abstract base class for language models.

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
        """
        Token the input `text`.

        :param text: Samples to be tokenized.
        :type text: Format as expected by the `tokenizer`
        :return: Tokenized output by the tokenizer.
        :rtype: Format as produced by the `tokenizer`
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def encode(self, text: Any, **kwargs) -> Any:
        """
        Encode the input `text`.

        :param text: Samples to be encoded.
        :type text: Format as expected by the `tokenizer`
        :return: Encoded output by the tokenizer.
        :rtype: Format as produced by the `tokenizer`
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def decode(self, tokens: Any, **kwargs) -> Any:
        """
        Decode the input `tokens`.

        :param tokens: Samples to be decoded.
        :type tokens: Format as expected by the `tokenizer`
        :return: decoded output by the tokenizer.
        :rtype: Format as produced by the `tokenizer`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, text: Any, **kwargs) -> Any:
        """
        Perform prediction of the language model for input `text`.

        :param text: Samples to be tokenized.
        :type text: Format as expected by the `tokenizer`
        :return: Predictions by the model.
        :rtype: Format as produced by the `model`
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def generate(self, text: Any, **kwargs) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def fit(self, x: Any, y: Any, **kwargs) -> None:
        """
        Fit the estimator using the training data `(x, y)`.

        :param x: Training data.
        :type x: Format as expected by the `model`
        :param y: Target values.
        :type y: Format as expected by the `model`
        """
        raise NotImplementedError
