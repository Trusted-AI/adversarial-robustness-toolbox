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
from typing import Any

from art.estimators.estimator import BaseEstimator


class LanguageModelMixin(abc.ABC):
    """
    Mix-in base class for ART language models.
    """

    pass


class LanguageModel(LanguageModelMixin, BaseEstimator, abc.ABC):
    """
    Abstract base class for ART language models used to define common types and methods.
    """

    @abc.abstractmethod
    def tokenize(self, x: Any, **kwargs) -> Any:
        """
        Token the input `x`.

        :param text: Samples to be tokenized.
        :type text: Format as expected by the `tokenizer`
        :return: Tokenized output by the tokenizer.
        :rtype: Format as produced by the `tokenizer`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def encode(self, x: Any, **kwargs) -> Any:
        """
        Encode the input `x`.

        :param text: Samples to be encoded.
        :type text: Format as expected by the `tokenizer`
        :return: Encoded output by the tokenizer.
        :rtype: Format as produced by the `tokenizer`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def batch_encode(self, x: Any, **kwargs) -> Any:
        """
        Encode the input `x`.

        :param text: Samples to be encoded.
        :type text: Format as expected by the `tokenizer`
        :return: Encoded output by the tokenizer.
        :rtype: Format as produced by the `tokenizer`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def decode(self, x: Any, **kwargs) -> Any:
        """
        Decode the input `x`.

        :param tokens: Samples to be decoded.
        :type tokens: Format as expected by the `tokenizer`
        :return: decoded output by the tokenizer.
        :rtype: Format as produced by the `tokenizer`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def batch_decode(self, x: Any, **kwargs) -> Any:
        """
        Decode the input `x`.

        :param tokens: Samples to be decoded.
        :type tokens: Format as expected by the `tokenizer`
        :return: decoded output by the tokenizer.
        :rtype: Format as produced by the `tokenizer`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, x: Any, **kwargs) -> Any:
        """
        Perform prediction of the language model for input `x`.

        :param text: Samples to be tokenized.
        :type text: Format as expected by the `tokenizer`
        :return: Predictions by the model.
        :rtype: Format as produced by the `model`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def generate(self, x: Any, **kwargs) -> Any:
        """
        Generate text using the language model from input `x`.

        :param text: Samples to be tokenized.
        :type text: Format as expected by the `tokenizer`
        :return: Generated text by the model.
        :rtype: Format as produced by the `tokenizer`
        """
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
