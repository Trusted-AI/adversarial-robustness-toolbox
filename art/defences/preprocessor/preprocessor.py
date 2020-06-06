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
This module implements the abstract base class for defences that pre-process input data.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import abc
from typing import List, Optional, Tuple

import numpy as np


class Preprocessor(abc.ABC):
    """
    Abstract base class for preprocessing defences.
    """

    params: List[str] = []

    def __init__(self) -> None:
        """
        Create a preprocessing object.
        """
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """
        Return the state of the preprocessing object.

        :return: `True` if the preprocessing model has been fitted (if this applies).
        """
        return self._is_fitted

    @property
    @abc.abstractmethod
    def apply_fit(self) -> bool:
        """
        Property of the defence indicating if it should be applied at training time.

        :return: `True` if the defence should be applied when fitting a model, `False` otherwise.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def apply_predict(self) -> bool:
        """
        Property of the defence indicating if it should be applied at test time.

        :return: `True` if the defence should be applied at prediction time, `False` otherwise.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Perform data preprocessing and return preprocessed data as tuple.

        :param x: Dataset to be preprocessed.
        :param y: Labels to be preprocessed.
        :return: Preprocessed data.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fit(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> None:
        """
        Fit the parameters of the data preprocessor if it has any.

        :param x: Training set to fit the preprocessor.
        :param y: Labels for the training set.
        :param kwargs: Other parameters.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def estimate_gradient(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """
        Provide an estimate of the gradients of the defence for the backward pass. If the defence is not differentiable,
        this is an estimate of the gradient, most often replacing the computation performed by the defence with the
        identity function.

        :param x: Input data for which the gradient is estimated. First dimension is the batch size.
        :param grad: Gradient value so far.
        :return: The gradient (estimate) of the defence.
        """
        raise NotImplementedError

    def set_params(self, **kwargs) -> None:
        """
        Take in a dictionary of parameters and apply checks before saving them as attributes.
        """
        for key, value in kwargs.items():
            if key in self.params:
                setattr(self, key, value)
        self._check_params()

    def _check_params(self) -> None:
        pass
