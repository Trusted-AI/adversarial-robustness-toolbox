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
This module implements the abstract base class for defences that transform a classifier into a more robust classifier.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import abc
from typing import List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE


class Transformer(abc.ABC):
    """
    Abstract base class for transformation defences.
    """

    params: List[str] = list()

    def __init__(self, classifier: "CLASSIFIER_TYPE") -> None:
        """
        Create a transformation object.

        :param classifier: A trained classifier.
        """
        self.classifier = classifier
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """
        Return the state of the transformation object.

        :return: `True` if the transformation model has been fitted (if this applies).
        """
        return self._is_fitted

    def get_classifier(self) -> "CLASSIFIER_TYPE":
        """
        Get the internal classifier.

        :return: The internal classifier.
        """
        return self.classifier

    @abc.abstractmethod
    def __call__(self, x: np.ndarray, transformed_classifier: "CLASSIFIER_TYPE") -> "CLASSIFIER_TYPE":
        """
        Perform the transformation defence and return a robuster classifier.

        :param x: Dataset for training the transformed classifier.
        :param transformed_classifier: A classifier to be transformed for increased robustness.
        :return: The transformed classifier.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fit(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> None:
        """
        Fit the parameters of the transformer if it has any.

        :param x: Training set to fit the transformer.
        :param y: Labels for the training set.
        :param kwargs: Other parameters.
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
