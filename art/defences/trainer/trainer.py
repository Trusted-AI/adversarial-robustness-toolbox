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
This module implements the abstract base class for defences that adversarially train models.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import abc
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_LOSS_GRADIENTS_TYPE


class Trainer(abc.ABC):
    """
    Abstract base class for training defences.
    """

    def __init__(self, classifier: "CLASSIFIER_LOSS_GRADIENTS_TYPE") -> None:
        """
        Create a adversarial training object
        """
        self._classifier = classifier

    @abc.abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Train the model.

        :param x: Training data.
        :param y: Labels for the training data.
        :param kwargs: Other parameters.
        """
        raise NotImplementedError

    @property
    def classifier(self) -> "CLASSIFIER_LOSS_GRADIENTS_TYPE":
        """
        Access function to get the classifier.

        :return: The classifier.
        """
        return self._classifier

    def get_classifier(self) -> "CLASSIFIER_LOSS_GRADIENTS_TYPE":
        """
        Return the classifier trained via adversarial training.

        :return: The classifier.
        """
        return self._classifier
