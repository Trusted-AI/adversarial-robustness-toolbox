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
This module implements the abstract base class for all poison filtering defences.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import abc
import sys
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

import numpy as np

# Ensure compatibility with Python 2 and 3 when using ABCMeta
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str("ABC"), (), {})

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE


class PoisonFilteringDefence(ABC):
    """
    Base class for all poison filtering defences.
    """

    defence_params = ["classifier"]

    def __init__(self, classifier: "CLASSIFIER_TYPE", x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Create an :class:`.ActivationDefence` object with the provided classifier.

        :param classifier: Model evaluated for poison.
        :param x_train: dataset used to train the classifier.
        :param y_train: labels used to train the classifier.
        """
        self.classifier = classifier
        self.x_train = x_train
        self.y_train = y_train

    @abc.abstractmethod
    def detect_poison(self, **kwargs) -> Tuple[dict, List[int]]:
        """
        Detect poison.

        :param kwargs: Defence-specific parameters used by child classes.
        :return: Dictionary with report and list with items identified as poison.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate_defence(self, is_clean: np.ndarray, **kwargs) -> str:
        """
        Evaluate the defence given the labels specifying if the data is poisoned or not.

        :param is_clean: 1-D array where is_clean[i]=1 means x_train[i] is clean and is_clean[i]=0 that it's poison.
        :param kwargs: Defence-specific parameters used by child classes.
        :return: JSON object with confusion matrix.
        """
        raise NotImplementedError

    def set_params(self, **kwargs) -> None:
        """
        Take in a dictionary of parameters and apply attack-specific checks before saving them as attributes.

        :param kwargs: A dictionary of defence-specific parameters.
        """
        for key, value in kwargs.items():
            if key in self.defence_params:
                setattr(self, key, value)
        self._check_params()

    def get_params(self) -> Dict[str, Any]:
        """
        Returns dictionary of parameters used to run defence.

        :return: Dictionary of parameters of the method.
        """
        dictionary = {param: getattr(self, param) for param in self.defence_params}
        return dictionary

    def _check_params(self) -> None:
        pass
