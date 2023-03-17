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
This module implements the abstract base class for all evasion detectors.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import abc
from typing import Any, Dict, List, Tuple

import numpy as np


class EvasionDetector(abc.ABC):
    """
    Abstract base class for all evasion detectors.
    """

    defence_params: List[str] = []

    def __init__(self) -> None:
        """
        Create an evasion detector object.
        """
        pass

    @abc.abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray, batch_size: int = 128, nb_epochs: int = 20, **kwargs) -> None:
        """
        Fit the detection classifier if necessary.

        :param x: Training set to fit the detector.
        :param y: Labels for the training set.
        :param batch_size: Size of batches.
        :param nb_epochs: Number of epochs to use for training.
        :param kwargs: Other parameters.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def detect(self, x: np.ndarray, batch_size: int = 128, **kwargs) -> Tuple[dict, np.ndarray]:
        """
        Perform detection of adversarial data and return prediction as tuple.

        :param x: Data sample on which to perform detection.
        :param batch_size: Size of batches.
        :param kwargs: Defence-specific parameters used by child classes.
        :return: (report, is_adversarial):
                where report is a dictionary containing information specific to the detection defence;
                where is_adversarial is a boolean list of per-sample prediction whether the sample is adversarial
        """
        raise NotImplementedError

    def set_params(self, **kwargs) -> None:
        """
        Take in a dictionary of parameters and apply defence-specific checks before saving them as attributes.

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
