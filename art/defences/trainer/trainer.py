# MIT License
#
# Copyright (C) IBM Corporation 2020
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


class Trainer(abc.ABC):
    """
    Abstract base class for training defences.
    """

    def __init__(self, classifier, **kwargs):
        """
        Create a adversarial training object
        """
        self._classifier = classifier

    @abc.abstractmethod
    def fit(self, x, y, **kwargs):  # lgtm [py/inheritance/incorrect-overridden-signature]
        """
        Train the model.

        :param x: Training data.
        :type x: `np.ndarray`
        :param y: Labels for the training data.
        :type y: `np.ndarray`
        :param kwargs: Other parameters.
        :type kwargs: `dict`
        :return: None
        """
        raise NotImplementedError

    def get_classifier(self):
        """
        Train the model.

        :param x: Training data.
        :type x: `np.ndarray`
        :param y: Labels for the training data.
        :type y: `np.ndarray`
        :param kwargs: Other parameters.
        :type kwargs: `dict`
        :return: None
        """
        return self._classifier
