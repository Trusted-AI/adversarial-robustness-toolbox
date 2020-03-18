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
This module implements the abstract base class for defences that transform a classifier into a more robust classifier.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import abc


class Transformer(abc.ABC):
    """
    Abstract base class for transformation defences.
    """

    params = list()

    def __init__(self, classifier):
        """
        Create a transformation object.

        :param classifier: A trained classifier.
        :type classifier: :class:`.Classifier`
        """
        self.classifier = classifier
        self._is_fitted = False

    @property
    def is_fitted(self):
        """
        Return the state of the transformation object.

        :return: `True` if the transformation model has been fitted (if this applies).
        :rtype: `bool`
        """
        return self._is_fitted

    def get_classifier(self):
        """
        Get the internal classifier.

        :return: The internal classifier.
        :rtype: :class:`.Classifier`
        """
        return self.classifier

    @abc.abstractmethod
    def __call__(self, x, transformed_classifier):
        """
        Perform the transformation defence and return a robuster classifier.

        :param x: Dataset for training the transformed classifier.
        :type x: `np.ndarray`
        :param transformed_classifier: A classifier to be transformed for increased robustness.
        :type transformed_classifier: :class:`.Classifier`
        :return: The transformed classifier.
        :rtype: :class:`.Classifier`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fit(self, x, y=None, **kwargs):
        """
        Fit the parameters of the transformer if it has any.

        :param x: Training set to fit the transformer.
        :type x: `np.ndarray`
        :param y: Labels for the training set.
        :type y: `np.ndarray`
        :param kwargs: Other parameters.
        :type kwargs: `dict`
        :return: None.
        """
        raise NotImplementedError

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and apply checks before saving them as attributes.

        :return: `True` when parsing was successful.
        """
        for key, value in kwargs.items():
            if key in self.params:
                setattr(self, key, value)
        return True
