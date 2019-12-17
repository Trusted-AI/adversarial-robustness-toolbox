# MIT License
#
# Copyright (C) IBM Corporation 2018
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
This module implements the abstract base classes for all attacks.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import abc

from art.classifiers.classifier import Classifier

logger = logging.getLogger(__name__)


class Attack(abc.ABC):
    """
    Abstract base class for all attack abstract base classes.
    """
    attack_params = list()

    def __init__(self, classifier):
        """
        :param classifier: A trained classifier.
        :type classifier: :class:`.Classifier`
        """
        if not isinstance(classifier, Classifier):
            raise (TypeError('For `' + self.__class__.__name__ + '` classifier must be an instance of '
                             '`art.classifiers.classifier.Classifier`, the provided classifier is instance of ' + str(
                                 classifier.__class__.__bases__) + '.'))
        self.classifier = classifier

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and apply attack-specific checks before saving them as attributes.

        :param kwargs: a dictionary of attack-specific parameters
        :type kwargs: `dict`
        :return: `True` when parsing was successful
        """
        for key, value in kwargs.items():
            if key in self.attack_params:
                setattr(self, key, value)
        return True


class EvasionAttack(Attack):
    """
    Abstract base class for evasion attack classes.
    """

    def __init__(self, classifier):
        """
        :param classifier: A trained classifier.
        :type classifier: :class:`.Classifier`
        """
        super().__init__(classifier)

    @abc.abstractmethod
    def generate(self, x, y=None, **kwargs):
        """
        Generate adversarial examples and return them as an array. This method should be overridden by all concrete
        evasion attack implementations.

        :param x: An array with the original inputs to be attacked.
        :type x: `np.ndarray`
        :param y: Correct labels or target labels for `x`, depending if the attack is targeted
               or not. This parameter is only used by some of the attacks.
        :type y: `np.ndarray`
        :return: An array holding the adversarial examples.
        :rtype: `np.ndarray`
        """
        raise NotImplementedError


class PoisoningAttack(Attack):
    """
    Abstract base class for poisoning attack classes.
    """

    def __init__(self, classifier):
        """
        :param classifier: A trained classifier.
        :type classifier: :class:`.Classifier`
        """
        super().__init__(classifier)

    @abc.abstractmethod
    def generate(self, x, y=None, **kwargs):
        """
        Generate poisoning examples and return them as an array. This method should be overridden by all concrete
        poisoning attack implementations.

        :param x: An array with the original inputs to be attacked.
        :type x: `np.ndarray`
        :param y: Correct labels or target labels for `x`, depending if the attack is targeted
               or not. This parameter is only used by some of the attacks.
        :type y: `np.ndarray`
        :return: An array holding the poisoning examples.
        :rtype: `np.ndarray`
        """
        raise NotImplementedError


class ExtractionAttack(Attack):
    """
    Abstract base class for extraction attack classes.
    """

    def __init__(self, classifier):
        """
        :param classifier: A trained classifier targeted for extraction.
        :type classifier: :class:`.Classifier`
        """
        super().__init__(classifier)

    @abc.abstractmethod
    def extract(self, x, y=None, **kwargs):
        """
        Extract models and return them as an ART classifier. This method should be overridden by all concrete extraction
        attack implementations.

        :param x: An array with the original inputs to be attacked.
        :type x: `np.ndarray`
        :param y: Correct labels or target labels for `x`, depending if the attack is targeted
               or not. This parameter is only used by some of the attacks.
        :type y: `np.ndarray`
        :return: ART classifier of the extracted model.
        :rtype: :class:`.Classifier`
        """
        raise NotImplementedError
