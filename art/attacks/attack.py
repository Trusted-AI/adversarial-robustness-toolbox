from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import abc
import sys

logger = logging.getLogger(__name__)


# Ensure compatibility with Python 2 and 3 when using ABCMeta
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str('ABC'), (), {})


class Attack(ABC):
    """
    Abstract base class for all attack classes.
    """
    attack_params = ['classifier']

    def __init__(self, classifier):
        """
        :param classifier: A trained model.
        :type classifier: :class:`.Classifier`
        """
        self.classifier = classifier

    def generate(self, x, **kwargs):
        """
        Generate adversarial examples and return them as an array. This method should be overridden by all concrete
        attack implementations.

        :param x: An array with the original inputs to be attacked.
        :type x: `np.ndarray`
        :param kwargs: Attack-specific parameters used by child classes.
        :type kwargs: `dict`
        :return: An array holding the adversarial examples.
        :rtype: `np.ndarray`
        """
        raise NotImplementedError

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
