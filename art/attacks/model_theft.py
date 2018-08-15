from __future__ import absolute_import, division, print_function, unicode_literals

import abc
import sys


# Ensure compatibility with Python 2 and 3 when using ABCMeta
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str('ABC'), (), {})


class ModelTheft(ABC):
    """
    Abstract base class for all model stealing attack classes.
    """
    attack_params = []

    def __init__(self):
        pass

    def steal(self, model, stolen_model, budget, **kwargs):
        """
        Generate adversarial examples and return them as an array. This method should be overridden by all concrete
        attack implementations.

        :param model: An original model to steal.
        :type model: `Classifier`
        :param model: An untrained model to update with stealing.
        :type model: `Classifier`
        :param kwargs: Attack-specific parameters used by child classes.
        :type kwargs: `dict`
        :return: A stolen models.
        :rtype: `Classifier`
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

