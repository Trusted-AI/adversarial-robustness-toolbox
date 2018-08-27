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
        Query a given model to replicate it.

        :param model: An original model to steal.
        :type model: `Classifier`
        :param stolen_model: An untrained model to update with stealing.
        :type stolen_model: `Classifier`
        :param budget: The number of queries that the attacker can use.
        :type budget: `int`
        :param kwargs: Attack-specific parameters used by child classes.
        :type kwargs: `dict`
        :return: A stolen model.
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
