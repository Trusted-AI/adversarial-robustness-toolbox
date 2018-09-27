from __future__ import absolute_import, division, print_function, unicode_literals

import abc
import sys

# Ensure compatibility with Python 2 and 3 when using ABCMeta
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str('ABC'), (), {})


class NetworkManipulator(ABC):
    """
    Abstract base class for defences performing model hardening by manipulating network structure.
    """
    params = []

    @abc.abstractmethod
    def __call__(self, classifier):
        """
        Perform data preprocessing and return preprocessed data as tuple.

        :param classifier: Classifier to be manipulated.
        :type classifier: `Classifier`
        :return: A manipulated model derived from the given model. While the given model will be unchanged, its
                 component can be used in the manipulated model, and hence the manipulated model will be updated if the
                 given model is updated.
        """
        raise NotImplementedError

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and apply checks before saving them as attributes.

        :return: `True` when parsing was successful
        """
        for key, value in kwargs.items():
            if key in self.params:
                setattr(self, key, value)
        return True
