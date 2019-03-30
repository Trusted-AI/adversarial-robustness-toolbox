from __future__ import absolute_import, division, print_function, unicode_literals

import abc
import sys

# Ensure compatibility with Python 2 and 3 when using ABCMeta
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str('ABC'), (), {})


class Preprocessor(ABC):
    """
    Abstract base class for defences performing model hardening by preprocessing data.
    """
    params = []

    def __init__(self):
        """
        Create a preprocessing object
        """
        self._is_fitted = False

    @property
    def is_fitted(self):
        """
        Return the state of the preprocessing object.

        :return: `True` if the preprocessing model has been fitted (if this applies).
        :rtype: `bool`
        """
        return self._is_fitted

    @property
    @abc.abstractmethod
    def apply_fit(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def apply_predict(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, x, y=None):
        """
        Perform data preprocessing and return preprocessed data as tuple.

        :param x: Dataset to be preprocessed.
        :type x: `np.ndarray`
        :param y: Labels to be preprocessed.
        :type y: `np.ndarray`
        :return: Preprocessed data
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fit(self, x, y=None, **kwargs):
        """
        Fit the parameters of the data preprocessor if it has any.

        :param x: Training set to fit the preprocessor.
        :type x: `np.ndarray`
        :param y: Labels for the training set.
        :type y: `np.ndarray`
        :param kwargs: Other parameters.
        :type kwargs: `dict`
        :return: None
        """
        raise NotImplementedError

    @abc.abstractmethod
    def estimate_gradient(self, grad):
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
