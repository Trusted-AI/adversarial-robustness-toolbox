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
    Abstract base class for defenses performing model hardening by preprocessing data.
    """
    params = []

    def __init__(self):
        """
        Create a preprocessing object
        """
        pass

    @abc.abstractmethod
    def __call__(self, x_val, y_val=None):
        """
        Perform data preprocessing and return preprocessed data as tuple.

        :param x_val: (np.ndarray) Dataset to be preprocessed
        :param y_val: (np.ndarray) Labels to be preprocessed
        :return: Preprocessed data
        """
        pass

    @abc.abstractmethod
    def fit(self, x_val, y_val=None, **kwargs):
        """
        Fit the parameters of the data preprocessor if it has any.

        :param x_val: (np.ndarray) Training set to fit the preprocessor
        :param y_val: (np.ndarray) Labels for the training set
        :param kwargs: (dict) Other parameters
        :return: None
        """
        self.is_fitted = True

    def predict(self, x_val, y_val=None):
        """
        Perform data preprocessing and return preprocessed data as tuple.

        :param x_val: (np.ndarray) Dataset to be preprocessed
        :param y_val: (np.ndarray) Labels to be preprocessed
        :return: Preprocessed data
        """
        return self.__call__(x_val, y_val)

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and apply checks before saving them as attributes.
        :return: True when parsing was successful
        """
        for key, value in kwargs.items():
            if key in self.params:
                setattr(self, key, value)
        return True
