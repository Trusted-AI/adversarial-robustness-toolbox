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
