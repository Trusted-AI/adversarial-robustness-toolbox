from __future__ import absolute_import, division, print_function, unicode_literals

import abc
import sys


# Ensure compatibility with Python 2 and 3 when using ABCMeta
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str('ABC'), (), {})


class Detector(ABC):
    """
    Base abstract class for all detection methods.
    """
    def __init__(self):
        """
        Create a detector.
        """
        self._is_fitted = False

    @property
    def is_fitted(self):
        """
        Return the state of the detector.

        :return: `True` if the detection model has been fitted (if this applies).
        :rtype: `bool`
        """
        return self._is_fitted

    @abc.abstractmethod
    def fit(self, x, y=None, **kwargs):
        """
        Fit the detector using training data (if this applies).

        :param x: Training set to fit the detector.
        :type x: `np.ndarray`
        :param y: Labels for the training set.
        :type y: `np.ndarray`
        :param kwargs: Other parameters.
        :type kwargs: `dict`
        :return: None
        """
        self._is_fitted = True

    @abc.abstractmethod
    def __call__(self, x):
        """
        Perform detection of adversarial data and return preprocessed data as tuple.

        :param x: Data sample on which to perform detection.
        :type x: `np.ndarray`
        :return: Per-sample prediction whether data is adversarial or not, where `0` means non-adversarial.
                 Return variable has the same `batch_size` (first dimension) as `x`.
        :rtype: `np.ndarray`
        """
        raise NotImplementedError

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and apply checks before saving them as attributes.
        :return: True when parsing was successful
        """
        for key, value in kwargs.items():
            if key in self.params:
                setattr(self, key, value)
        return True
