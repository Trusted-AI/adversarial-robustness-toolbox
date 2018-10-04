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
import logging
import sys

logger = logging.getLogger(__name__)


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

        :return: `True` when parsing was successful.
        :rtype: `bool`
        """
        for key, value in kwargs.items():
            if key in self.params:
                setattr(self, key, value)
        return True


class BinaryInputDetector(Detector):
    """
    Binary detector of adversarial samples coming from evasion attacks. The detector uses an architecture provided by
    the user and trains it on data labeled as clean (label 0) or adversarial (label 1).
    """
    def __init__(self, detector):
        """
        Create a `BinaryInputDetector` instance which performs binary classification on input data.

        :param detector: The detector architecture to be trained and applied for the binary classification.
        :type detector: `art.classifier.Classifier`
        """
        super(BinaryInputDetector, self).__init__()
        self._detector = detector

    def fit(self, x, y, **kwargs):
        """
        Fit the detector using training data.

        :param x: Training set to fit the detector.
        :type x: `np.ndarray`
        :param y: Labels for the training set.
        :type y: `np.ndarray`
        :param kwargs: Other parameters.
        :type kwargs: `dict`
        :return: None
        """
        self._detector.fit(x, y, **kwargs)
        self._is_fitted = True

    def __call__(self, x):
        """
        Perform detection of adversarial data and return prediction as tuple.

        :param x: Data sample on which to perform detection.
        :type x: `np.ndarray`
        :return: Per-sample prediction whether data is adversarial or not, where `0` means non-adversarial.
                 Return variable has the same `batch_size` (first dimension) as `x`.
        :rtype: `np.ndarray`
        """
        return self._detector.predict(x)


class BinaryActivationDetector(Detector):
    """
    Binary detector of adversarial samples coming from evasion attacks. The detector uses an architecture provided by
    the user and is trained on the values of the activations of a classifier at a given layer.
    """
    def __init__(self, classifier, detector, layer):
        """
        Create a `BinaryActivationDetector` instance which performs binary classification on activation information.
        The shape of the input of the detector has to match that of the output of the chosen layer.

        :param classifier: The classifier of which the activation information is to be used for detection.
        :type classifier: `art.classifier.Classifier`
        :param detector: The detector architecture to be trained and applied for the binary classification.
        :type detector: `art.classifier.Classifier`
        :param layer: Layer for computing the activations to use for training the detector.
        :type layer: `int` or `str`
        """
        super(BinaryActivationDetector, self).__init__()
        self._classifier = classifier
        self._detector = detector

        # Ensure that layer is well-defined:
        if type(layer) is str:
            if layer not in classifier.layer_names:
                raise ValueError('Layer name %s is not part of the graph.' % layer)
            self._layer_name = layer
        elif type(layer) is int:
            if layer < 0 or layer >= len(classifier.layer_names):
                raise ValueError('Layer index %d is outside of range (0 to %d included).'
                                 % (layer, len(classifier.layer_names) - 1))
            self._layer_name = classifier.layer_names[layer]
        else:
            raise TypeError('Layer must be of type `str` or `int`.')

        self._is_fitted = False

    def fit(self, x, y, **kwargs):
        """
        Fit the detector using training data.

        :param x: Training set to fit the detector.
        :type x: `np.ndarray`
        :param y: Labels for the training set.
        :type y: `np.ndarray`
        :param kwargs: Other parameters.
        :type kwargs: `dict`
        :return: None
        """
        x_activations = self._classifier.get_activations(x, self._layer_name)
        self._detector.fit(x_activations, y, **kwargs)
        self._is_fitted = True

    def __call__(self, x):
        """
        Perform detection of adversarial data and return prediction as tuple.

        :param x: Data sample on which to perform detection.
        :type x: `np.ndarray`
        :return: Per-sample prediction whether data is adversarial or not, where `0` means non-adversarial.
                 Return variable has the same `batch_size` (first dimension) as `x`.
        :rtype: `np.ndarray`
        """
        return self._detector.predict(self._classifier.get_activations(x, self._layer_name))
