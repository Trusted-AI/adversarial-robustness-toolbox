from __future__ import absolute_import, division, print_function, unicode_literals

import abc
import sys

import numpy as np
# Ensure compatibility with Python 2 and 3 when using ABCMeta
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str('ABC'), (), {})



class Feature(ABC):
    """
    Base class for features

    Objects instantiated for this class can be used to extract features for a given classifier
    and a set of inages
    """

    def __init__(self, classifier):
        """

        :param classifier: ART classifier
        """
        self.classifier = classifier

    @abc.abstractmethod
    def extract(self, x):
        """

        :param x: Sample input with shape as expected by the model.
        :return: extracted features
        :rtype: `np.ndarray`
        """
        return NotImplementedError


