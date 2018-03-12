from __future__ import absolute_import, division, print_function, unicode_literals

import abc
import sys


# Ensure compatibility with Python 2 and 3 when using ABCMeta
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {})


class Classifier(ABC):
    def __init__(self, clip_values):
        self._clip_values = clip_values

    def predict(self, inputs):
        raise NotImplementedError

    @abc.abstractmethod
    def fit(self):
        raise NotImplementedError

    def nb_classes(self):
        return self._nb_classes

    @abc.abstractmethod
    def gradients(self, inputs, labels):
        raise NotImplementedError
