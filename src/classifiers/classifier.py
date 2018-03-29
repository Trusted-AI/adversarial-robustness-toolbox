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
        """

        :param clip_values:
        """
        self._clip_values = clip_values

    def predict(self, inputs):
        """

        :param inputs:
        :return:
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fit(self, inputs, outputs, batch_size=128, nb_epochs=10):
        """

        :param inputs:
        :param outputs:
        :param batch_size:
        :param nb_epochs:
        :return:
        """
        raise NotImplementedError

    def nb_classes(self):
        return self._nb_classes

    @abc.abstractmethod
    def class_gradient(self, input):
        """

        :param input:
        :return: Array [n_classes, shape_input]
        :rtype: `np.ndarray`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def loss_gradient(self, input, label):
        """

        :param input:
        :param label:
        :return:
        """
        raise NotImplementedError
