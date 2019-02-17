from __future__ import absolute_import, division, print_function, unicode_literals

import abc
import sys

# Ensure compatibility with Python 2 and 3 when using ABCMeta
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str('ABC'), (), {})


class PoisonFilteringDefence(ABC):
    """
    Base class for all poison filtering defences.
    """
    defence_params = ['classifier']

    def __init__(self, classifier, x_train, y_train):
        """
        Create an :class:`.ActivationDefence` object with the provided classifier.

        :param classifier: model evaluated for poison
        :type classifier: :class:`.Classifier`
        :param x_train: dataset used to train the classifier.
        :type x_train: :class:`numpy.ndarray`
        :param y_train: labels used to train the classifier.
        :type y_train: :class:`numpy.ndarray`
        """
        self.classifier = classifier
        self.x_train = x_train
        self.y_train = y_train

    @abc.abstractmethod
    def detect_poison(self, **kwargs):
        """
        Detect poison.

        :param kwargs: Defence-specific parameters used by child classes.
        :type kwargs: `dict`
        :return: `(dict, list)` dictionary with report and list with items identified as poison
        """
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate_defence(self, is_clean, **kwargs):
        """
        Evaluate the defence given the labels specifying if the data is poisoned or not.

        :param is_clean: 1-D array where is_clean[i]=1 means x_train[i] is clean and is_clean[i]=0 that it's poison.
        :param kwargs: Defence-specific parameters used by child classes.
        :type kwargs: `dict`
        :return: JSON object with confusion matrix
        """
        raise NotImplementedError

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and apply attack-specific checks before saving them as attributes.

        :param kwargs: a dictionary of defence-specific parameters
        :type kwargs: `dict`
        :return: `True` when parsing was successful
        """
        for key, value in kwargs.items():
            if key in self.defence_params:
                setattr(self, key, value)
        return True

    def get_params(self):
        """
        Returns dictionary of parameters used to run defence.

        :return: `dict`
        """
        dictionary = {}
        for param in self.defence_params:
            dictionary.update({param: getattr(self, param)})
        return dictionary
