from __future__ import absolute_import, division, print_function, unicode_literals

import abc
import sys


# Ensure compatibility with Python 2 and 3 when using ABCMeta
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {})


class Classifier(ABC):
    """
    Base class for all classifiers.
    """
    def __init__(self, clip_values):
        """
        Initialize a `Classifier` object.

        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :type clip_values: `tuple`
        """
        self._clip_values = clip_values

    def predict(self, inputs):
        """
        Perform prediction for a batch of inputs.

        :param inputs: Test set.
        :type inputs: `np.ndarray`
        :return: Array of predictions of shape `(nb_inputs, self.nb_classes)`.
        :rtype: `np.ndarray`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fit(self, inputs, outputs, batch_size=128, nb_epochs=20):
        """
        Fit the classifier on the training set `(inputs, outputs)`.

        :param inputs: Training data.
        :type inputs: `np.ndarray`
        :param outputs: Labels.
        :type outputs: `np.ndarray`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :param nb_epochs: Number of epochs to use for trainings.
        :type nb_epochs: `int`
        :return: `None`
        """
        raise NotImplementedError

    def nb_classes(self):
        """
        Return the number of output classes.

        :return: Number of classes in the data.
        :rtype: `int`
        """
        return self._nb_classes

    def clip_values(self):
        """
        :return: Tuple of the form `(min, max)` representing the minimum and maximum values allowed for features.
        :rtype: `tuple`
        """
        return self._clip_values

    @abc.abstractmethod
    def class_gradient(self, input):
        """
        Compute per-class derivatives w.r.t. `input`.

        :param input: One sample input with shape as expected by the model.
        :type input: `np.ndarray`
        :return: Array of gradients of input features w.r.t. each class in the form `(self.nb_classes, input_shape)`
        :rtype: `np.ndarray`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def loss_gradient(self, input, label):
        """
        Compute the gradient of the loss function w.r.t. `input`.

        :param input: One sample input with shape as expected by the model.
        :type input: `np.ndarray`
        :param label: Correct label.
        :type label: `int`
        :return: Array of gradients of the same shape as `input`.
        :rtype: `np.ndarray`
        """
        raise NotImplementedError
