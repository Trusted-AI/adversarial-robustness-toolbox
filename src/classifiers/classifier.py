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
        Initialize a `Classifier` object.

        :param clip_values: (tuple) Tuple of the form `(min, max)` representing the minimum and maximum values allowed
        for features.
        """
        self._clip_values = clip_values

    def predict(self, inputs):
        """
        Perform prediction for a batch of inputs.

        :param inputs: (np.ndarray) Test set.
        :return: Array of predictions of shape `(nb_inputs, self.nb_classes)`.
        :rtype: `np.ndarray`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fit(self, inputs, outputs, batch_size=128, nb_epochs=20):
        """
        Fit the classifier on the training set `(inputs, outputs)`.

        :param inputs: (np.ndarray) Training data.
        :param outputs: (np.ndarray) Labels.
        :param batch_size: (optinal int, default 128) Size of batches.
        :param nb_epochs: (optinal int, default 20) Number of epochs.
        """
        raise NotImplementedError

    def nb_classes(self):
        """
        :return: Number of classes in the data.
        :rtype: int
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

        :param input: (np.ndarray) One sample input with shape as expected by the model.
        :return: Array of gradients of input features w.r.t. each class in the form `[self.nb_classes, input_shape]`
        :rtype: `np.ndarray`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def loss_gradient(self, input, label):
        """
        Compute the gradient of the loss function w.r.t. `input`.

        :param input: (np.ndarray) One sample input with shape as expected by the model.
        :param label: Correct label.
        :return: Array of gradients of the same shape as `input`.
        :rtype: `np.ndarray`
        """
        raise NotImplementedError
