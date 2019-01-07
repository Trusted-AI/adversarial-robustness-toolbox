from __future__ import absolute_import, division, print_function, unicode_literals

import abc
import sys

from art.attacks.expectation_over_transformations import ExpectationOverTransformations


# Ensure compatibility with Python 2 and 3 when using ABCMeta
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str('ABC'), (), {})


class Attack(ABC):
    """
    Abstract base class for all attack classes.
    """
    attack_params = ['classifier', 'expectation_over_transformations']

    def __init__(self, classifier, expectation_over_transformations=None):
        """
        :param classifier: A trained model.
        :type classifier: :class:`Classifier`
        :param expectation_over_transformations: An expectation over transformations to be applied when computing 
                                                 classifier gradients.
        :type expectation_over_transformations: :class:`ExpectationOverTransformations`
        """
        self.classifier = classifier
        self.expectation_over_transformations = expectation_over_transformations

    def predict(self, x, logits=False, batch_size=128):
        """
        Perform prediction for a batch of inputs.

        :param x: Test set.
        :type x: `np.ndarray`
        :param logits: `True` if the prediction should be done at the logits layer.
        :type logits: `bool`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :return: Array of predictions of shape `(nb_inputs, self.nb_classes)`.
        :rtype: `np.ndarray`
        """
        if self.expectation_over_transformations is None:
            return self.classifier.predict(x,logits,batch_size)
        else:
            return self.expectation_over_transformations.loss_gradient(self.classifier,x,logits,batch_size)

    def loss_gradient(self, x, y):
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param y: Correct labels, one-vs-rest encoding.
        :type y: `np.ndarray`
        :return: Array of gradients of the same shape as `x`.
        :rtype: `np.ndarray`
        """
        if self.expectation_over_transformations is None:
            return self.classifier.loss_gradient(x,y)
        else:
            return self.expectation_over_transformations.loss_gradient(self.classifier,x,y)

    def class_gradient(self, x, label=None, logits=False):
        """
        Compute per-class derivatives w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param label: Index of a specific per-class derivative. If an integer is provided, the gradient of that class
                      output is computed for all samples. If multiple values as provided, the first dimension should
                      match the batch size of `x`, and each value will be used as target for its corresponding sample in
                      `x`. If `None`, then gradients for all classes will be computed for each sample.
        :type label: `int` or `list`
        :param logits: `True` if the prediction should be done at the logits layer.
        :type logits: `bool`
        :return: Array of gradients of input features w.r.t. each class in the form
                 `(batch_size, nb_classes, input_shape)` when computing for all classes, otherwise shape becomes
                 `(batch_size, 1, input_shape)` when `label` parameter is specified.
        :rtype: `np.ndarray`
        """
        if self.expectation_over_transformations is None:
            return self.classifier.class_gradient(x,label,logits)
        else:
            return self.expectation_over_transformations.class_gradient(self.classifier,x,label,logits)

    def generate(self, x, **kwargs):
        """
        Generate adversarial examples and return them as an array. This method should be overridden by all concrete
        attack implementations.

        :param x: An array with the original inputs to be attacked.
        :type x: `np.ndarray`
        :param kwargs: Attack-specific parameters used by child classes.
        :type kwargs: `dict`
        :return: An array holding the adversarial examples.
        :rtype: `np.ndarray`
        """
        raise NotImplementedError

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and apply attack-specific checks before saving them as attributes.

        :param kwargs: a dictionary of attack-specific parameters
        :type kwargs: `dict`
        :return: `True` when parsing was successful
        """
        for key, value in kwargs.items():
            if key in self.attack_params:
                setattr(self, key, value)
        return True
