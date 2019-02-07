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

import logging

import abc
import sys

logger = logging.getLogger(__name__)


# Ensure compatibility with Python 2 and 3 when using ABCMeta
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str('ABC'), (), {})


class ExpectationOverTransformations:
    """
    Implementation of Expectation Over Transformations applied to classifier predictions and gradients, as introduced
    in Athalye et al. (2017). Paper link: https://arxiv.org/pdf/1707.07397.pdf
    """

    def __init__(self, sample_size, transformation):
        """
        Create a NewtonFool attack instance.

        :param sample_size: Number of transformations to sample
        :type sample_size: `int`
        :param transformation: An iterator over transformations.
        :type transformation: :class:`.Classifier`
        """
        self.sample_size = sample_size
        self.transformation = transformation

    def predict(self, classifier, x, logits=False, batch_size=128):
        """
        Perform prediction of the given classifier for a batch of inputs, taking an expectation over transformations.

        :param classifier: A trained model.
        :type classifier: :class:`.Classifier`
        :param x: Test set.
        :type x: `np.ndarray`
        :param logits: `True` if the prediction should be done at the logits layer.
        :type logits: `bool`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :return: Array of predictions of shape `(nb_inputs, self.nb_classes)`.
        :rtype: `np.ndarray`
        """
        logger.info('Apply Expectation over Transformations.')
        prediction = classifier.predict(next(self.transformation())(x), logits, batch_size)
        for _ in range(self.sample_size-1):
            prediction += classifier.predict(next(self.transformation())(x), logits, batch_size)
        return prediction/self.sample_size

    def loss_gradient(self, classifier, x, y):
        """
        Compute the gradient of the given classifier's loss function w.r.t. `x`, taking an expectation
        over transformations.

        :param classifier: A trained model.
        :type classifier: :class:`.Classifier`
        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param y: Correct labels, one-vs-rest encoding.
        :type y: `np.ndarray`
        :return: Array of gradients of the same shape as `x`.
        :rtype: `np.ndarray`
        """
        logger.info('Apply Expectation over Transformations.')
        loss_gradient = classifier.loss_gradient(next(self.transformation())(x), y)
        for _ in range(self.sample_size-1):
            loss_gradient += classifier.loss_gradient(next(self.transformation())(x), y)
        return loss_gradient/self.sample_size

    def class_gradient(self, classifier, x, label=None, logits=False):
        """
        Compute per-class derivatives of the given classifier w.r.t. `x`, taking an expectation over transformations.

        :param classifier: A trained model.
        :type classifier: :class:`.Classifier`
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
        logger.info('Apply Expectation over Transformations.')
        class_gradient = classifier.class_gradient(next(self.transformation())(x), label, logits)
        for _ in range(self.sample_size-1):
            class_gradient += classifier.class_gradient(next(self.transformation())(x), label, logits)
        return class_gradient/self.sample_size


class Attack(ABC):
    """
    Abstract base class for all attack classes.
    """
    attack_params = ['classifier', 'expectation']

    def __init__(self, classifier, expectation=None):
        """
        :param classifier: A trained model.
        :type classifier: :class:`.Classifier`
        :param expectation: An expectation over transformations to be applied when computing
                            classifier gradients and predictions.
        :type expectation: :class:`.ExpectationOverTransformations`
        """
        self.classifier = classifier
        self.expectation = expectation

    def _predict(self, x, logits=False, batch_size=128):
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
        if self.expectation is None:
            return self.classifier.predict(x, logits, batch_size)
        else:
            return self.expectation.predict(self.classifier, x, logits, batch_size)

    def _loss_gradient(self, x, y):
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param y: Correct labels, one-vs-rest encoding.
        :type y: `np.ndarray`
        :return: Array of gradients of the same shape as `x`.
        :rtype: `np.ndarray`
        """
        if self.expectation is None:
            return self.classifier.loss_gradient(x, y)
        else:
            return self.expectation.loss_gradient(self.classifier, x, y)

    def _class_gradient(self, x, label=None, logits=False):
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
        if self.expectation is None:
            return self.classifier.class_gradient(x, label, logits)
        else:
            return self.expectation.class_gradient(self.classifier, x, label, logits)

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
