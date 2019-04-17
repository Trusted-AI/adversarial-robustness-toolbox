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

from art.wrappers.wrapper import ClassifierWrapper

logger = logging.getLogger(__name__)


class ExpectationOverTransformations(ClassifierWrapper):
    """
    Implementation of Expectation Over Transformations applied to classifier predictions and gradients, as introduced
    in Athalye et al. (2017). Paper link: https://arxiv.org/pdf/1707.07397.pdf
    """

    def __init__(self, classifier, sample_size, transformation):
        """
        Create an expectation over transformations wrapper.

        :param classifier: The Classifier we want to wrap the functionality for the purpose of an attack.
        :type classifier: :class:`.Classifier`
        :param sample_size: Number of transformations to sample
        :type sample_size: `int`
        :param transformation: An iterator over transformations.
        :type transformation: :class:`.Classifier`
        """
        super(ExpectationOverTransformations, self).__init__(classifier)
        self.sample_size = sample_size
        self.transformation = transformation

    def predict(self, x, logits=False, batch_size=128):
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
        prediction = self.classifier.predict(next(self.transformation())(x), logits, batch_size)
        for _ in range(self.sample_size-1):
            prediction += self.classifier.predict(next(self.transformation())(x), logits, batch_size)
        return prediction/self.sample_size

    def loss_gradient(self, x, y):
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
        loss_gradient = self.classifier.loss_gradient(next(self.transformation())(x), y)
        for _ in range(self.sample_size-1):
            loss_gradient += self.classifier.loss_gradient(next(self.transformation())(x), y)
        return loss_gradient/self.sample_size

    def class_gradient(self, x, label=None, logits=False):
        """
        Compute per-class derivatives of the given classifier w.r.t. `x`, taking an expectation over transformations.

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
        class_gradient = self.classifier.class_gradient(next(self.transformation())(x), label, logits)
        for _ in range(self.sample_size-1):
            class_gradient += self.classifier.class_gradient(next(self.transformation())(x), label, logits)
        return class_gradient/self.sample_size
