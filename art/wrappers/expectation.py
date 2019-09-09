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
"""
This module implements the Expectation Over Transformation applied to classifier predictions and gradients.

| Paper link: https://arxiv.org/abs/1707.07397
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

from art.wrappers.wrapper import ClassifierWrapper
from art.classifiers.classifier import Classifier, ClassifierNeuralNetwork, ClassifierGradients

logger = logging.getLogger(__name__)


class ExpectationOverTransformations(ClassifierWrapper, ClassifierGradients, ClassifierNeuralNetwork, Classifier):
    """
    Implementation of Expectation Over Transformations applied to classifier predictions and gradients, as introduced
    in Athalye et al. (2017).

    | Paper link: https://arxiv.org/abs/1707.07397
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
        self._predict = self.classifier.predict

    def predict(self, x, batch_size=128, **kwargs):
        """
        Perform prediction of the given classifier for a batch of inputs, taking an expectation over transformations.

        :param x: Test set.
        :type x: `np.ndarray`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        :rtype: `np.ndarray`
        """
        logger.info('Applying expectation over transformations.')
        prediction = self._predict(next(self.transformation())(x), batch_size)
        for _ in range(self.sample_size - 1):
            prediction += self._predict(next(self.transformation())(x), batch_size)
        return prediction / self.sample_size

    def fit(self, x, y, batch_size=128, nb_epochs=20, **kwargs):
        """
        Fit the classifier using the training data `(x, y)`.

        :param x: Features in array of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2)
        :type x: `np.ndarray`
        :param y: Target values (class labels in classification) in array of shape (nb_samples, nb_classes) in
                  One Hot Encoding format.
        :type y: `np.ndarray`
        :param kwargs: Dictionary of framework-specific arguments.
        :type kwargs: `dict`
        :return: `None`
        """
        raise NotImplementedError

    def loss_gradient(self, x, y, **kwargs):
        """
        Compute the gradient of the given classifier's loss function w.r.t. `x`, taking an expectation
        over transformations.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param y: Correct labels, one-hot encoded.
        :type y: `np.ndarray`
        :return: Array of gradients of the same shape as `x`.
        :rtype: `np.ndarray`
        """
        logger.info('Applying expectation over transformations.')
        loss_gradient = self.classifier.loss_gradient(next(self.transformation())(x), y)
        for _ in range(self.sample_size - 1):
            loss_gradient += self.classifier.loss_gradient(next(self.transformation())(x), y)
        return loss_gradient / self.sample_size

    def class_gradient(self, x, label=None, **kwargs):
        """
        Compute per-class derivatives of the given classifier w.r.t. `x`, taking an expectation over transformations.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param label: Index of a specific per-class derivative. If an integer is provided, the gradient of that class
                      output is computed for all samples. If multiple values as provided, the first dimension should
                      match the batch size of `x`, and each value will be used as target for its corresponding sample in
                      `x`. If `None`, then gradients for all classes will be computed for each sample.
        :type label: `int` or `list`
        :return: Array of gradients of input features w.r.t. each class in the form
                 `(batch_size, nb_classes, input_shape)` when computing for all classes, otherwise shape becomes
                 `(batch_size, 1, input_shape)` when `label` parameter is specified.
        :rtype: `np.ndarray`
        """
        logger.info('Apply Expectation over Transformations.')
        class_gradient = self.classifier.class_gradient(next(self.transformation())(x), label)

        for _ in range(self.sample_size - 1):
            class_gradient += self.classifier.class_gradient(next(self.transformation())(x), label)

        return class_gradient / self.sample_size

    @property
    def layer_names(self):
        """
        Return the hidden layers in the model, if applicable.

        :return: The hidden layers in the model, input and output layers excluded.
        :rtype: `list`

        .. warning:: `layer_names` tries to infer the internal structure of the model.
                     This feature comes with no guarantees on the correctness of the result.
                     The intended order of the layers tries to match their order in the model, but this is not
                     guaranteed either.
        """
        raise NotImplementedError

    def get_activations(self, x, layer, batch_size):
        """
        Return the output of the specified layer for input `x`. `layer` is specified by layer index (between 0 and
        `nb_layers - 1`) or by name. The number of layers can be determined by counting the results returned by
        calling `layer_names`.

        :param x: Input for computing the activations.
        :type x: `np.ndarray`
        :param layer: Layer for computing the activations
        :type layer: `int` or `str`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :return: The output of `layer`, where the first dimension is the batch size corresponding to `x`.
        :rtype: `np.ndarray`
        """
        raise NotImplementedError

    def set_learning_phase(self, train):
        """
        Set the learning phase for the backend framework.

        :param train: `True` if the learning phase is training, `False` if learning phase is not training.
        :type train: `bool`
        """
        raise NotImplementedError

    def save(self, filename, path=None):
        """
        Save a model to file specific to the backend framework.

        :param filename: Name of the file where to save the model.
        :type filename: `str`
        :param path: Path of the directory where to save the model. If no path is specified, the model will be stored in
                     the default data location of ART at `DATA_PATH`.
        :type path: `str`
        :return: None
        """
        raise NotImplementedError
