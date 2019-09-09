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
Provides black-box gradient estimation using NES.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np
from scipy.stats import entropy

from art.wrappers.wrapper import ClassifierWrapper
from art.classifiers.classifier import Classifier, ClassifierGradients
from art.utils import clip_and_round

logger = logging.getLogger(__name__)


class QueryEfficientBBGradientEstimation(ClassifierWrapper, ClassifierGradients, Classifier):
    """
    Implementation of Query-Efficient Black-box Adversarial Examples. The attack approximates the gradient by
    maximizing the loss function over samples drawn from random Gaussian noise around the input.

    | Paper link: https://arxiv.org/abs/1712.07113
    """
    attack_params = ['num_basis', 'sigma', 'round_samples']

    def __init__(self, classifier, num_basis, sigma, round_samples=0):
        """
        :param classifier: An instance of a `Classifier` whose loss_gradient is being approximated
        :type classifier: `Classifier`
        :param num_basis:  The number of samples to draw to approximate the gradient
        :type num_basis: `int`
        :param sigma: Scaling on the Gaussian noise N(0,1)
        :type sigma: `float`
        :param round_samples: The resolution of the input domain to round the data to, e.g., 1.0, or 1/255. Set to 0 to
                              disable.
        :type round_samples: `float`
        """
        super(QueryEfficientBBGradientEstimation, self).__init__(classifier)
        # self.predict refers to predict of classifier
        # pylint: disable=E0203
        self._predict = self.classifier.predict
        self.set_params(num_basis=num_basis, sigma=sigma, round_samples=round_samples)

    def predict(self, x, **kwargs):
        """
        Perform prediction of the classifier for input `x`.

        :param x: Features in array of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2)
        :type x: `np.ndarray`
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        :rtype: `np.ndarray`
        """
        return self._wrap_predict(x, **kwargs)

    def fit(self, x, y, **kwargs):
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

    def _generate_samples(self, x, epsilon_map):
        """
        Generate samples around the current image.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param epsilon_map: Samples drawn from search space
        :type epsilon_map: `np.ndarray`
        :return: Two arrays of new input samples to approximate gradient
        :rtype: `list(np.ndarray)`
        """
        minus = clip_and_round(np.repeat(x, self.num_basis, axis=0) - epsilon_map, self.clip_values, self.round_samples)
        plus = clip_and_round(np.repeat(x, self.num_basis, axis=0) + epsilon_map, self.clip_values, self.round_samples)
        return minus, plus

    def class_gradient(self, x, label=None, **kwargs):
        """
        Compute per-class derivatives w.r.t. `x`.

        :param x: Input with shape as expected by the classifier's model.
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
        raise NotImplementedError

    def loss_gradient(self, x, y, **kwargs):
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param y: Correct labels, one-vs-rest encoding.
        :type y: `np.ndarray`
        :return: Array of gradients of the same shape as `x`.
        :rtype: `np.ndarray`
        """
        epsilon_map = self.sigma * np.random.normal(size=([self.num_basis] + list(self.input_shape)))
        grads = []
        for i in range(len(x)):
            minus, plus = self._generate_samples(x[i:i + 1], epsilon_map)

            # Vectorized; small tests weren't faster
            # ent_vec = np.vectorize(lambda p: entropy(y[i], p), signature='(n)->()')
            # new_y_minus = ent_vec(self.predict(minus))
            # new_y_plus = ent_vec(self.predict(plus))
            # Vanilla
            new_y_minus = np.array([entropy(y[i], p) for p in self.predict(minus)])
            new_y_plus = np.array([entropy(y[i], p) for p in self.predict(plus)])
            query_efficient_grad = 2 * np.mean(np.multiply(
                epsilon_map.reshape(self.num_basis, -1),
                (new_y_plus - new_y_minus).reshape(self.num_basis, -1) /
                (2 * self.sigma)).reshape([-1] + list(self.input_shape)), axis=0)
            grads.append(query_efficient_grad)
        grads = self._apply_preprocessing_normalization_gradient(np.array(grads))
        return grads

    def _wrap_predict(self, x, batch_size=128):
        """
        Perform prediction for a batch of inputs. Rounds results first.

        :param x: Test set.
        :type x: `np.ndarray`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        :rtype: `np.ndarray`
        """
        return self._predict(clip_and_round(x, self.clip_values, self.round_samples), **{'batch_size': batch_size})

    def nb_classes(self):
        """
        Return the number of output classes.

        :return: Number of classes in the data.
        :rtype: `int`
        """
        # pylint: disable=W0212
        return self.classifier.nb_classes()

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
