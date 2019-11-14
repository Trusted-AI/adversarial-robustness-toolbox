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
This module implements Randomized Smoothing applied to classifier predictions.

| Paper link: https://arxiv.org/abs/1902.02918
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np

from art.wrappers.wrapper import ClassifierWrapper
from art.classifiers.classifier import Classifier, ClassifierGradients

logger = logging.getLogger(__name__)


class RandomizedSmoothing(ClassifierWrapper, ClassifierGradients, Classifier):
    """
    Implementation of Randomized Smoothing applied to classifier predictions and gradients, as introduced
    in Cohen et al. (2019).

    | Paper link: https://arxiv.org/abs/1902.02918
    """

    def __init__(self, classifier, sample_size, scale=0.1, alpha=0.001):
        """
        Create a randomized smoothing wrapper.

        :param classifier: The Classifier we want to wrap the functionality for the purpose of smoothing.
        :type classifier: :class:`.Classifier`
        :param sample_size: Number of samples for smoothing
        :type sample_size: `int`
        :param scale: Standard deviation of Gaussian noise added.
        :type scale: `float`
        :param alpha: The failure probability of smoothing
        :type alpha: `float`
        """
        super(RandomizedSmoothing, self).__init__(classifier)
        self.sample_size = sample_size
        self.scale = scale
        self.alpha = alpha

    # pylint: disable=W0221
    def predict(self, x, batch_size=128, is_abstain=True, **kwargs):
        """
        Perform prediction of the given classifier for a batch of inputs, taking an expectation over transformations.

        :param x: Test set.
        :type x: `np.ndarray`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :param is_abstain: True if function will abstain from prediction and return 0s
        :type is_abstain: `boolean`
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        :rtype: `np.ndarray`
        """
        from scipy.stats import binom_test

        logger.info('Applying randomized smoothing.')
        n_abstained = 0
        prediction = []
        for x_i in x:

            # get class counts
            counts_pred = self._prediction_counts(x_i, batch_size=batch_size)
            top = counts_pred.argsort()[::-1]
            count1 = np.max(counts_pred)
            count2 = counts_pred[top[1]]

            # predict or abstain
            smooth_prediction = np.zeros(counts_pred.shape)
            if (not is_abstain) or (binom_test(count1, count1 + count2, p=0.5) <= self.alpha):
                smooth_prediction[np.argmax(counts_pred)] = 1
            elif is_abstain:
                n_abstained += 1

            prediction.append(smooth_prediction)
        if n_abstained > 0:
            print('%s prediction(s) abstained.' % n_abstained)
        return np.array(prediction)

    def loss_gradient(self, x, y, **kwargs):
        """
        Compute the gradient of the given classifier's loss function w.r.t. `x` of the original classifier.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param y: Correct labels, one-hot encoded.
        :type y: `np.ndarray`
        :return: Array of gradients of the same shape as `x`.
        :rtype: `np.ndarray`
        """
        logger.info('Applying randomized smoothing.')
        return self.classifier.loss_gradient(x, y)

    def class_gradient(self, x, label=None, **kwargs):
        """
        Compute per-class derivatives of the given classifier w.r.t. `x` of original classifier.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param label: Index of a specific per-class derivative. If an integer is provided, the gradient of that class
                      output is computed for all samples. If multiple values as provided, the first dimension should
                      match the batch size of `x`, and each value will be used as target for its corresponding sample in
                      `x`. If `None`, then gradients for all classes will be computed for each sample.
        :type label: `list`
        :return: Array of gradients of input features w.r.t. each class in the form
                 `(batch_size, nb_classes, input_shape)` when computing for all classes, otherwise shape becomes
                 `(batch_size, 1, input_shape)` when `label` parameter is specified.
        :rtype: `np.ndarray`
        """
        logger.info('Apply randomized smoothing.')
        return self.classifier.class_gradient(x, label)

    def certify(self, x, n):
        """
        Computes certifiable radius around input `x` and returns radius `r` and prediction.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param n: Number of samples for estimate certifiable radius
        :type n: `int`
        :return: Tuple of length 2 of the selected class and certified radius
        :rtype: `tuple`
        """
        from scipy.stats import norm

        prediction = []
        radius = []
        for x_i in x:

            # get sample prediction for classification
            counts_pred = self._prediction_counts(x_i)
            class_select = np.argmax(counts_pred)

            # get sample prediction for certification
            counts_est = self._prediction_counts(x_i, n=n)
            count_class = counts_est[class_select]

            prob_class = self._lower_confidence_bound(count_class, n)

            if prob_class < 0.5:
                prediction.append(-1)
                radius.append(0.0)
            else:
                prediction.append(class_select)
                radius.append(self.scale * norm.ppf(prob_class))

        return np.array(prediction), np.array(radius)

    def _noisy_samples(self, x, n=None):
        """
        Adds Gaussian noise to `x` to generate samples. Optionally augments `y` similarly.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param n: Number of noisy samples to create
        :type n: `int`
        :return: Array of samples of the same shape as `x`.
        :rtype: `np.ndarray`
        """
        # set default value to sample_size
        if n is None:
            n = self.sample_size

        # augment x
        x = np.expand_dims(x, axis=0)
        x = np.repeat(x, n, axis=0)
        x = x + np.random.normal(scale=self.scale, size=x.shape)

        return x

    def _prediction_counts(self, x, n=None, batch_size=128):
        """
        Makes predictions and then converts probability distribution to counts

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :return: Array of counts with length equal to number of columns of `x`.
        :rtype: `np.ndarray`
        """
        # sample and predict
        x_new = self._noisy_samples(x, n=n)
        predictions = self.classifier.predict(x_new, batch_size)

        # convert to binary predictions
        idx = np.argmax(predictions, axis=-1)
        pred = np.zeros(predictions.shape)
        pred[np.arange(pred.shape[0]), idx] = 1

        # get class counts
        counts = np.sum(pred, axis=0)

        return counts

    def _lower_confidence_bound(self, n_class_samples, n_total_samples):
        """
        Uses Clopper-Pearson method to return a (1-alpha) lower confidence bound on bernoulli proportion

        :param n_class_samples: Number of samples of a specific class.
        :type n_class_samples: `int`
        :param n_total_samples: Number of samples for certification.
        :type n_total_samples: `int`
        :return: Lower bound on the binomial proportion w.p. (1-alpha) over samples
        :rtype: `float`
        """
        from statsmodels.stats.proportion import proportion_confint

        return proportion_confint(n_class_samples, n_total_samples, alpha=2 * self.alpha, method="beta")[0]

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
