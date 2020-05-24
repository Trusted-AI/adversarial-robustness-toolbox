# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
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
from typing import List, Optional, Union

import numpy as np
from scipy.stats import binom_test, norm
from statsmodels.stats.proportion import proportion_confint

from art.wrappers.wrapper import ClassifierWrapper
from art.estimators.classification.classifier import ClassifierGradients

logger = logging.getLogger(__name__)


class RandomizedSmoothing(ClassifierWrapper, ClassifierGradients):
    """
    Implementation of Randomized Smoothing applied to classifier predictions and gradients, as introduced
    in Cohen et al. (2019).

    | Paper link: https://arxiv.org/abs/1902.02918
    """

    def __init__(
        self,
        classifier: ClassifierGradients,
        sample_size: int,
        scale: float = 0.1,
        alpha: float = 0.001,
    ) -> None:
        """
        Create a randomized smoothing wrapper.

        :param classifier: The Classifier we want to wrap the functionality for the purpose of smoothing.
        :param sample_size: Number of samples for smoothing.
        :param scale: Standard deviation of Gaussian noise added.
        :param alpha: The failure probability of smoothing.
        """
        super(RandomizedSmoothing, self).__init__(classifier)
        self.sample_size = sample_size
        self.scale = scale
        self.alpha = alpha
        self._nb_classes = classifier.nb_classes

    # pylint: disable=W0221
    def predict(
        self, x: np.ndarray, batch_size: int = 128, is_abstain: bool = True, **kwargs
    ) -> np.ndarray:
        """
        Perform prediction of the given classifier for a batch of inputs, taking an expectation over transformations.

        :param x: Test set.
        :param batch_size: Size of batches.
        :param is_abstain: True if function will abstain from prediction and return 0s.
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        """
        logger.info("Applying randomized smoothing.")
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
            if (not is_abstain) or (
                binom_test(count1, count1 + count2, p=0.5) <= self.alpha
            ):
                smooth_prediction[np.argmax(counts_pred)] = 1
            elif is_abstain:
                n_abstained += 1

            prediction.append(smooth_prediction)
        if n_abstained > 0:
            print("%s prediction(s) abstained." % n_abstained)
        return np.array(prediction)

    def loss_gradient(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute the gradient of the given classifier's loss function w.r.t. `x` of the original classifier.

        :param x: Sample input with shape as expected by the model.
        :param y: Correct labels, one-hot encoded.
        :return: Array of gradients of the same shape as `x`.
        """
        logger.info("Applying randomized smoothing.")
        return self.classifier.loss_gradient(x, y)

    def class_gradient(
        self, x: np.ndarray, label: Union[int, List[int], None] = None, **kwargs
    ) -> np.ndarray:
        """
        Compute per-class derivatives of the given classifier w.r.t. `x` of original classifier.

        :param x: Sample input with shape as expected by the model.
        :param label: Index of a specific per-class derivative. If an integer is provided, the gradient of that class
                      output is computed for all samples. If multiple values as provided, the first dimension should
                      match the batch size of `x`, and each value will be used as target for its corresponding sample in
                      `x`. If `None`, then gradients for all classes will be computed for each sample.
        :return: Array of gradients of input features w.r.t. each class in the form
                 `(batch_size, nb_classes, input_shape)` when computing for all classes, otherwise shape becomes
                 `(batch_size, 1, input_shape)` when `label` parameter is specified.
        """
        logger.info("Apply randomized smoothing.")
        return self.classifier.class_gradient(x, label)

    def certify(self, x: np.ndarray, n: int) -> tuple:
        """
        Computes certifiable radius around input `x` and returns radius `r` and prediction.

        :param x: Sample input with shape as expected by the model.
        :param n: Number of samples for estimate certifiable radius.
        :return: Tuple of length 2 of the selected class and certified radius.
        """
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

    def _noisy_samples(self, x: np.ndarray, n: Optional[int] = None) -> np.ndarray:
        """
        Adds Gaussian noise to `x` to generate samples. Optionally augments `y` similarly.

        :param x: Sample input with shape as expected by the model.
        :param n: Number of noisy samples to create.
        :return: Array of samples of the same shape as `x`.
        """
        # set default value to sample_size
        if n is None:
            n = self.sample_size

        # augment x
        x = np.expand_dims(x, axis=0)
        x = np.repeat(x, n, axis=0)
        x = x + np.random.normal(scale=self.scale, size=x.shape)

        return x

    def _prediction_counts(
        self, x: np.ndarray, n: Optional[int] = None, batch_size: int = 128
    ) -> np.ndarray:
        """
        Makes predictions and then converts probability distribution to counts.

        :param x: Sample input with shape as expected by the model.
        :param batch_size: Size of batches.
        :return: Array of counts with length equal to number of columns of `x`.
        """
        # sample and predict
        x_new = self._noisy_samples(x, n=n)
        predictions = self.classifier.predict(x_new, batch_size=batch_size)

        # convert to binary predictions
        idx = np.argmax(predictions, axis=-1)
        pred = np.zeros(predictions.shape)
        pred[np.arange(pred.shape[0]), idx] = 1

        # get class counts
        counts = np.sum(pred, axis=0)

        return counts

    def _lower_confidence_bound(
        self, n_class_samples: int, n_total_samples: int
    ) -> float:
        """
        Uses Clopper-Pearson method to return a (1-alpha) lower confidence bound on bernoulli proportion

        :param n_class_samples: Number of samples of a specific class.
        :param n_total_samples: Number of samples for certification.
        :return: Lower bound on the binomial proportion w.p. (1-alpha) over samples.
        """
        return proportion_confint(
            n_class_samples, n_total_samples, alpha=2 * self.alpha, method="beta"
        )[0]

    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Fit the classifier using the training data `(x, y)`.

        :param x: Features in array of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).
        :param y: Target values (class labels in classification) in array of shape (nb_samples, nb_classes) in
                  One Hot Encoding format.
        :param kwargs: Dictionary of framework-specific arguments.
        """
        raise NotImplementedError

    def get_activations(
        self, x: np.ndarray, layer: Union[int, str], batch_size: int
    ) -> np.ndarray:
        """
        Return the output of the specified layer for input `x`. `layer` is specified by layer index (between 0 and
        `nb_layers - 1`) or by name. The number of layers can be determined by counting the results returned by
        calling `layer_names`.

        :param x: Input for computing the activations.
        :param layer: Layer for computing the activations.
        :param batch_size: Size of batches.
        :return: The output of `layer`, where the first dimension is the batch size corresponding to `x`.
        """
        raise NotImplementedError

    def set_learning_phase(self, train):
        """
        Set the learning phase for the backend framework.

        :param train: `True` if the learning phase is training, `False` if learning phase is not training.
        :type train: `bool`
        """
        raise NotImplementedError

    def save(self, filename: str, path: Optional[str] = None) -> None:
        """
        Save a model to file specific to the backend framework.

        :param filename: Name of the file where to save the model.
        :param path: Path of the directory where to save the model. If no path is specified, the model will be stored in
                     the default data location of ART at `ART_DATA_PATH`.
        """
        raise NotImplementedError
