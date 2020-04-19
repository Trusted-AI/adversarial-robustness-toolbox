# MIT License
#
# Copyright (C) IBM Corporation 2020
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
This module implements model inversion attacks.

| Paper link: https://dl.acm.org/doi/10.1145/2810103.2813677
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np

from art.config import ART_NUMPY_DTYPE
from art.classifiers.classifier import ClassifierGradients
from art.attacks.attack import InferenceAttack
from art.utils import get_labels_np_array, check_and_transform_label_format


logger = logging.getLogger(__name__)


class MIFace(InferenceAttack):
    """
    Implementation of the MIFace algorithm from Fredrikson et al. (2015). While in that paper the attack is
    demonstrated specifically against face recognition models, it is applicable more broadly to classifiers with
    continuous features which expose class gradients.

    | Paper link: https://dl.acm.org/doi/10.1145/2810103.2813677
    """

    attack_params = InferenceAttack.attack_params + ["max_iter", "window_length", "threshold", "learning_rate",
                                                     "batch_size"]

    def __init__(self, classifier, max_iter=10000, window_length=100, threshold=0.99, learning_rate=0.1, batch_size=1):
        """
        Create an MIFace attack instance.

        :param classifier: Target classifier.
        :type classifier: :class:`.Classifier`
        :param max_iter: Maximum number of gradient descent iterations for the model inversion.
        :type max_iter: `int`
        :param window_length: Length of window for checking whether descent should be aborted.
        :type window_length: `int`
        :param threshold: Threshold for descent stopping criterion.
        :type threshold: `float`
        :param batch_size: Size of internal batches.
        :type batch_size: `int`
        """
        super(MIFace, self).__init__(classifier=classifier)

        params = {
            "max_iter": max_iter,
            "window_length": window_length,
            "threshold": threshold,
            "learning_rate": learning_rate,
            "batch_size": batch_size
        }
        self.set_params(**params)

    @classmethod
    def is_valid_classifier_type(cls, classifier):
        """
        Checks whether the classifier provided is a classifer which this class can perform an attack on
        :param classifier:
        :return:
        """
        return True if isinstance(classifier, ClassifierGradients) else False

    def infer(self, x, y=None, **kwargs):
        """
        Extract a thieved classifier.

        :param x: An array with the initial input to the victim classifier. If `None`, then initial input will be
                  initialized as zero array.
        :type x: `np.ndarray` or `None`
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :type y: `np.ndarray` or `None`
        :return: The inferred training samples.
        :rtype: `np.ndarray`
        """

        if x is None and y is None:
            return None

        y = check_and_transform_label_format(y, self.classifier.nb_classes())

        if x is None:
            x = np.zeros((len(y),) + self.classifier.input_shape)

        if y is None:
            y = get_labels_np_array(self.classifier.predict(x, batch_size=self.batch_size))

        x_infer = x.astype(ART_NUMPY_DTYPE)

        # Compute inversions with implicit batching
        for batch_id in range(int(np.ceil(x.shape[0] / float(self.batch_size)))):
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            batch = x_infer[batch_index_1:batch_index_2]
            batch_labels = y[batch_index_1:batch_index_2]

            active = np.array([True] * len(batch))
            window = np.inf * np.ones((len(batch), self.window_length))

            i = 0

            while i < self.max_iter and sum(active) > 0:
                grads = self.classifier.class_gradient(batch[active], np.argmax(batch_labels[active], axis=1))
                grads = np.reshape(grads, (grads.shape[0],) + grads.shape[2:])
                batch[active] = batch[active] + self.learning_rate * grads

                if hasattr(self.classifier, "clip_values") and self.classifier.clip_values is not None:
                    clip_min, clip_max = self.classifier.clip_values
                    batch[active] = np.clip(batch[active], clip_min, clip_max)

                cost = 1 - self.classifier.predict(batch)[np.arange(len(batch)), np.argmax(batch_labels, axis=1)]
                active = (cost <= self.threshold) + (cost >= np.max(window, axis=1))

                i_window = i % self.window_length
                window[::, i_window] = cost

                i = i + 1

            x_infer[batch_index_1:batch_index_2] = batch

        return x_infer
