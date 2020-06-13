# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
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
from typing import Optional

import numpy as np
from tqdm import trange

from art.config import ART_NUMPY_DTYPE
from art.estimators.classification.classifier import ClassGradientsMixin, Classifier
from art.estimators.estimator import BaseEstimator
from art.attacks import InferenceAttack
from art.utils import get_labels_np_array, check_and_transform_label_format


logger = logging.getLogger(__name__)


class MIFace(InferenceAttack):
    """
    Implementation of the MIFace algorithm from Fredrikson et al. (2015). While in that paper the attack is demonstrated
    specifically against face recognition models, it is applicable more broadly to classifiers with continuous features
    which expose class gradients.

    | Paper link: https://dl.acm.org/doi/10.1145/2810103.2813677
    """

    attack_params = InferenceAttack.attack_params + [
        "max_iter",
        "window_length",
        "threshold",
        "learning_rate",
        "batch_size",
    ]

    _estimator_requirements = (BaseEstimator, ClassGradientsMixin)

    def __init__(
        self,
        classifier: Classifier,
        max_iter: int = 10000,
        window_length: int = 100,
        threshold: float = 0.99,
        learning_rate: float = 0.1,
        batch_size: int = 1,
    ):
        """
        Create an MIFace attack instance.

        :param classifier: Target classifier.
        :param max_iter: Maximum number of gradient descent iterations for the model inversion.
        :param window_length: Length of window for checking whether descent should be aborted.
        :param threshold: Threshold for descent stopping criterion.
        :param batch_size: Size of internal batches.
        """
        super().__init__(estimator=classifier)

        self.max_iter = max_iter
        self.window_length = window_length
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self._check_params()

    def infer(self, x: Optional[np.ndarray], y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Extract a thieved classifier.

        :param x: An array with the initial input to the victim classifier. If `None`, then initial input will be
                  initialized as zero array.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :return: The inferred training samples.
        """
        if x is None and y is None:
            raise ValueError("Either `x` or `y` should be provided.")

        y = check_and_transform_label_format(y, self.estimator.nb_classes)
        if x is None:
            x = np.zeros((len(y),) + self.estimator.input_shape)

        if y is None:
            y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))

        x_infer = x.astype(ART_NUMPY_DTYPE)

        # Compute inversions with implicit batching
        for batch_id in trange(int(np.ceil(x.shape[0] / float(self.batch_size))), desc="Model inversion"):
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            batch = x_infer[batch_index_1:batch_index_2]
            batch_labels = y[batch_index_1:batch_index_2]

            active = np.array([True] * len(batch))
            window = np.inf * np.ones((len(batch), self.window_length))

            i = 0

            while i < self.max_iter and sum(active) > 0:
                grads = self.estimator.class_gradient(batch[active], np.argmax(batch_labels[active], axis=1))
                grads = np.reshape(grads, (grads.shape[0],) + grads.shape[2:])
                batch[active] = batch[active] + self.learning_rate * grads

                if self.estimator.clip_values is not None:
                    clip_min, clip_max = self.estimator.clip_values
                    batch[active] = np.clip(batch[active], clip_min, clip_max)

                cost = 1 - self.estimator.predict(batch)[np.arange(len(batch)), np.argmax(batch_labels, axis=1)]
                active = (cost <= self.threshold) + (cost >= np.max(window, axis=1))

                i_window = i % self.window_length
                window[::, i_window] = cost

                i = i + 1

            x_infer[batch_index_1:batch_index_2] = batch

        return x_infer
