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
This module implements ``Wasserstein Adversarial Examples via Projected Sinkhorn Iterations`` as evasion attack.

| Paper link: https://arxiv.org/abs/1902.07906
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np

from art.config import ART_NUMPY_DTYPE
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassGradientsMixin
from art.attacks.attack import EvasionAttack
from art.utils import compute_success, get_labels_np_array
from art.utils import check_and_transform_label_format

logger = logging.getLogger(__name__)


class Wasserstein(EvasionAttack):
    """
    Implements ``Wasserstein Adversarial Examples via Projected Sinkhorn Iterations`` as evasion attack.

    | Paper link: https://arxiv.org/abs/1902.07906
    """

    attack_params = EvasionAttack.attack_params + [
        "targeted",
        "regularization",
        "p",
        "alpha",
        "norm",
        "ball",
        "epsilon",
        "epsilon_factor",
        "max_iter",
        "batch_size",
    ]

    _estimator_requirements = (BaseEstimator, ClassGradientsMixin)

    def __init__(
        self,
        estimator,
        targeted=False,
        regularization=3000,
        p=2,
        alpha=0.1,
        norm=np.Inf,
        ball='wasserstein',
        epsilon=0.001,
        epsilon_factor=1.17,
        max_iter=400,
        batch_size=1,
    ):
        """
        Create a Wasserstein attack instance.

        :param estimator: A trained estimator.
        :type estimator: :class:`.BaseEstimator`
        """
        super(Wasserstein, self).__init__(estimator=estimator)

        kwargs = {
            "targeted": targeted,
            "regularization": regularization,
            "p": p,
            "alpha": alpha,
            "norm": norm,
            "ball": ball,
            "epsilon": epsilon,
            "epsilon_factor": epsilon_factor,
            "max_iter": max_iter,
            "batch_size": batch_size,
        }
        Wasserstein.set_params(self, **kwargs)


    def generate(self, x, y=None, **kwargs):
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs to be attacked.
        :type x: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,). If `self.targeted` is true, then `y` represents the target labels. If `self.targeted`
                  is true, then `y_val` represents the target labels. Otherwise, the targets are the original class
                  labels.
        :return: An array holding the adversarial examples.
        :rtype: `np.ndarray`
        """
        y = check_and_transform_label_format(y, self.estimator.nb_classes)
        x_adv = x.astype(ART_NUMPY_DTYPE)

        if self.estimator.clip_values is not None:
            clip_min, clip_max = self.estimator.clip_values
        else:
            clip_min, clip_max = np.amin(x), np.amax(x)

        # Assert that, if attack is targeted, y_val is provided:
        if self.targeted and y is None:
            raise ValueError("Target labels `y` need to be provided for a targeted attack.")

        # No labels provided, use model prediction as correct class
        if y is None:
            y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))

        # Compute perturbation with implicit batching
        nb_batches = int(np.ceil(x_adv.shape[0] / float(self.batch_size)))
        for batch_id in range(nb_batches):
            logger.debug("Processing batch %i out of %i", batch_id, nb_batches)

            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            x_batch = x_adv[batch_index_1:batch_index_2]
            y_batch = y[batch_index_1:batch_index_2]


        return x_adv

    def set_params(self, **kwargs):
        """Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.

        :param confidence: Confidence of adversarial examples: a higher value produces examples that are farther away,
               from the original input, but classified with higher confidence as the target class.
        :type confidence: `float`
        :param targeted: Should the attack target one specific class
        :type targeted: `bool`
        :param learning_rate: The learning rate for the attack algorithm. Smaller values produce better results but are
               slower to converge.
        :type learning_rate: `float`
        :param binary_search_steps: number of times to adjust constant with binary search (positive value)
        :type binary_search_steps: `int`
        :param max_iter: The maximum number of iterations.
        :type max_iter: `int`
        :param initial_const: (optional float, positive) The initial trade-off constant c to use to tune the relative
               importance of distance and confidence. If binary_search_steps is large,
               the initial constant is not important. The default value 1e-4 is suggested in Carlini and Wagner (2016).
        :type initial_const: `float`
        :param max_halving: Maximum number of halving steps in the line search optimization.
        :type max_halving: `int`
        :param max_doubling: Maximum number of doubling steps in the line search optimization.
        :type max_doubling: `int`
        :param batch_size: Internal size of batches on which adversarial samples are generated.
        :type batch_size: `int`
        """
        # Save attack-specific parameters
        super(Wasserstein, self).set_params(**kwargs)


        return True


