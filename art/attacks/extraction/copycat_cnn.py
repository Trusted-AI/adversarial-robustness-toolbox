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
This module implements the copycat cnn attack `CopycatCNN`.

| Paper link: https://arxiv.org/abs/1806.05476
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np

from art import NUMPY_DTYPE
from art.attacks.attack import Attack


logger = logging.getLogger(__name__)


class CopycatCNN(Attack):
    """
    Implementation of the copycat cnn attack from Jacson et al. (2018).

    | Paper link: https://arxiv.org/abs/1806.05476
    """
    attack_params = Attack.attack_params + ['batch_size', 'nb_epochs']

    def __init__(self, classifier, batch_size=1, nb_epochs=10):
        """
        Create a copycat cnn attack instance.

        :param classifier: A victim classifier.
        :type classifier: :class:`.Classifier`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :param nb_epochs: Number of epochs to use for training.
        :type nb_epochs: `int`
        """
        super(CopycatCNN, self).__init__(classifier=classifier)

        params = {'batch_size': batch_size,
                  'nb_epochs': nb_epochs}
        self.set_params(**params)

    def generate(self, x, y=None, **kwargs):
        """
        Generate a thieved classifier.

        :param x: An array with the original inputs to be attacked.
        :type x: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,). If `self.targeted` is true, then `y` represents the target labels.
        :type y: `np.ndarray` or `None`
        :param x_adv_init: Initial array to act as initial adversarial examples. Same shape as `x`.
        :type x_adv_init: `np.ndarray`
        :return: An array holding the adversarial examples.
        :rtype: `np.ndarray`
        """

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.

        :param targeted: Should the attack target one specific class.
        :type targeted: `bool`
        :param delta: Initial step size for the orthogonal step.
        :type delta: `float`
        :param epsilon: Initial step size for the step towards the target.
        :type epsilon: `float`
        :param step_adapt: Factor by which the step sizes are multiplied or divided, must be in the range (0, 1).
        :type step_adapt: `float`
        :param max_iter: Maximum number of iterations.
        :type max_iter: `int`
        :param num_trial: Maximum number of trials per iteration.
        :type num_trial: `int`
        :param sample_size: Number of samples per trial.
        :type sample_size: `int`
        :param init_size: Maximum number of trials for initial generation of adversarial examples.
        :type init_size: `int`
        """
        # Save attack-specific parameters
        super(BoundaryAttack, self).set_params(**kwargs)

        if not isinstance(self.max_iter, (int, np.int)) or self.max_iter < 0:
            raise ValueError("The number of iterations must be a non-negative integer.")

        if not isinstance(self.num_trial, (int, np.int)) or self.num_trial < 0:
            raise ValueError("The number of trials must be a non-negative integer.")

        if not isinstance(self.sample_size, (int, np.int)) or self.sample_size <= 0:
            raise ValueError("The number of samples must be a positive integer.")

        if not isinstance(self.init_size, (int, np.int)) or self.init_size <= 0:
            raise ValueError("The number of initial trials must be a positive integer.")

        if self.epsilon <= 0:
            raise ValueError("The initial step size for the step towards the target must be positive.")

        if self.delta <= 0:
            raise ValueError("The initial step size for the orthogonal step must be positive.")

        if self.step_adapt <= 0 or self.step_adapt >= 1:
            raise ValueError("The adaptation factor must be in the range (0, 1).")

        return True
