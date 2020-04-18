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
This module implements the `Fast Adaptive Boundary` attack.

| Paper link: https://arxiv.org/abs/1907.02044
"""
import logging

import numpy as np

from art.config import ART_NUMPY_DTYPE
from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator
from art.utils import check_and_transform_label_format

logger = logging.getLogger(__name__)


class FastAdaptiveBoundary(EvasionAttack):

    attack_params = EvasionAttack.attack_params + [
        "norm",
        "eps",
        "max_iter",
        "nb_restarts",
        "alpha_max",
        "eta",
        "beta",
        "targeted",
        "batch_size",
    ]

    _estimator_requirements = (BaseEstimator,)

    def __init__(
        self,
        estimator,
        norm=np.inf,
        eps=0.3,
        max_iter=100,
        nb_restarts=1,
        alpha_max=0.1,
        eta=1.05,
        beta=0.9,
        targeted=False,
        batch_size=32,
    ):
        """
        Create a :class:`.ProjectedGradientDescent` instance.

        :param estimator: An trained estimator.
        :type estimator: :class:`.BaseEstimator`
        :param norm: The norm of the adversarial perturbation. Possible values: np.inf, 1 or 2.
        :type norm: `int`
        :param eps: Maximum perturbation that the attacker can introduce.
        :type eps: `float`
        :param max_iter: Maximum number of iterations.
        :type max_iter: `int`
        :param nb_restarts: Number of random restarts.
        :type nb_restarts: `int`
        :param alpha_max: Maximum biased gradient step scaling factor in range [0, 1].
        :type alpha_max: `float`
        :param eta: Overshoot parameter, eta >= 1
        :type eta: `float`
        :param beta: Backward step scaling factor in range [0, 1].
        :type beta: `float`
        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False)
        :type targeted: `bool`
        :param batch_size: Size of the batch on which adversarial samples are generated.
        :type batch_size: `int`
        """
        super().__init__(estimator=estimator)

        kwargs = {
            "norm": norm,
            "eps": eps,
            "max_iter": max_iter,
            "nb_restarts": nb_restarts,
            "alpha_max": alpha_max,
            "eta": eta,
            "beta": beta,
            "targeted": targeted,
            "batch_size": batch_size,
        }
        self.set_params(**kwargs)

    def generate(self, x, y=None, **kwargs):

        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :type x: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :type y: `np.ndarray`
        :return: An array holding the adversarial examples.
        :rtype: `np.ndarray`
        """
        x_adv = x.astype(ART_NUMPY_DTYPE)

        y = check_and_transform_label_format(y, self.estimator.nb_classes)

        return x_adv


    def set_params(self, **kwargs):
        super().set_params(**kwargs)
