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
This module implements Backdoor Attacks to poison data used in ML models.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np

from art.attacks.attack import PoisoningAttackBlackBox


logger = logging.getLogger(__name__)


class PoisoningAttackBackdoor(PoisoningAttackBlackBox):
    """
    Implementation of backdoor attacks introduced in Gu, et. al. 2017

    Applies a number of backdoor perturbation functions and switches label to target label

    | Paper link: https://arxiv.org/abs/1708.06733
    """

    attack_params = PoisoningAttackBlackBox.attack_params + ["perturbation"]

    def __init__(self, perturbation, **kwargs):
        """
        Initialize a backdoor poisoning attack

        :param perturbation: a single perturbation function or list of perturbation functions that modify input
        :type perturbation: a `function` that takes an np.array and returns an np.array or
                            a list of functions of this type
        :param kwargs: Extra optional keyword arguments
        """

        super().__init__()

        self.perturbation = perturbation
        self.set_params(**kwargs)

    def poison(self, x, y=None, **kwargs):
        """
        Iteratively finds optimal attack points starting at values at x

        :param x: An array with the points that initialize attack points.
        :type x: `np.ndarray`
        :param y: The target labels for
        :return: An tuple holding the (poisoning examples, poisoning labels).
        :rtype: `(np.ndarray, np.ndarray)`
        """

        if y is None:
            raise ValueError("Target labels `y` need to be provided for a targeted attack.")
        else:
            y_attack = np.copy(y)

        num_poison = len(x)

        if num_poison == 0:
            raise ValueError("Must input at least one poison point")

        poisoned = np.copy(x)

        if callable(self.perturbation):
            return self.perturbation(poisoned), y_attack

        for perturb in self.perturbation:
            poisoned = perturb(poisoned)

        return poisoned, y_attack

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and apply attack-specific checks before saving them as attributes.

        :param kwargs: a dictionary of attack-specific parameters
        :type kwargs: `dict`
        :return: `True` when parsing was successful
        """
        super().set_params(**kwargs)
        if not (callable(self.perturbation) or all((callable(perturb) for perturb in self.perturbation))):
            raise ValueError("Perturbation must be a function or a list of functions")
