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
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np

from art.attacks import ProjectedGradientDescent

logger = logging.getLogger(__name__)


class BasicIterativeMethod(ProjectedGradientDescent):
    """
    The Basic Iterative Method is the iterative version of FGM and FGSM.
    Paper link: https://arxiv.org/abs/1607.02533
    """
    attack_params = ProjectedGradientDescent.attack_params
    attack_params = [ap for ap in attack_params if ap not in ['norm', 'num_random_init']]

    def __init__(self, classifier, eps=.3, eps_step=0.1, max_iter=100, targeted=False, batch_size=1):
        """
        Create a :class:`.ProjectedGradientDescent` instance.

        :param classifier: A trained model.
        :type classifier: :class:`.Classifier`
        :param norm: Order of the norm. Possible values: np.inf, 1 or 2.
        :type norm: `int`
        :param eps: Maximum perturbation that the attacker can introduce.
        :type eps: `float`
        :param eps_step: Attack step size (input variation) at each iteration.
        :type eps_step: `float`
        :param targeted: Should the attack target one specific class
        :type targeted: `bool`
        :param num_random_init: Number of random initialisations within the epsilon ball. For random_init=0 starting at
            the original input.
        :type num_random_init: `int`
        :param batch_size: Batch size
        :type batch_size: `int`
        """
        super(BasicIterativeMethod, self).__init__(classifier, norm=np.inf, eps=eps, eps_step=eps_step,
                                                       max_iter=max_iter, targeted=targeted,
                                                       num_random_init=0, batch_size=batch_size)
