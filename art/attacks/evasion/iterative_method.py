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
This module implements the Basic Iterative Method attack `BasicIterativeMethod` as the iterative version of FGM and
FGSM. This is a white-box attack.

| Paper link: https://arxiv.org/abs/1607.02533
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Union, TYPE_CHECKING

import numpy as np

from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_LOSS_GRADIENTS_TYPE

logger = logging.getLogger(__name__)


class BasicIterativeMethod(ProjectedGradientDescent):
    """
    The Basic Iterative Method is the iterative version of FGM and FGSM.

    | Paper link: https://arxiv.org/abs/1607.02533
    """

    attack_params = ProjectedGradientDescent.attack_params

    def __init__(
        self,
        estimator: "CLASSIFIER_LOSS_GRADIENTS_TYPE",
        eps: Union[int, float, np.ndarray] = 0.3,
        eps_step: Union[int, float, np.ndarray] = 0.1,
        max_iter: int = 100,
        targeted: bool = False,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> None:
        """
        Create a :class:`.ProjectedGradientDescent` instance.

        :param estimator: A trained classifier.
        :param eps: Maximum perturbation that the attacker can introduce.
        :param eps_step: Attack step size (input variation) at each iteration.
        :param max_iter: The maximum number of iterations.
        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False).
        :param batch_size: Size of the batch on which adversarial samples are generated.
        :param verbose: Show progress bars.
        """
        super().__init__(
            estimator=estimator,
            norm=np.inf,
            eps=eps,
            eps_step=eps_step,
            max_iter=max_iter,
            targeted=targeted,
            num_random_init=0,
            batch_size=batch_size,
            verbose=verbose,
        )
