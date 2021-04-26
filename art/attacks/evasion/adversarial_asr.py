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
This module implements the audio adversarial attack on automatic speech recognition systems of Carlini and Wagner
(2018). It generates an adversarial audio example.

| Paper link: https://arxiv.org/abs/1801.01944
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import TYPE_CHECKING

from art.attacks.attack import EvasionAttack
from art.attacks.evasion.imperceptible_asr.imperceptible_asr import ImperceptibleASR

if TYPE_CHECKING:
    from art.utils import SPEECH_RECOGNIZER_TYPE

logger = logging.getLogger(__name__)


class CarliniWagnerASR(ImperceptibleASR):
    """
    Implementation of the Carlini and Wagner audio adversarial attack against a speech recognition model.

    | Paper link: https://arxiv.org/abs/1801.01944
    """

    attack_params = EvasionAttack.attack_params + [
        "eps",
        "learning_rate",
        "max_iter",
        "batch_size",
        "decrease_factor_eps",
        "num_iter_decrease_eps",
    ]

    def __init__(
        self,
        estimator: "SPEECH_RECOGNIZER_TYPE",
        eps: float = 2000.0,
        learning_rate: float = 100.0,
        max_iter: int = 1000,
        decrease_factor_eps: float = 0.8,
        num_iter_decrease_eps: int = 10,
        batch_size: int = 16,
    ):
        """
        Create an instance of the :class:`.CarliniWagnerASR`.

        :param estimator: A trained speech recognition estimator.
        :param eps: Initial max norm bound for adversarial perturbation.
        :param learning_rate: Learning rate of attack.
        :param max_iter: Number of iterations.
        :param decrease_factor_eps: Decrease factor for epsilon (Paper default: 0.8).
        :param num_iter_decrease_eps: Iterations after which to decrease epsilon if attack succeeds (Paper default: 10).
        :param batch_size: Batch size.
        """
        # pylint: disable=W0231

        # re-implement init such that inherited methods work
        EvasionAttack.__init__(self, estimator=estimator)  # pylint: disable=W0233
        self.masker = None  # type: ignore
        self.eps = eps
        self.learning_rate_1 = learning_rate
        self.max_iter_1 = max_iter
        self.max_iter_2 = 0
        self._targeted = True
        self.decrease_factor_eps = decrease_factor_eps
        self.num_iter_decrease_eps = num_iter_decrease_eps
        self.batch_size = batch_size

        # set remaining stage 2 params to some random values
        self.alpha = 0.1
        self.learning_rate_2 = 0.1
        self.loss_theta_min = 0.0
        self.increase_factor_alpha: float = 1.0
        self.num_iter_increase_alpha: int = 1
        self.decrease_factor_alpha: float = 1.0
        self.num_iter_decrease_alpha: int = 1

        self._check_params()
