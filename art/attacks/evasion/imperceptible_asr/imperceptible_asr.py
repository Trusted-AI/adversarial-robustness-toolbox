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
This module implements the adversarial and imperceptible attack on automatic speech recognition systems of Qin et al.
(2019). It generates an adversarial audio example.

| Paper link: http://proceedings.mlr.press/v97/qin19a.html
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np

from art.attacks.attack import EvasionAttack
from art.estimators.speech_recognition.speech_recognizer import SpeechRecognizerMixin
from art.estimators.tensorflow import TensorFlowV2Estimator

logger = logging.getLogger(__name__)


class ImperceptibleAsr(EvasionAttack):
    """
    Implementation of the imperceptible attack against a speech recognition model.

    | Paper link: http://proceedings.mlr.press/v97/qin19a.html
    """

    attack_params = EvasionAttack.attack_params + []

    _estimator_requirements = (TensorFlowV2Estimator, SpeechRecognizerMixin)

    def __init__(
        self,
        estimator: "TensorFlowV2Estimator",
    ) -> None:
        """
        Create an instance of the :class:`.ImperceptibleAsr`.

        :param estimator: A trained classifier.
        """
        # Super initialization
        super().__init__(estimator=estimator)
        self._check_params()

    def generate(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate adversarial examples and return them as an array. This method should be overridden by all concrete
        evasion attack implementations.

        :param x: An array with the original inputs to be attacked.
        :param y: Correct labels or target labels for `x`, depending if the attack is targeted or not. This parameter
            is only used by some of the attacks.
        :return: An array holding the adversarial examples.
        """
        pass

    def _check_params(self) -> None:
        """
        Apply attack-specific checks.
        """
        pass
