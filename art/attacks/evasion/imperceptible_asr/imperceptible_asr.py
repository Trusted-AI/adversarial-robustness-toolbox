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

    attack_params = EvasionAttack.attack_params + [
        "eps",
        "learning_rate_1",
        "max_iter_1",
    ]

    _estimator_requirements = (TensorFlowV2Estimator, SpeechRecognizerMixin)

    def __init__(
        self,
        estimator: "TensorFlowV2Estimator",
        eps: float = 2000,
        learning_rate_1: float = 100,
        max_iter_1: int = 1000,
    ) -> None:
        """
        Create an instance of the :class:`.ImperceptibleAsr`.

        :param estimator: A trained classifier.
        :param eps: Initial max norm bound for adversarial perturbation.
        :param learning_rate_1: Learning rate for stage 1 of attack.
        :param max_iter_1: Number of iterations for stage 1 of attack.
        """
        # Super initialization
        super().__init__(estimator=estimator)
        self.eps = eps
        self.learning_rate_1 = learning_rate_1
        self.max_iter_1 = max_iter_1
        self._targeted = True
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

    def _create_adversarial(self, x, y) -> np.ndarray:
        """
        Create adversarial example with small perturbation that successfully deceives the estimator.

        The method implements the part of the paper by Qin et al. (2019) that is referred to as the first stage of the
        attack. The authors basically follow Carlini and Wagner (2018).

        | Paper link: https://arxiv.org/abs/1801.01944.

        :param x: An array with the original inputs to be attacked.
        :param y: Target values of shape (batch_size,). Each sample in `y` is a string and it may possess different
            lengths. A possible example of `y` could be: `y = np.array(['SIXTY ONE', 'HELLO'])`.
        :return: An array with the adversarial outputs.
        """
        batch_size = x.shape[0]

        epsilon = [self.eps] * batch_size
        x_adversarial = [None] * batch_size

        x_perturbed = x.copy()
        perturbation = np.zeros_like(x_perturbed)

        for i in range(self.max_iter_1):
            # perform FGSM step for x
            gradients = self.estimator.loss_gradient(x_perturbed, y, batch_mode=True)
            x_perturbed = x_perturbed - self.learning_rate_1 * np.array([np.sign(g) for g in gradients], dtype=object)

            # clip perturbation
            perturbation = x_perturbed - x
            perturbation = np.array([np.clip(p, -e, e) for p, e in zip(perturbation, epsilon)], dtype=object)

            # re-apply clipped perturbation to x
            x_perturbed = x + perturbation

            if i % 10 == 0:
                prediction = self.estimator.predict(x_perturbed, batch_size=batch_size)
                for j in range(batch_size):
                    # validate adversarial target, i.e. f(x_perturbed)=y
                    if prediction[j] == y[j].upper():
                        # decrease max norm bound epsilon
                        perturbation_norm = np.max(np.abs(perturbation[j]))
                        if epsilon[j] > perturbation_norm:
                            epsilon[j] = perturbation_norm
                        epsilon[j] *= 0.8
                        # save current best adversarial example
                        x_adversarial[j] = x_perturbed[j]
                logger.info("Current iteration %s, epsilon %s", i, epsilon)

        # return perturbed x if no adversarial example found
        for j in range(batch_size):
            if x_adversarial[j] is None:
                logger.critical("Adversarial attack stage 1 for x_%s was not successful", j)
                x_adversarial[j] = x_perturbed[j]

        return np.array(x_adversarial, dtype=object)

    def _check_params(self) -> None:
        """
        Apply attack-specific checks.
        """
        pass
