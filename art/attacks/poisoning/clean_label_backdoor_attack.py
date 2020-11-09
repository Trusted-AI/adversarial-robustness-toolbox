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
This module implements Backdoor Attacks to poison data used in ML models.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np

from typing import Optional, Tuple, TYPE_CHECKING, Union

from art.attacks import PoisoningAttackBlackBox
from art.attacks.evasion import ProjectedGradientDescent
from art.attacks.poisoning import PoisoningAttackBackdoor
from art.estimators.classification.classifier import ClassifierLossGradients

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_LOSS_GRADIENTS_TYPE


logger = logging.getLogger(__name__)


class PoisoningAttackCleanLabelBackdoor(PoisoningAttackBlackBox):
    """
    Implementation of Clean-Label Backdoor Attacks introduced in Gu, et. al. 2017

    Applies a number of backdoor perturbation functions and switches label to target label

    | Paper link: https://arxiv.org/abs/1708.06733
    """

    attack_params = PoisoningAttackBlackBox.attack_params + ["backdoor", "proxy_classifier", "norm", "eps", "eps_step",
                                                             "max_iter", "num_random_init"]
    _estimator_requirements = ()

    def __init__(self,
                 backdoor: PoisoningAttackBackdoor,
                 proxy_classifier: "CLASSIFIER_LOSS_GRADIENTS_TYPE",
                 norm: Union[int, float, str] = np.inf,
                 eps: float = 0.3,
                 eps_step: float = 0.1,
                 max_iter: int = 100,
                 num_random_init: int = 0,
                 ) -> None:
        """
        Creates a new Clean Label Backdoor poisoning attack

        :param backdoor: the backdoor chosen for this attack
        :param proxy_classifier: the classifier for this attack ideally it solves the same or similar classification
                                 task as the original classifier
        :param norm: The norm of the adversarial perturbation supporting "inf", np.inf, 1 or 2.
        :param eps: Maximum perturbation that the attacker can introduce.
        :param eps_step: Attack step size (input variation) at each iteration.
        :param max_iter: The maximum number of iterations.
        :param num_random_init: Number of random initialisations within the epsilon ball. For num_random_init=0 starting
                                at the original input.
        """
        super().__init__()
        self.backdoor = backdoor
        self.proxy_classifier = proxy_classifier
        self.attack = ProjectedGradientDescent(proxy_classifier,
                                               norm=norm,
                                               eps=eps,
                                               eps_step=eps_step,
                                               max_iter=max_iter,
                                               targeted=False,
                                               num_random_init=num_random_init)
        self._check_params()

    def poison(
        self, x: np.ndarray, y: Optional[np.ndarray] = None, broadcast=False, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calls perturbation function on input x and returns the perturbed input and poison labels for the data.

        :param x: An array with the points that initialize attack points.
        :param y: The target labels for the attack.
        :param broadcast: whether or not to broadcast single target label
        :return: An tuple holding the `(poisoning_examples, poisoning_labels)`.
        """
        # perform an evasion attack on proxy classifier
        estimated_labels = self.proxy_classifier.predict(x) if y is None else y
        perturbed_input = self.attack.generate(x, estimated_labels)

        # add the backdoor trigger from the backdoor attack
        return self.backdoor.poison(perturbed_input, estimated_labels)

    def _check_params(self) -> None:
        if not isinstance(self.backdoor, PoisoningAttackBackdoor):
            raise ValueError("Backdoor must be of type PoisoningAttackBackdoor")
        if not isinstance(self.proxy_classifier, ClassifierLossGradients):
            raise ValueError("Proxy classifier should have loss gradients")
        if not isinstance(self.attack, ProjectedGradientDescent):
            raise ValueError("There was an issue creating the PGD attack")
