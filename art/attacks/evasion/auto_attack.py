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
This module implements the `AutoAttack` attack.

| Paper link: https://arxiv.org/abs/2003.01690
"""
import logging
from typing import List, Optional, Union

import numpy as np

from art.config import ART_NUMPY_DTYPE
from art.attacks.attack import EvasionAttack
from art.attacks.evasion.auto_projected_gradient_descent import AutoProjectedGradientDescent
from art.attacks.evasion.deepfool import DeepFool
from art.attacks.evasion.square_attack import SquareAttack
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin, ClassifierGradients
from art.utils import get_labels_np_array, check_and_transform_label_format

logger = logging.getLogger(__name__)


class AutoAttack(EvasionAttack):
    attack_params = EvasionAttack.attack_params + [
        "norm",
        "eps",
        "eps_step",
        "attacks",
        "batch_size",
        "estimator_orig",
    ]

    _estimator_requirements = (BaseEstimator, ClassifierMixin)

    def __init__(
        self,
        estimator: ClassifierGradients,
        norm: Union[int, float] = np.inf,
        eps: float = 0.3,
        eps_step: float = 0.1,
        attacks: Optional[List[EvasionAttack]] = None,
        batch_size: int = 32,
        estimator_orig: Optional[BaseEstimator] = None,
    ):
        """
        Create a :class:`.ProjectedGradientDescent` instance.

        :param estimator: An trained estimator.
        :param norm: The norm of the adversarial perturbation. Possible values: np.inf, 1 or 2.
        :param eps: Maximum perturbation that the attacker can introduce.
        :param eps_step: Attack step size (input variation) at each iteration.
        :param attacks: The list of `art.attacks.EvasionAttack` attacks to be used for AutoAttack. If it is `None` the
                        original AutoAttack (PGD, APGD-ce, APGD-dlr, FAB, Square) will be used.
        :param batch_size: Size of the batch on which adversarial samples are generated.
        :param estimator_orig: Original estimator to be attacked by adversarial examples.
        """
        super().__init__(estimator=estimator)

        if estimator_orig is None:
            estimator_orig = estimator

        if attacks is None:
            attacks = list()
            attacks.append(
                AutoProjectedGradientDescent(
                    estimator=estimator,
                    norm=norm,
                    eps=eps,
                    eps_step=eps_step,
                    max_iter=100,
                    targeted=False,
                    nb_random_init=5,
                    batch_size=batch_size,
                    loss_type="cross_entropy",
                )
            )
            attacks.append(
                AutoProjectedGradientDescent(
                    estimator=estimator,
                    norm=norm,
                    eps=eps,
                    eps_step=eps_step,
                    max_iter=100,
                    targeted=False,
                    nb_random_init=5,
                    batch_size=batch_size,
                    loss_type="difference_logits_ratio",
                )
            )
            attacks.append(
                DeepFool(classifier=estimator, max_iter=100, epsilon=1e-6, nb_grads=3, batch_size=batch_size)
            )
            attacks.append(
                SquareAttack(estimator=estimator, norm=norm, max_iter=5000, eps=eps, p_init=0.8, nb_restarts=5)
            )

        self.norm = norm
        self.eps = eps
        self.eps_step = eps_step
        self.attacks = attacks
        self.batch_size = batch_size
        self.estimator_orig = estimator_orig
        self._check_params()

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :return: An array holding the adversarial examples.
        """
        x_adv = x.astype(ART_NUMPY_DTYPE)
        y = check_and_transform_label_format(y, self.estimator.nb_classes)

        if y is None:
            y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))

        # Determine correctly predicted samples
        y_pred = self.estimator_orig.predict(x_adv)
        sample_is_robust = np.argmax(y_pred, axis=1) == np.argmax(y, axis=1)

        for attack in self.attacks:
            if np.sum(sample_is_robust) == 0:
                break

            x_robust = x_adv[sample_is_robust]
            y_robust = y[sample_is_robust]

            # Generate adversarial examples with untargeted attack
            x_robust_adv = attack.generate(x=x_robust, y=y_robust)
            y_pred_robust_adv = self.estimator_orig.predict(x_robust_adv)

            norm_is_smaller_eps = (
                np.linalg.norm((x_robust_adv - x_robust).reshape((x_robust_adv.shape[0], -1)), axis=1, ord=self.norm)
                <= self.eps
            )

            sample_is_not_robust = np.logical_and(
                np.argmax(y_pred_robust_adv, axis=1) != np.argmax(y_robust, axis=1), norm_is_smaller_eps
            )

            x_robust[sample_is_not_robust] = x_robust_adv[sample_is_not_robust]
            x_adv[sample_is_robust] = x_robust

            sample_is_robust[sample_is_robust] = np.invert(sample_is_not_robust)

        return x_adv

    def _check_params(self) -> None:
        if self.norm not in [1, 2, np.inf]:
            raise ValueError("The argument norm has to be either 1, 2, or np.inf.")

        if not isinstance(self.eps, (int, float)) or self.eps <= 0.0:
            raise ValueError("The argument eps has to be either of type int or float and larger than zero.")

        if not isinstance(self.eps_step, (int, float)) or self.eps_step <= 0.0:
            raise ValueError("The argument eps_step has to be either of type int or float and larger than zero.")

        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("The argument batch_size has to be of type int and larger than zero.")
