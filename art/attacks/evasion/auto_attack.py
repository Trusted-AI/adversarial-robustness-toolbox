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

import numpy as np

from art.config import ART_NUMPY_DTYPE
from art.attacks.attack import EvasionAttack
from art.attacks.evasion.auto_projected_gradient_descent import AutoProjectedGradientDescent
from art.attacks.evasion.deepfool import DeepFool
from art.attacks.evasion.square_attack import SquareAttack
from art.estimators.estimator import BaseEstimator
from art.utils import get_labels_np_array, check_and_transform_label_format

logger = logging.getLogger(__name__)


class AutoAttack(EvasionAttack):
    attack_params = EvasionAttack.attack_params + [
        "norm",
        "eps",
        "eps_step",
        "attacks",
        "targeted",
        "batch_size",
        "estimator_orig",
    ]

    _estimator_requirements = (BaseEstimator,)

    def __init__(
        self,
        estimator,
        norm=np.inf,
        eps=0.3,
        eps_step=0.1,
        attacks=None,
        targeted=False,
        batch_size=32,
        estimator_orig=None,
    ):
        """
        Create a :class:`.ProjectedGradientDescent` instance.

        :param estimator: An trained estimator.
        :type estimator: :class:`.BaseEstimator`
        :param norm: The norm of the adversarial perturbation. Possible values: np.inf, 1 or 2.
        :type norm: `int`
        :param eps: Maximum perturbation that the attacker can introduce.
        :type eps: `float`
        :param eps_step: Attack step size (input variation) at each iteration.
        :type eps_step: `float`
        :param attacks: The list of `art.attacks.EvasionAttack` attacks to be used for AutoAttack. If it is `None` the
                        original AutoAttack (PGD, APGD-ce, APGD-dlr, FAB, Square) will be used.
        :type attacks: `[.art.attacks.EvasionAttack]`
        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False)
        :type targeted: `bool`
        :param batch_size: Size of the batch on which adversarial samples are generated.
        :type batch_size: `int`
        :param estimator_orig: Original estimator to be attacked by adversarial examples.
        :type estimator_orig: :class:`.BaseEstimator`
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
                    targeted=targeted,
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
                    targeted=targeted,
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

        kwargs = {
            "norm": norm,
            "eps": eps,
            "eps_step": eps_step,
            "attacks": attacks,
            "targeted": targeted,
            "batch_size": batch_size,
            "estimator_orig": estimator_orig,
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

        if y is None:
            if self.targeted:
                raise ValueError("Target labels `y` need to be provided for a targeted attack.")
            y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))

        # Determine correctly predicted samples
        y_pred = self.estimator_orig.predict(x_adv)
        sample_is_robust = np.argmax(y_pred, axis=1) == np.argmax(y, axis=1)

        for attack in self.attacks:

            if np.sum(sample_is_robust) == 0:
                break

            x_robust = x_adv[sample_is_robust]
            y_robust = y[sample_is_robust]

            # Generate adversarial examples
            x_robust_adv = attack.generate(x=x_robust, y=y_robust)
            y_pred_robust_adv = self.estimator_orig.predict(x_robust_adv)

            norm_is_smaller_eps = np.linalg.norm(
                (x_robust_adv - x_robust).reshape((x_robust_adv.shape[0], -1)), axis=1, ord=self.norm
            ) <= (self.eps + 0.001)

            sample_is_not_robust = np.logical_and(
                np.argmax(y_pred_robust_adv, axis=1) != np.argmax(y_robust, axis=1), norm_is_smaller_eps
            )

            x_robust[sample_is_not_robust] = x_robust_adv[sample_is_not_robust]
            x_adv[sample_is_robust] = x_robust

            sample_is_robust[sample_is_robust] = np.invert(sample_is_not_robust)

        return x_adv

    def set_params(self, **kwargs):
        super().set_params(**kwargs)

    def get_attacks(self):
        """
        Return the list of evasion attacks applied in this AutoAttack instance.

        :return: [.art.attacks.EvasionAttack]
        """
        return self.attacks
