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
from typing import List, Optional, Union, Tuple

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
        "defined_attack_only",
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
        defined_attack_only: bool = False,
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
        :param defined_attack_only: A bool variable to indicate whether to run only the attacks in the list as they are
                                    if `defined_attack_only` is True, otherwise the attacks in the list will be run
                                    with untargeted option following by targeted options.
        """
        super().__init__(estimator=estimator)

        if estimator_orig is None:
            estimator_orig = estimator

        # Only run attacks as they are if attack list is not None
        if defined_attack_only and attacks is None:
            raise ValueError(
                "The `defined_attack_only` option is only supported when the list of attacks input is provided."
            )

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
        self.defined_attack_only = defined_attack_only
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
        # Setup labels
        y = check_and_transform_label_format(y, self.estimator.nb_classes)

        if y is None:
            y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))

        # Determine correctly predicted samples
        y_pred = self.estimator_orig.predict(x.astype(ART_NUMPY_DTYPE))
        sample_is_robust = np.argmax(y_pred, axis=1) == np.argmax(y, axis=1)

        # Compute labels for targeted attacks
        y_ = np.array([range(y.shape[1])] * y.shape[0])
        y_idx = np.argmax(y, axis=1)
        y_idx = np.expand_dims(y_idx, 1)
        y_ = y_[y_ != y_idx]
        targeted_labels = np.reshape(y_, (y.shape[0], -1))

        # Auto attack
        if self.defined_attack_only:
            x_adv = self._run_original_attacks(
                x=x, y=y, targeted_labels=targeted_labels, sample_is_robust=sample_is_robust
            )
        else:
            x_adv = self._run_strengthen_attacks(
                x=x, y=y, targeted_labels=targeted_labels, sample_is_robust=sample_is_robust
            )

        return x_adv

    def _run_original_attacks(
        self,
        x: np.ndarray,
        y: np.ndarray,
        targeted_labels: np.ndarray,
        sample_is_robust: np.ndarray
    ) -> np.ndarray:
        """
        This function is used to run only the attacks in the attack list input as they are when `defined_attack_only`
        is True.

        :param x: An array with the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)`.
        :param targeted_labels: Target values (class labels) indices of shape `(nb_samples, nb_classes - 1)`. Used in
                                targeted attacks.
        :param sample_is_robust: Store the initial robustness of examples.
        :return: An array holding the adversarial examples.
        """
        # Create a new attack list
        full_attack_list = list()

        for attack in self.attacks:
            if hasattr(attack, 'targeted') and attack.targeted:
                for i in range(y.shape[1] - 1):
                    target = check_and_transform_label_format(targeted_labels[:, i], self.estimator.nb_classes)
                    full_attack_list.append((attack, target))

            else:
                full_attack_list.append((attack, y))

        # Auto attack
        x_adv = self._run_main_auto_attack_algorithm(
            x=x, full_attack_list=full_attack_list, sample_is_robust=sample_is_robust
        )

        return x_adv

    def _run_strengthen_attacks(
        self,
        x: np.ndarray,
        y: np.ndarray,
        targeted_labels: np.ndarray,
        sample_is_robust: np.ndarray
    ) -> np.ndarray:
        """
        This function is used to run the attacks in the attack list input with untargeted option following by targeted
        options when `defined_attack_only` is False.

        :param x: An array with the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)`.
        :param targeted_labels: Target values (class labels) indices of shape `(nb_samples, nb_classes - 1)`. Used in
                                targeted attacks.
        :param sample_is_robust: Store the initial robustness of examples.
        :return: An array holding the adversarial examples.
        """
        # Create a new attack list
        untargeted_attacks = list()
        targeted_attacks = list()

        for attack in self.attacks:
            if hasattr(attack, 'targeted'):
                # For untargeted attacks
                attack.targeted = False
                untargeted_attacks.append((attack, y))

                # For targeted attacks
                attack_ = attack.clone()
                attack_.targeted = True

                for i in range(y.shape[1] - 1):
                    target = check_and_transform_label_format(targeted_labels[:, i], self.estimator.nb_classes)
                    targeted_attacks.append((attack_, target))

            else:
                untargeted_attacks.append((attack, y))

        # Unite the 2 attack lists
        full_attack_list = untargeted_attacks + targeted_attacks

        # Auto attack
        x_adv = self._run_main_auto_attack_algorithm(
            x=x, full_attack_list=full_attack_list, sample_is_robust=sample_is_robust
        )

        return x_adv

    def _run_main_auto_attack_algorithm(
        self,
        x: np.ndarray,
        full_attack_list: List[Tuple[EvasionAttack, np.ndarray]],
        sample_is_robust: np.ndarray
    ) -> np.ndarray:
        """
        Run the main auto attack algorithm.

        :param x: An array with the original inputs.
        :param full_attack_list: A list of tuples of attacks and labels.
        :param sample_is_robust: Store the initial robustness of examples.
        :return: An array holding the adversarial examples.
        """
        # Initialize adversarial examples with original examples
        x_adv = x.astype(ART_NUMPY_DTYPE)

        # Auto attack algorithm
        for (attack, label) in full_attack_list:
            # Stop when all examples are attacked successfully
            if np.sum(sample_is_robust) == 0:
                break

            # Only attack on unsuccessful examples
            x_robust = x_adv[sample_is_robust]
            y_robust = label[sample_is_robust]

            # Generate adversarial examples
            x_robust_adv = attack.generate(x=x_robust, y=y_robust)
            y_pred_robust_adv = self.estimator_orig.predict(x_robust_adv)

            # Check and update successful examples
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

        if not isinstance(self.defined_attack_only, bool):
            raise ValueError("The flag `defined_attack_only` has to be of type bool.")
