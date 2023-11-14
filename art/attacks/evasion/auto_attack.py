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
from copy import deepcopy
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np

from art.attacks.attack import EvasionAttack
from art.attacks.evasion.auto_projected_gradient_descent import AutoProjectedGradientDescent
from art.attacks.evasion.deepfool import DeepFool
from art.attacks.evasion.square_attack import SquareAttack
from art.config import ART_NUMPY_DTYPE
from art.estimators.classification.classifier import ClassifierMixin
from art.estimators.estimator import BaseEstimator
from art.utils import check_and_transform_label_format, get_labels_np_array

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE

logger = logging.getLogger(__name__)


class AutoAttack(EvasionAttack):
    """
    Implementation of the `AutoAttack` attack.

    | Paper link: https://arxiv.org/abs/2003.01690
    """

    attack_params = EvasionAttack.attack_params + [
        "norm",
        "eps",
        "eps_step",
        "attacks",
        "batch_size",
        "estimator_orig",
        "targeted",
        "parallel",
    ]

    _estimator_requirements = (BaseEstimator, ClassifierMixin)

    # Identify samples yet to have attack metadata identified
    SAMPLE_DEFAULT = -1
    # Identify samples misclassified therefore no attack metadata required
    SAMPLE_MISCLASSIFIED = -2

    def __init__(
        self,
        estimator: "CLASSIFIER_TYPE",
        norm: Union[int, float, str] = np.inf,
        eps: float = 0.3,
        eps_step: float = 0.1,
        attacks: Optional[List[EvasionAttack]] = None,
        batch_size: int = 32,
        estimator_orig: Optional["CLASSIFIER_TYPE"] = None,
        targeted: bool = False,
        parallel: bool = False,
    ):
        """
        Create a :class:`.AutoAttack` instance.

        :param estimator: An trained estimator.
        :param norm: The norm of the adversarial perturbation. Possible values: "inf", np.inf, 1 or 2.
        :param eps: Maximum perturbation that the attacker can introduce.
        :param eps_step: Attack step size (input variation) at each iteration.
        :param attacks: The list of `art.attacks.EvasionAttack` attacks to be used for AutoAttack. If it is `None` or
                        empty the standard attacks (PGD, APGD-ce, APGD-dlr, DeepFool, Square) will be used.
        :param batch_size: Size of the batch on which adversarial samples are generated.
        :param estimator_orig: Original estimator to be attacked by adversarial examples.
        :param targeted: If False run only untargeted attacks, if True also run targeted attacks against each possible
                         target.
        :param parallel: If True run attacks in parallel.
        """
        super().__init__(estimator=estimator)

        if attacks is None or not attacks:
            attacks = []
            attacks.append(
                AutoProjectedGradientDescent(
                    estimator=estimator,  # type: ignore
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
                    estimator=estimator,  # type: ignore
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
                (
                    DeepFool(
                        classifier=estimator,  # type: ignore
                        max_iter=100,
                        epsilon=1e-3,
                        nb_grads=10,
                        batch_size=batch_size,
                    )
                )
            )
            attacks.append(
                SquareAttack(estimator=estimator, norm=norm, max_iter=5000, eps=eps, p_init=0.8, nb_restarts=5)
            )

        self.norm = norm
        self.eps = eps
        self.eps_step = eps_step
        self.attacks = attacks
        self.batch_size = batch_size
        if estimator_orig is not None:
            self.estimator_orig = estimator_orig
        else:
            self.estimator_orig = estimator

        self._targeted = targeted
        self.parallel = parallel
        self.best_attacks: np.ndarray = np.array([])
        self._check_params()

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :param mask: An array with a mask broadcastable to input `x` defining where to apply adversarial perturbations.
                     Shape needs to be broadcastable to the shape of x and can also be of the same shape as `x`. Any
                     features for which the mask is zero will not be adversarially perturbed.
        :type mask: `np.ndarray`
        :return: An array holding the adversarial examples.
        """
        import multiprocess

        x_adv = x.astype(ART_NUMPY_DTYPE)
        if y is not None:
            y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes)

        if y is None:
            y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))

        # Determine correctly predicted samples
        y_pred = self.estimator_orig.predict(x.astype(ART_NUMPY_DTYPE))
        sample_is_robust = np.argmax(y_pred, axis=1) == np.argmax(y, axis=1)

        # Set slots for images which have yet to be filled as SAMPLE_DEFAULT
        self.best_attacks = np.array([self.SAMPLE_DEFAULT] * len(x))
        # Set samples that are misclassified and do not need to be filled as SAMPLE_MISCLASSIFIED
        self.best_attacks[np.logical_not(sample_is_robust)] = self.SAMPLE_MISCLASSIFIED

        args = []
        # Untargeted attacks
        for attack in self.attacks:

            # Stop if all samples are misclassified
            if np.sum(sample_is_robust) == 0:
                break

            if attack.targeted:
                attack.set_params(targeted=False)

            if self.parallel:
                args.append(
                    (
                        deepcopy(x_adv),
                        deepcopy(y),
                        deepcopy(sample_is_robust),
                        deepcopy(attack),
                        deepcopy(self.estimator),
                        deepcopy(self.norm),
                        deepcopy(self.eps),
                    )
                )
            else:
                x_adv, sample_is_robust = run_attack(
                    x=x_adv,
                    y=y,
                    sample_is_robust=sample_is_robust,
                    attack=attack,
                    estimator_orig=self.estimator,
                    norm=self.norm,
                    eps=self.eps,
                    **kwargs,
                )
                # create a mask which identifies images which this attack was effective on
                # not including originally misclassified images
                atk_mask = np.logical_and(
                    np.array([i == self.SAMPLE_DEFAULT for i in self.best_attacks]), np.logical_not(sample_is_robust)
                )
                # update attack at image index with index of attack that was successful
                self.best_attacks[atk_mask] = self.attacks.index(attack)

        # Targeted attacks
        if self.targeted:
            # Labels for targeted attacks
            y_t = np.array([range(y.shape[1])] * y.shape[0])
            y_idx = np.argmax(y, axis=1)
            y_idx = np.expand_dims(y_idx, 1)
            y_t = y_t[y_t != y_idx]
            targeted_labels = np.reshape(y_t, (y.shape[0], self.SAMPLE_DEFAULT))

            for attack in self.attacks:

                try:
                    attack.set_params(targeted=True)

                    for i in range(self.estimator.nb_classes - 1):
                        # Stop if all samples are misclassified
                        if np.sum(sample_is_robust) == 0:
                            break

                        target = check_and_transform_label_format(
                            targeted_labels[:, i], nb_classes=self.estimator.nb_classes
                        )

                        if self.parallel:
                            args.append(
                                (
                                    deepcopy(x_adv),
                                    deepcopy(target),
                                    deepcopy(sample_is_robust),
                                    deepcopy(attack),
                                    deepcopy(self.estimator),
                                    deepcopy(self.norm),
                                    deepcopy(self.eps),
                                )
                            )
                        else:
                            x_adv, sample_is_robust = run_attack(
                                x=x_adv,
                                y=target,
                                sample_is_robust=sample_is_robust,
                                attack=attack,
                                estimator_orig=self.estimator,
                                norm=self.norm,
                                eps=self.eps,
                                **kwargs,
                            )
                            # create a mask which identifies images which this attack was effective on
                            # not including originally misclassified images
                            atk_mask = np.logical_and(
                                np.array([i == self.SAMPLE_DEFAULT for i in self.best_attacks]),
                                np.logical_not(sample_is_robust),
                            )
                            # update attack at image index with index of attack that was successful
                            self.best_attacks[atk_mask] = self.attacks.index(attack)
                except ValueError as error:
                    logger.warning("Error completing attack: %s}", str(error))

        if self.parallel:
            with multiprocess.get_context("spawn").Pool() as pool:
                # Results come back in the order that they were issued
                results = pool.starmap(run_attack, args)
            perturbations = []
            is_robust = []
            for img_idx in range(len(x)):
                perturbations.append(np.array([np.linalg.norm(x[img_idx] - i[0][img_idx]) for i in results]))
                is_robust.append([i[1][img_idx] for i in results])
            best_attacks = np.argmin(np.where(np.invert(np.array(is_robust)), np.array(perturbations), np.inf), axis=1)
            x_adv = np.concatenate([results[best_attacks[img]][0][[img]] for img in range(len(x))])
            self.best_attacks = best_attacks
            self.args = args
        return x_adv

    def _check_params(self) -> None:
        if self.norm not in [1, 2, np.inf, "inf"]:
            raise ValueError('The argument norm has to be either 1, 2, np.inf, "inf".')

        if not isinstance(self.eps, (int, float)) or self.eps <= 0.0:
            raise ValueError("The argument eps has to be either of type int or float and larger than zero.")

        if not isinstance(self.eps_step, (int, float)) or self.eps_step <= 0.0:
            raise ValueError("The argument eps_step has to be either of type int or float and larger than zero.")

        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("The argument batch_size has to be of type int and larger than zero.")

    def __repr__(self) -> str:
        """
        This method returns a summary of the best performing (lowest perturbation in the parallel case) attacks
        per image passed to the AutoAttack class.
        """
        if self.parallel:
            best_attack_meta = "\n".join(
                [
                    f"image {i+1}: {str(self.args[idx][3])}" if idx != 0 else f"image {i+1}: n/a"
                    for i, idx in enumerate(self.best_attacks)
                ]
            )
            auto_attack_meta = (
                f"AutoAttack(targeted={self.targeted}, parallel={self.parallel}, num_attacks={len(self.args)})"
            )
            return f"{auto_attack_meta}\nBestAttacks:\n{best_attack_meta}"

        best_attack_meta = "\n".join(
            [
                f"image {i+1}: {str(self.attacks[idx])}" if idx != -2 else f"image {i+1}: n/a"
                for i, idx in enumerate(self.best_attacks)
            ]
        )
        auto_attack_meta = (
            f"AutoAttack(targeted={self.targeted}, parallel={self.parallel}, num_attacks={len(self.attacks)})"
        )
        return f"{auto_attack_meta}\nBestAttacks:\n{best_attack_meta}"


def run_attack(
    x: np.ndarray,
    y: np.ndarray,
    sample_is_robust: np.ndarray,
    attack: EvasionAttack,
    estimator_orig: "CLASSIFIER_TYPE",
    norm: Union[int, float, str] = np.inf,
    eps: float = 0.3,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run attack.

    :param x: An array of the original inputs.
    :param y: An array of the labels.
    :param sample_is_robust: Store the initial robustness of examples.
    :param attack: Evasion attack to run.
    :param estimator_orig: Original estimator to be attacked by adversarial examples.
    :param norm: The norm of the adversarial perturbation. Possible values: "inf", np.inf, 1 or 2.
    :param eps: Maximum perturbation that the attacker can introduce.
    :return: An array holding the adversarial examples.
    """
    # Attack only correctly classified samples
    x_robust = x[sample_is_robust]
    y_robust = y[sample_is_robust]

    # Generate adversarial examples
    x_robust_adv = attack.generate(x=x_robust, y=y_robust, **kwargs)
    y_pred_robust_adv = estimator_orig.predict(x_robust_adv)

    # Check and update successful examples
    rel_acc = 1e-4
    order = np.inf if norm == "inf" else norm
    assert isinstance(order, (int, float))
    norm_is_smaller_eps = (1 - rel_acc) * np.linalg.norm(
        (x_robust_adv - x_robust).reshape((x_robust_adv.shape[0], -1)), axis=1, ord=order
    ) <= eps

    if attack.targeted:
        samples_misclassified = np.argmax(y_pred_robust_adv, axis=1) == np.argmax(y_robust, axis=1)
    elif not attack.targeted:
        samples_misclassified = np.argmax(y_pred_robust_adv, axis=1) != np.argmax(y_robust, axis=1)
    else:  # pragma: no cover
        raise ValueError

    sample_is_not_robust = np.logical_and(samples_misclassified, norm_is_smaller_eps)

    x_robust[sample_is_not_robust] = x_robust_adv[sample_is_not_robust]
    x[sample_is_robust] = x_robust

    sample_is_robust[sample_is_robust] = np.invert(sample_is_not_robust)

    return x, sample_is_robust
