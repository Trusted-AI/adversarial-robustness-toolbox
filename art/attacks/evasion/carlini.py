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
This module implements the L2, LInf and L0 optimized attacks `CarliniL2Method`, `CarliniLInfMethod` and `CarliniL0Method
of Carlini and Wagner (2016). These attacks are among the most effective white-box attacks and should be used among the
primary attacks to evaluate potential defences. A major difference with respect to the original implementation
(https://github.com/carlini/nn_robust_attacks) is that this implementation uses line search in the optimization of the
attack objective.

| Paper link: https://arxiv.org/abs/1608.04644
"""
# pylint: disable=C0302
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np
from tqdm.auto import trange

from art.config import ART_NUMPY_DTYPE
from art.optimizers import Adam
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassGradientsMixin
from art.attacks.attack import EvasionAttack
from art.utils import (
    compute_success,
    get_labels_np_array,
    tanh_to_original,
    original_to_tanh,
)
from art.utils import check_and_transform_label_format

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE

logger = logging.getLogger(__name__)


class CarliniL2Method(EvasionAttack):
    """
    The L_2 optimized attack of Carlini and Wagner (2016). This attack is among the most effective and should be used
    among the primary attacks to evaluate potential defences. A major difference wrt to the original implementation
    (https://github.com/carlini/nn_robust_attacks) is that we use line search in the optimization of the attack
    objective.

    | Paper link: https://arxiv.org/abs/1608.04644
    """

    attack_params = EvasionAttack.attack_params + [
        "confidence",
        "targeted",
        "learning_rate",
        "max_iter",
        "binary_search_steps",
        "initial_const",
        "max_halving",
        "max_doubling",
        "batch_size",
        "verbose",
    ]
    _estimator_requirements = (BaseEstimator, ClassGradientsMixin)

    def __init__(
        self,
        classifier: "CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE",
        confidence: float = 0.0,
        targeted: bool = False,
        learning_rate: float = 0.01,
        binary_search_steps: int = 10,
        max_iter: int = 10,
        initial_const: float = 0.01,
        max_halving: int = 5,
        max_doubling: int = 5,
        batch_size: int = 1,
        verbose: bool = True,
    ) -> None:
        """
        Create a Carlini&Wagner L_2 attack instance.

        :param classifier: A trained classifier.
        :param confidence: Confidence of adversarial examples: a higher value produces examples that are farther away,
               from the original input, but classified with higher confidence as the target class.
        :param targeted: Should the attack target one specific class.
        :param learning_rate: The initial learning rate for the attack algorithm. Smaller values produce better results
               but are slower to converge.
        :param binary_search_steps: Number of times to adjust constant with binary search (positive value). If
                                    `binary_search_steps` is large, then the algorithm is not very sensitive to the
                                    value of `initial_const`. Note that the values gamma=0.999999 and c_upper=10e10 are
                                    hardcoded with the same values used by the authors of the method.
        :param max_iter: The maximum number of iterations.
        :param initial_const: The initial trade-off constant `c` to use to tune the relative importance of distance and
                confidence. If `binary_search_steps` is large, the initial constant is not important, as discussed in
                Carlini and Wagner (2016).
        :param max_halving: Maximum number of halving steps in the line search optimization.
        :param max_doubling: Maximum number of doubling steps in the line search optimization.
        :param batch_size: Size of the batch on which adversarial samples are generated.
        :param verbose: Show progress bars.
        """
        super().__init__(estimator=classifier)

        self.confidence = confidence
        self._targeted = targeted
        self.learning_rate = learning_rate
        self.binary_search_steps = binary_search_steps
        self.max_iter = max_iter
        self.initial_const = initial_const
        self.max_halving = max_halving
        self.max_doubling = max_doubling
        self.batch_size = batch_size
        self.verbose = verbose
        CarliniL2Method._check_params(self)

        # There are internal hyperparameters:
        # Abort binary search for c if it exceeds this threshold (suggested in Carlini and Wagner (2016)):
        self._c_upper_bound = 10e10

        # Smooth arguments of arctanh by multiplying with this constant to avoid division by zero.
        # It appears this is what Carlini and Wagner (2016) are alluding to in their footnote 8. However, it is not
        # clear how their proposed trick ("instead of scaling by 1/2 we scale by 1/2 + eps") works in detail.
        self._tanh_smoother = 0.999999

    def _loss(
        self, x: np.ndarray, x_adv: np.ndarray, target: np.ndarray, c_weight: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the objective function value.

        :param x: An array with the original input.
        :param x_adv: An array with the adversarial input.
        :param target: An array with the target class (one-hot encoded).
        :param c_weight: Weight of the loss term aiming for classification as target.
        :return: A tuple holding the current logits, l2 distance and overall loss.
        """
        l2dist = np.sum(np.square(x - x_adv).reshape(x.shape[0], -1), axis=1)
        z_predicted = self.estimator.predict(
            np.array(x_adv, dtype=ART_NUMPY_DTYPE),
            logits=True,
            batch_size=self.batch_size,
        )
        z_target = np.sum(z_predicted * target, axis=1)
        z_other = np.max(
            z_predicted * (1 - target) + (np.min(z_predicted, axis=1) - 1)[:, np.newaxis] * target,
            axis=1,
        )

        # The following differs from the exact definition given in Carlini and Wagner (2016). There (page 9, left
        # column, last equation), the maximum is taken over Z_other - Z_target (or Z_target - Z_other respectively)
        # and -confidence. However, it doesn't seem that that would have the desired effect (loss term is <= 0 if and
        # only if the difference between the logit of the target and any other class differs by at least confidence).
        # Hence the rearrangement here.

        if self.targeted:
            # if targeted, optimize for making the target class most likely
            loss = np.maximum(z_other - z_target + self.confidence, np.zeros(x.shape[0]))
        else:
            # if untargeted, optimize for making any other class most likely
            loss = np.maximum(z_target - z_other + self.confidence, np.zeros(x.shape[0]))

        return z_predicted, l2dist, c_weight * loss + l2dist

    def _loss_gradient(
        self,
        z_logits: np.ndarray,
        target: np.ndarray,
        x: np.ndarray,
        x_adv: np.ndarray,
        x_adv_tanh: np.ndarray,
        c_weight: np.ndarray,
        clip_min: float,
        clip_max: float,
    ) -> np.ndarray:
        """
        Compute the gradient of the loss function.

        :param z_logits: An array with the current logits.
        :param target: An array with the target class (one-hot encoded).
        :param x: An array with the original input.
        :param x_adv: An array with the adversarial input.
        :param x_adv_tanh: An array with the adversarial input in tanh space.
        :param c_weight: Weight of the loss term aiming for classification as target.
        :param clip_min: Minimum clipping value.
        :param clip_max: Maximum clipping value.
        :return: An array with the gradient of the loss function.
        """
        if self.targeted:
            i_sub = np.argmax(target, axis=1)
            i_add = np.argmax(
                z_logits * (1 - target) + (np.min(z_logits, axis=1) - 1)[:, np.newaxis] * target,
                axis=1,
            )
        else:
            i_add = np.argmax(target, axis=1)
            i_sub = np.argmax(
                z_logits * (1 - target) + (np.min(z_logits, axis=1) - 1)[:, np.newaxis] * target,
                axis=1,
            )

        loss_gradient = self.estimator.class_gradient(x_adv, label=i_add)
        loss_gradient -= self.estimator.class_gradient(x_adv, label=i_sub)
        loss_gradient = loss_gradient.reshape(x.shape)

        c_mult = c_weight
        for _ in range(len(x.shape) - 1):
            c_mult = c_mult[:, np.newaxis]

        loss_gradient *= c_mult
        loss_gradient += 2 * (x_adv - x)
        loss_gradient *= clip_max - clip_min
        loss_gradient *= (1 - np.square(np.tanh(x_adv_tanh))) / (2 * self._tanh_smoother)

        return loss_gradient

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs to be attacked.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,). If `self.targeted` is true, then `y` represents the target labels. If `self.targeted`
                  is true, then `y_val` represents the target labels. Otherwise, the targets are the original class
                  labels.
        :return: An array holding the adversarial examples.
        """
        if y is not None:
            y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes)
        x_adv = x.astype(ART_NUMPY_DTYPE)

        if self.estimator.clip_values is not None:
            clip_min, clip_max = self.estimator.clip_values
        else:
            clip_min, clip_max = np.amin(x), np.amax(x)

        # Assert that, if attack is targeted, y_val is provided:
        if self.targeted and y is None:
            raise ValueError("Target labels `y` need to be provided for a targeted attack.")

        # No labels provided, use model prediction as correct class
        if y is None:
            y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))

        if self.estimator.nb_classes == 2 and y.shape[1] == 1:
            raise ValueError(  # pragma: no cover
                "This attack has not yet been tested for binary classification with a single output classifier."
            )

        # Compute perturbation with implicit batching
        nb_batches = int(np.ceil(x_adv.shape[0] / float(self.batch_size)))
        for batch_id in trange(nb_batches, desc="C&W L_2", disable=not self.verbose):
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            x_batch = x_adv[batch_index_1:batch_index_2]
            y_batch = y[batch_index_1:batch_index_2]

            # The optimization is performed in tanh space to keep the adversarial images bounded in correct range
            x_batch_tanh = original_to_tanh(x_batch, clip_min, clip_max, self._tanh_smoother)

            # Initialize binary search:
            c_current = self.initial_const * np.ones(x_batch.shape[0])
            c_lower_bound = np.zeros(x_batch.shape[0])
            c_double = np.ones(x_batch.shape[0]) > 0

            # Initialize placeholders for best l2 distance and attack found so far
            best_l2dist = np.inf * np.ones(x_batch.shape[0])
            best_x_adv_batch = x_batch.copy()

            for bss in range(self.binary_search_steps):
                logger.debug(
                    "Binary search step %i out of %i (c_mean==%f)",
                    bss,
                    self.binary_search_steps,
                    np.mean(c_current),
                )
                nb_active = int(np.sum(c_current < self._c_upper_bound))
                logger.debug(
                    "Number of samples with c_current < _c_upper_bound: %i out of %i",
                    nb_active,
                    x_batch.shape[0],
                )
                if nb_active == 0:  # pragma: no cover
                    break
                learning_rate = self.learning_rate * np.ones(x_batch.shape[0])

                # Initialize perturbation in tanh space:
                x_adv_batch = x_batch.copy()
                x_adv_batch_tanh = x_batch_tanh.copy()

                z_logits, l2dist, loss = self._loss(x_batch, x_adv_batch, y_batch, c_current)  # type: ignore
                attack_success = loss - l2dist <= 0
                overall_attack_success = attack_success

                for i_iter in range(self.max_iter):
                    logger.debug("Iteration step %i out of %i", i_iter, self.max_iter)
                    logger.debug("Average Loss: %f", np.mean(loss))
                    logger.debug("Average L2Dist: %f", np.mean(l2dist))
                    logger.debug("Average Margin Loss: %f", np.mean(loss - l2dist))
                    logger.debug(
                        "Current number of succeeded attacks: %i out of %i",
                        int(np.sum(attack_success)),
                        len(attack_success),
                    )

                    improved_adv = attack_success & (l2dist < best_l2dist)
                    logger.debug("Number of improved L2 distances: %i", int(np.sum(improved_adv)))
                    if np.sum(improved_adv) > 0:
                        best_l2dist[improved_adv] = l2dist[improved_adv]
                        best_x_adv_batch[improved_adv] = x_adv_batch[improved_adv]

                    active = (c_current < self._c_upper_bound) & (learning_rate > 0)
                    nb_active = int(np.sum(active))
                    logger.debug(
                        "Number of samples with c_current < _c_upper_bound and learning_rate > 0: %i out of %i",
                        nb_active,
                        x_batch.shape[0],
                    )
                    if nb_active == 0:  # pragma: no cover
                        break

                    # compute gradient:
                    logger.debug("Compute loss gradient")
                    perturbation_tanh = -self._loss_gradient(
                        z_logits[active],
                        y_batch[active],
                        x_batch[active],
                        x_adv_batch[active],
                        x_adv_batch_tanh[active],
                        c_current[active],
                        clip_min,
                        clip_max,
                    )

                    # perform line search to optimize perturbation
                    # first, halve the learning rate until perturbation actually decreases the loss:
                    prev_loss = loss.copy()
                    best_loss = loss.copy()
                    best_lr = np.zeros(x_batch.shape[0])
                    halving = np.zeros(x_batch.shape[0])

                    for i_halve in range(self.max_halving):
                        logger.debug(
                            "Perform halving iteration %i out of %i",
                            i_halve,
                            self.max_halving,
                        )
                        do_halving = loss[active] >= prev_loss[active]
                        logger.debug(
                            "Halving to be performed on %i samples",
                            int(np.sum(do_halving)),
                        )
                        if np.sum(do_halving) == 0:
                            break
                        active_and_do_halving = active.copy()
                        active_and_do_halving[active] = do_halving

                        lr_mult = learning_rate[active_and_do_halving]
                        for _ in range(len(x.shape) - 1):
                            lr_mult = lr_mult[:, np.newaxis]

                        x_adv1 = x_adv_batch_tanh[active_and_do_halving]
                        new_x_adv_batch_tanh = x_adv1 + lr_mult * perturbation_tanh[do_halving]
                        new_x_adv_batch = tanh_to_original(new_x_adv_batch_tanh, clip_min, clip_max)
                        _, l2dist[active_and_do_halving], loss[active_and_do_halving] = self._loss(
                            x_batch[active_and_do_halving],
                            new_x_adv_batch,
                            y_batch[active_and_do_halving],
                            c_current[active_and_do_halving],
                        )

                        logger.debug("New Average Loss: %f", np.mean(loss))
                        logger.debug("New Average L2Dist: %f", np.mean(l2dist))
                        logger.debug("New Average Margin Loss: %f", np.mean(loss - l2dist))

                        best_lr[loss < best_loss] = learning_rate[loss < best_loss]
                        best_loss[loss < best_loss] = loss[loss < best_loss]
                        learning_rate[active_and_do_halving] /= 2
                        halving[active_and_do_halving] += 1
                    learning_rate[active] *= 2

                    # if no halving was actually required, double the learning rate as long as this
                    # decreases the loss:
                    for i_double in range(self.max_doubling):
                        logger.debug(
                            "Perform doubling iteration %i out of %i",
                            i_double,
                            self.max_doubling,
                        )
                        do_doubling = (halving[active] == 1) & (loss[active] <= best_loss[active])
                        logger.debug(
                            "Doubling to be performed on %i samples",
                            int(np.sum(do_doubling)),
                        )
                        if np.sum(do_doubling) == 0:
                            break
                        active_and_do_doubling = active.copy()
                        active_and_do_doubling[active] = do_doubling
                        learning_rate[active_and_do_doubling] *= 2

                        lr_mult = learning_rate[active_and_do_doubling]
                        for _ in range(len(x.shape) - 1):
                            lr_mult = lr_mult[:, np.newaxis]

                        x_adv2 = x_adv_batch_tanh[active_and_do_doubling]
                        new_x_adv_batch_tanh = x_adv2 + lr_mult * perturbation_tanh[do_doubling]
                        new_x_adv_batch = tanh_to_original(new_x_adv_batch_tanh, clip_min, clip_max)
                        _, l2dist[active_and_do_doubling], loss[active_and_do_doubling] = self._loss(
                            x_batch[active_and_do_doubling],
                            new_x_adv_batch,
                            y_batch[active_and_do_doubling],
                            c_current[active_and_do_doubling],
                        )
                        logger.debug("New Average Loss: %f", np.mean(loss))
                        logger.debug("New Average L2Dist: %f", np.mean(l2dist))
                        logger.debug("New Average Margin Loss: %f", np.mean(loss - l2dist))
                        best_lr[loss < best_loss] = learning_rate[loss < best_loss]
                        best_loss[loss < best_loss] = loss[loss < best_loss]

                    learning_rate[halving == 1] /= 2

                    update_adv = best_lr[active] > 0
                    logger.debug(
                        "Number of adversarial samples to be finally updated: %i",
                        int(np.sum(update_adv)),
                    )

                    if np.sum(update_adv) > 0:
                        active_and_update_adv = active.copy()
                        active_and_update_adv[active] = update_adv
                        best_lr_mult = best_lr[active_and_update_adv]
                        for _ in range(len(x.shape) - 1):
                            best_lr_mult = best_lr_mult[:, np.newaxis]

                        x_adv4 = x_adv_batch_tanh[active_and_update_adv]
                        best_lr1 = best_lr_mult * perturbation_tanh[update_adv]
                        x_adv_batch_tanh[active_and_update_adv] = x_adv4 + best_lr1

                        x_adv6 = x_adv_batch_tanh[active_and_update_adv]
                        x_adv_batch[active_and_update_adv] = tanh_to_original(x_adv6, clip_min, clip_max)
                        (
                            z_logits[active_and_update_adv],
                            l2dist[active_and_update_adv],
                            loss[active_and_update_adv],
                        ) = self._loss(
                            x_batch[active_and_update_adv],
                            x_adv_batch[active_and_update_adv],
                            y_batch[active_and_update_adv],
                            c_current[active_and_update_adv],
                        )
                        attack_success = loss - l2dist <= 0
                        overall_attack_success = overall_attack_success | attack_success

                # Update depending on attack success:
                improved_adv = attack_success & (l2dist < best_l2dist)
                logger.debug("Number of improved L2 distances: %i", int(np.sum(improved_adv)))

                if np.sum(improved_adv) > 0:
                    best_l2dist[improved_adv] = l2dist[improved_adv]
                    best_x_adv_batch[improved_adv] = x_adv_batch[improved_adv]

                c_double[overall_attack_success] = False
                c_current[overall_attack_success] = (c_lower_bound + c_current)[overall_attack_success] / 2

                c_old = c_current
                c_current[~overall_attack_success & c_double] *= 2

                c_current1 = (c_current - c_lower_bound)[~overall_attack_success & ~c_double]
                c_current[~overall_attack_success & ~c_double] += c_current1 / 2
                c_lower_bound[~overall_attack_success] = c_old[~overall_attack_success]

            x_adv[batch_index_1:batch_index_2] = best_x_adv_batch

        logger.info(
            "Success rate of C&W L_2 attack: %.2f%%",
            100 * compute_success(self.estimator, x, y, x_adv, self.targeted, batch_size=self.batch_size),
        )

        return x_adv

    def _check_params(self) -> None:
        if (
            not isinstance(
                self.binary_search_steps,
                int,
            )
            or self.binary_search_steps < 0
        ):
            raise ValueError("The number of binary search steps must be a non-negative integer.")

        if not isinstance(self.max_iter, int) or self.max_iter < 0:
            raise ValueError("The number of iterations must be a non-negative integer.")

        if not isinstance(self.max_halving, int) or self.max_halving < 1:
            raise ValueError("The number of halving steps must be an integer greater than zero.")

        if not isinstance(self.max_doubling, int) or self.max_doubling < 1:
            raise ValueError("The number of doubling steps must be an integer greater than zero.")

        if not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError("The batch size must be an integer greater than zero.")


class CarliniLInfMethod(EvasionAttack):
    """
    This is a modified version of the L_2 optimized attack of Carlini and Wagner (2016). It controls the L_Inf
    norm, i.e. the maximum perturbation applied to each pixel.
    """

    attack_params = EvasionAttack.attack_params + [
        "confidence",
        "targeted",
        "learning_rate",
        "max_iter",
        "decrease_factor",
        "initial_const",
        "largest_const",
        "const_factor",
        "batch_size",
        "verbose",
    ]
    _estimator_requirements = (BaseEstimator, ClassGradientsMixin)

    def __init__(
        self,
        classifier: "CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE",
        confidence: float = 0.0,
        targeted: bool = False,
        learning_rate: float = 0.01,
        max_iter: int = 10,
        decrease_factor: float = 0.9,
        initial_const: float = 1e-5,
        largest_const: float = 20.0,
        const_factor: float = 2.0,
        batch_size: int = 1,
        verbose: bool = True,
    ) -> None:
        """
        Create a Carlini&Wagner L_Inf attack instance.

        :param classifier: A trained classifier.
        :param confidence: Confidence of adversarial examples: a higher value produces examples that are farther away,
                from the original input, but classified with higher confidence as the target class.
        :param targeted: Should the attack target one specific class.
        :param learning_rate: The initial learning rate for the attack algorithm. Smaller values produce better
                results but are slower to converge.
        :param max_iter: The maximum number of iterations.
        :param decrease_factor: The rate of shrinking tau, values in `0 < decrease_factor < 1` where larger is more
                                accurate.
        :param initial_const: The initial value of constant `c`.
        :param largest_const: The largest value of constant `c`.
        :param const_factor: The rate of increasing constant `c` with `const_factor > 1`, where smaller more accurate.
        :param batch_size: Size of the batch on which adversarial samples are generated.
        :param verbose: Show progress bars.
        """
        super().__init__(estimator=classifier)

        self.confidence = confidence
        self._targeted = targeted
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.decrease_factor = decrease_factor
        self.initial_const = initial_const
        self.largest_const = largest_const
        self.const_factor = const_factor
        self.batch_size = batch_size
        self.verbose = verbose
        self._check_params()

        # There is one internal hyperparameter:
        # Smooth arguments of arctanh by multiplying with this constant to avoid division by zero:
        self._tanh_smoother = 0.999999

    def _loss(
        self, x_adv: np.ndarray, target: np.ndarray, x, const, tau
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the objective function value.

        :param x_adv: An array with the adversarial examples.
        :param target: An array with the target class (one-hot encoded).
        :param x: Benign samples.
        :param  const: Current constant `c`.
        :param tau: Current limit `tau`.
        :return: A tuple of current predictions, total loss, logits loss and regularisation loss.
        """
        z_predicted = self.estimator.predict(np.array(x_adv, dtype=ART_NUMPY_DTYPE), batch_size=self.batch_size)
        z_target = np.sum(z_predicted * target, axis=1)
        z_other = np.max(
            z_predicted * (1 - target) + (np.min(z_predicted, axis=1) - 1)[:, np.newaxis] * target,
            axis=1,
        )

        if self.targeted:
            # if targeted, optimize for making the target class most likely
            loss_1 = np.maximum(z_other - z_target + self.confidence, np.zeros(x_adv.shape[0]))
        else:
            # if untargeted, optimize for making any other class most likely
            loss_1 = np.maximum(z_target - z_other + self.confidence, np.zeros(x_adv.shape[0]))

        loss_2 = np.sum(np.maximum(0.0, np.abs(x_adv - x) - tau))

        loss = loss_1 * const + loss_2

        return z_predicted, loss, loss_1, loss_2

    def _loss_gradient(
        self,
        z_logits: np.ndarray,
        target: np.ndarray,
        x_adv: np.ndarray,
        x_adv_tanh: np.ndarray,
        clip_min: np.ndarray,
        clip_max: np.ndarray,
        x,
        tau,
    ) -> np.ndarray:
        """
        Compute the gradient of the loss function.

        :param z_logits: An array with the current predictions.
        :param target: An array with the target class (one-hot encoded).
        :param x_adv: An array with the adversarial input.
        :param x_adv_tanh: An array with the adversarial input in tanh space.
        :param clip_min: Minimum clipping values.
        :param clip_max: Maximum clipping values.
        :param x: Benign samples.
        :param tau: Current limit `tau`.
        :return: An array with the gradient of the loss function.
        """
        if self.targeted:
            i_sub = np.argmax(target, axis=1)
            i_add = np.argmax(
                z_logits * (1 - target) + (np.min(z_logits, axis=1) - 1)[:, np.newaxis] * target,
                axis=1,
            )
        else:
            i_add = np.argmax(target, axis=1)
            i_sub = np.argmax(
                z_logits * (1 - target) + (np.min(z_logits, axis=1) - 1)[:, np.newaxis] * target,
                axis=1,
            )

        loss_gradient = self.estimator.class_gradient(x_adv, label=i_add)
        loss_gradient -= self.estimator.class_gradient(x_adv, label=i_sub)
        loss_gradient = loss_gradient.reshape(x_adv.shape)

        loss_gradient_2 = np.sign(np.maximum(0.0, np.abs(x_adv - x) - tau)) * np.sign(x_adv - x)
        loss_gradient_2 *= clip_max - clip_min
        loss_gradient_2 *= (1 - np.square(np.tanh(x_adv_tanh))) / (2 * self._tanh_smoother)

        loss_gradient *= clip_max - clip_min
        loss_gradient *= (1 - np.square(np.tanh(x_adv_tanh))) / (2 * self._tanh_smoother)

        loss_gradient = loss_gradient + loss_gradient_2

        return loss_gradient

    def _generate_single(self, x_batch, y_batch, clip_min, clip_max, const, tau):
        """
        Generate a single adversarial example.

        :param x_batch: Current benign sample.
        :param y_batch: Current label.
        :param clip_min: Minimum clipping values.
        :param clip_max: Maximum clipping values.
        :param  const: Current constant `c`.
        :param tau: Current limit `tau`.
        """

        # The optimization is performed in tanh space to keep the adversarial images bounded from clip_min and clip_max.
        x_adv_batch_tanh = original_to_tanh(x_batch, clip_min, clip_max, self._tanh_smoother)

        def func(x_i):
            x_adv_batch_tanh = x_i

            x_adv_batch = tanh_to_original(
                x_adv_batch_tanh,
                clip_min,
                clip_max,
            )

            _, loss, _, _ = self._loss(x_adv_batch, y_batch, x_batch, const, tau)

            return loss

        def func_der(x_i):
            x_adv_batch_tanh = x_i

            x_adv_batch = tanh_to_original(
                x_adv_batch_tanh,
                clip_min,
                clip_max,
            )

            z_logits, _, _, _ = self._loss(x_adv_batch, y_batch, x_batch, const, tau)

            perturbation_tanh = self._loss_gradient(
                z_logits,
                y_batch,
                x_adv_batch,
                x_adv_batch_tanh,
                clip_min,
                clip_max,
                x_batch,
                tau,
            )

            return perturbation_tanh

        x_0 = x_adv_batch_tanh.copy()

        adam = Adam(alpha=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        x_adv_batch_tanh = adam.optimize(func=func, jac=func_der, x_0=x_0, max_iter=self.max_iter, loss_converged=0.001)

        x_adv_batch = tanh_to_original(
            x_adv_batch_tanh,
            clip_min,
            clip_max,
        )

        return x_adv_batch

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs to be attacked.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,). If `self.targeted` is true, then `y_val` represents the target labels. Otherwise, the
                  targets are the original class labels.
        :return: An array holding the adversarial examples.
        """
        if y is not None:
            y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes)
        x_adv = x.astype(ART_NUMPY_DTYPE)

        if self.estimator.clip_values is not None:
            clip_min, clip_max = self.estimator.clip_values
        else:
            clip_min, clip_max = np.amin(x), np.amax(x)

        # Assert that, if attack is targeted, y_val is provided:
        if self.targeted and y is None:
            raise ValueError("Target labels `y` need to be provided for a targeted attack.")

        # No labels provided, use model prediction as correct class
        if y is None:
            y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))

        if self.estimator.nb_classes == 2 and y.shape[1] == 1:
            raise ValueError(  # pragma: no cover
                "This attack has not yet been tested for binary classification with a single output classifier."
            )

        # Compute perturbation with implicit batching
        for sample_id in trange(x.shape[0], desc="C&W L_inf", disable=not self.verbose):

            sample_done = False
            tau = 1.0
            delta_i_best = 1.0
            while tau > 1.0 / 256.0 and not sample_done:

                sample_done = True

                const = self.initial_const
                while const < self.largest_const:

                    x_batch = x[[sample_id]]
                    y_batch = y[[sample_id]]

                    x_adv_batch = self._generate_single(x_batch, y_batch, clip_min, clip_max, const=const, tau=tau)

                    # Update depending on attack success:
                    _, loss, loss_1, loss_2 = self._loss(x_adv_batch, y_batch, x_batch, const, tau)

                    delta_i = np.max(np.abs(x_adv_batch - x[sample_id]))

                    logger.debug(
                        "tau: %4.3f, const: %4.5f, loss: %4.3f, loss_1: %4.3f, loss_2: %4.3f, delta_i: %4.3f",
                        tau,
                        const,
                        loss,
                        loss_1,
                        loss_2,
                        delta_i,
                    )

                    if (
                        np.argmax(self.estimator.predict(x_adv_batch), axis=1) != np.argmax(y_batch, axis=1)
                        and delta_i < delta_i_best
                    ):
                        x_adv[sample_id] = x_adv_batch
                        delta_i_best = delta_i
                        sample_done = False

                    const *= self.const_factor

                tau_actual = np.max(np.abs(x_adv[sample_id] - x[sample_id]))

                if tau_actual < tau:
                    tau = tau_actual

                tau *= self.decrease_factor

        return x_adv

    def _check_params(self) -> None:

        if not isinstance(self.max_iter, int) or self.max_iter < 0:
            raise ValueError("The number of iterations must be a non-negative integer.")

        if not isinstance(self.decrease_factor, (int, float)) or not 0.0 < self.decrease_factor < 1.0:
            raise ValueError("The decrease factor must be a float between 0 and 1.")

        if not isinstance(self.initial_const, (int, float)) or self.initial_const < 0:
            raise ValueError("The initial constant value must be a positive float.")

        if not isinstance(self.largest_const, (int, float)) or self.largest_const < 0:
            print(self.largest_const)
            raise ValueError("The largest constant value must be a positive float.")

        if not isinstance(self.const_factor, (int, float)) or self.const_factor < 0:
            raise ValueError("The constant factor value must be a float and greater than 1.")

        if not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError("The batch size must be an integer greater than zero.")


class CarliniL0Method(CarliniL2Method):
    """
    The L_0 distance metric is non-differentiable and therefore is ill-suited for standard gradient descent.
    Instead, we use an iterative algorithm that, in each iteration, identifies some features that don’t have much effect
    on the classifier output and then fixes those features, so their value will never be changed.
    The set of fixed features grows in each iteration until we have, by process of elimination, identified a minimal
    (but possibly not minimum) subset of features that can be modified to generate an adversarial example.
    In each iteration, we use our L_2 attack to identify which features are unimportant [Carlini and Wagner, 2016].

    | Paper link: https://arxiv.org/abs/1608.04644
    """

    attack_params = EvasionAttack.attack_params + [
        "confidence",
        "targeted",
        "learning_rate",
        "max_iter",
        "binary_search_steps",
        "initial_const",
        "mask",
        "warm_start",
        "max_halving",
        "max_doubling",
        "batch_size",
        "verbose",
    ]

    _estimator_requirements = (BaseEstimator, ClassGradientsMixin)

    def __init__(
        self,
        classifier: "CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE",
        confidence: float = 0.0,
        targeted: bool = False,
        learning_rate: float = 0.01,
        binary_search_steps: int = 10,
        max_iter: int = 10,
        initial_const: float = 0.01,
        mask: Optional[np.ndarray] = None,
        warm_start: bool = True,
        max_halving: int = 5,
        max_doubling: int = 5,
        batch_size: int = 1,
        verbose: bool = True,
    ):
        """
        Create a Carlini&Wagner L_0 attack instance.

        :param classifier: A trained classifier.
        :param confidence: Confidence of adversarial examples: a higher value produces examples that are farther away,
                           from the original input, but classified with higher confidence as the target class.
        :param targeted: Should the attack target one specific class.
        :param learning_rate: The initial learning rate for the attack algorithm. Smaller values produce better results
                              but are slower to converge.
        :param binary_search_steps: Number of times to adjust constant with binary search (positive value). If
                                    `binary_search_steps` is large, then the algorithm is not very sensitive to the
                                    value of `initial_const`. Note that the values gamma=0.999999 and c_upper=10e10 are
                                    hardcoded with the same values used by the authors of the method.
        :param max_iter: The maximum number of iterations.
        :param initial_const: The initial trade-off constant `c` to use to tune the relative importance of distance and
                              confidence. If `binary_search_steps` is large, the initial constant is not important, as
                              discussed in Carlini and Wagner (2016).
        :param mask: The initial features that can be modified by the algorithm. If not specified, the
                     algorithm uses the full feature set.
        :param warm_start: Instead of starting gradient descent in each iteration from the initial image. we start the
                           gradient descent from the solution found on the previous iteration.
        :param max_halving: Maximum number of halving steps in the line search optimization.
        :param max_doubling: Maximum number of doubling steps in the line search optimization.
        :param batch_size: Size of the batch on which adversarial samples are generated.
        :param verbose: Show progress bars.
        """
        super().__init__(
            classifier=classifier,
            confidence=confidence,
            targeted=targeted,
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_halving=max_halving,
            max_doubling=max_doubling,
            batch_size=batch_size,
            verbose=verbose,
        )

        self.binary_search_steps = binary_search_steps
        self.initial_const = initial_const
        self.mask = mask
        self.warm_start = warm_start
        self._check_params()

        # There are internal hyperparameters:
        # Abort binary search for c if it exceeds this threshold (suggested in Carlini and Wagner (2016)):
        self._c_upper_bound = 10e10

        # Smooth arguments of arctanh by multiplying with this constant to avoid division by zero.
        # It appears this is what Carlini and Wagner (2016) are alluding to in their footnote 8. However, it is not
        # clear how their proposed trick ("instead of scaling by 1/2 we scale by 1/2 + eps") works in detail.
        self._tanh_smoother = 0.999999

        # The tanh transformation does not always map inputs back to their original values event if they are unmodified
        # To overcome this problem, we set a threshold of minimal difference considered as perturbation
        # Below this threshold, a difference between values is considered as tanh transformation difference.
        self._perturbation_threshold = 1e-06

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs to be attacked.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,). If `self.targeted` is true, then `y` represents the target labels. If `self.targeted`
                  is true, then `y_val` represents the target labels. Otherwise, the targets are the original class
                  labels.
        :return: An array holding the adversarial examples.
        """
        if y is not None:
            y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes)
        x_adv = x.astype(ART_NUMPY_DTYPE)

        if self.estimator.clip_values is not None:
            clip_min, clip_max = self.estimator.clip_values
        else:
            clip_min, clip_max = np.amin(x), np.amax(x)

        # Assert that, if attack is targeted, y_val is provided:
        if self.targeted and y is None:
            raise ValueError("Target labels `y` need to be provided for a targeted attack.")

        # No labels provided, use model prediction as correct class
        if y is None:
            y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))

        if self.estimator.nb_classes == 2 and y.shape[1] == 1:
            raise ValueError(
                "This attack has not yet been tested for binary classification with a single output classifier."
            )

        if self.mask is None:
            # No initial activation provided, use the full feature set
            activation = np.ones(x.shape)
        else:
            # Check if the initial activation has the same dimension as the input data
            if self.mask.shape != x.shape:  # pragma: no cover
                raise ValueError("The mask must have the same dimensions as the input data.")
            activation = np.array(self.mask).astype(float)

        # L_0 attack specific variables
        final_adversarial_example = x.astype(ART_NUMPY_DTYPE)
        old_activation = activation.copy()
        c_final = np.ones(x.shape[0])
        best_l0dist = np.inf * np.ones(x.shape[0])

        # Main loop of the L_0 attack.
        # For each iteration :
        #   - Calls the L_2 attack to compute an adversarial example
        #   - Computes the gradients of the objective function evaluated at the adversarial instance
        #   - Fix the attribute with the lowest value (gradient * perturbation)
        # Repeat until the L_2 attack fails to find an adversarial examples.
        for _ in range(x.shape[1] + 1):
            # Compute perturbation with implicit batching
            nb_batches = int(np.ceil(x_adv.shape[0] / float(self.batch_size)))
            for batch_id in range(nb_batches):
                logger.debug("Processing batch %i out of %i", batch_id, nb_batches)

                batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
                if self.warm_start:
                    # Start the gradient descent from the solution found on the previous iteration
                    x_batch = x_adv[batch_index_1:batch_index_2]
                else:
                    x_batch = x[batch_index_1:batch_index_2]
                y_batch = y[batch_index_1:batch_index_2]
                activation_batch = activation[batch_index_1:batch_index_2]

                # The optimization is performed in tanh space to keep the adversarial images bounded in correct range
                x_batch_tanh = original_to_tanh(x_batch, clip_min, clip_max, self._tanh_smoother)

                # Initialize binary search:
                c_current = self.initial_const * np.ones(x_batch.shape[0])
                c_lower_bound = np.zeros(x_batch.shape[0])
                c_double = np.ones(x_batch.shape[0]) > 0

                # Initialize placeholders for best l2 distance and attack found so far
                best_l0dist_batch = np.inf * np.ones(x_batch.shape[0])
                best_x_adv_batch = x_batch.copy()

                for bss in range(self.binary_search_steps):
                    logger.debug(
                        "Binary search step %i / %i (c_mean==%f)", bss, self.binary_search_steps, np.mean(c_current)
                    )
                    nb_active = int(np.sum(c_current < self._c_upper_bound))
                    logger.debug(
                        "Number of samples with c_current < _c_upper_bound: %i out of %i", nb_active, x_batch.shape[0]
                    )
                    if nb_active == 0:
                        break
                    learning_rate = self.learning_rate * np.ones(x_batch.shape[0])

                    # Initialize perturbation in tanh space:
                    x_adv_batch = x_batch.copy()
                    x_adv_batch_tanh = x_batch_tanh.copy()

                    z_logits, l2dist, loss = self._loss(x_batch, x_adv_batch, y_batch, c_current)  # type: ignore
                    attack_success = loss - l2dist <= 0
                    overall_attack_success = attack_success

                    for i_iter in range(self.max_iter):
                        logger.debug("Iteration step %i out of %i", i_iter, self.max_iter)
                        logger.debug("Average Loss: %f", np.mean(loss))
                        logger.debug("Average L2Dist: %f", np.mean(l2dist))
                        logger.debug("Average Margin Loss: %f", np.mean(loss - l2dist))
                        logger.debug(
                            "Current number of succeeded attacks: %i out of %i",
                            int(np.sum(attack_success)),
                            len(attack_success),
                        )

                        l0dist = np.sum(
                            (np.abs(x_batch - x_adv_batch) > self._perturbation_threshold).astype(int), axis=(1, 2, 3)
                        )
                        improved_adv = attack_success & (l0dist < best_l0dist_batch)
                        logger.debug("Number of improved L0 distances: %i", int(np.sum(improved_adv)))
                        if np.sum(improved_adv) > 0:
                            best_l0dist_batch[improved_adv] = l0dist[improved_adv]
                            best_x_adv_batch[improved_adv] = x_adv_batch[improved_adv]

                        active = (c_current < self._c_upper_bound) & (learning_rate > 0)
                        nb_active = int(np.sum(active))
                        logger.debug(
                            "Number of samples with c_current < _c_upper_bound and learning_rate > 0: %i out of %i",
                            nb_active,
                            x_batch.shape[0],
                        )
                        if nb_active == 0:  # pragma: no cover
                            break

                        # compute gradient:
                        logger.debug("Compute loss gradient")
                        perturbation_tanh = -self._loss_gradient(
                            z_logits[active],
                            y_batch[active],
                            x_batch[active],
                            x_adv_batch[active],
                            x_adv_batch_tanh[active],
                            c_current[active],
                            clip_min,
                            clip_max,
                        )

                        # perform line search to optimize perturbation
                        # first, halve the learning rate until perturbation actually decreases the loss:
                        prev_loss = loss.copy()
                        best_loss = loss.copy()
                        best_lr = np.zeros(x_batch.shape[0])
                        halving = np.zeros(x_batch.shape[0])

                        for i_halve in range(self.max_halving):
                            logger.debug("Perform halving iteration %i out of %i", i_halve, self.max_halving)
                            do_halving = loss[active] >= prev_loss[active]
                            logger.debug("Halving to be performed on %i samples", int(np.sum(do_halving)))
                            if np.sum(do_halving) == 0:
                                break
                            active_and_do_halving = active.copy()
                            active_and_do_halving[active] = do_halving

                            lr_mult = learning_rate[active_and_do_halving]
                            for _ in range(len(x.shape) - 1):
                                lr_mult = lr_mult[:, np.newaxis]

                            x_adv1 = x_adv_batch_tanh[active_and_do_halving]
                            new_x_adv_batch_tanh = (
                                x_adv1 + lr_mult * perturbation_tanh[do_halving] * activation_batch[do_halving]
                            )
                            new_x_adv_batch = tanh_to_original(new_x_adv_batch_tanh, clip_min, clip_max)
                            _, l2dist[active_and_do_halving], loss[active_and_do_halving] = self._loss(
                                x_batch[active_and_do_halving],
                                new_x_adv_batch,
                                y_batch[active_and_do_halving],
                                c_current[active_and_do_halving],
                            )

                            logger.debug("New Average Loss: %f", np.mean(loss))
                            logger.debug("New Average L2Dist: %f", np.mean(l2dist))
                            logger.debug("New Average Margin Loss: %f", np.mean(loss - l2dist))

                            best_lr[loss < best_loss] = learning_rate[loss < best_loss]
                            best_loss[loss < best_loss] = loss[loss < best_loss]
                            learning_rate[active_and_do_halving] /= 2
                            halving[active_and_do_halving] += 1
                        learning_rate[active] *= 2

                        # if no halving was actually required, double the learning rate as long as this
                        # decreases the loss:
                        for i_double in range(self.max_doubling):
                            logger.debug("Perform doubling iteration %i out of %i", i_double, self.max_doubling)
                            do_doubling = (halving[active] == 1) & (loss[active] <= best_loss[active])
                            logger.debug("Doubling to be performed on %i samples", int(np.sum(do_doubling)))
                            if np.sum(do_doubling) == 0:  # pragma: no cover
                                break
                            active_and_do_doubling = active.copy()
                            active_and_do_doubling[active] = do_doubling
                            learning_rate[active_and_do_doubling] *= 2

                            lr_mult = learning_rate[active_and_do_doubling]
                            for _ in range(len(x.shape) - 1):
                                lr_mult = lr_mult[:, np.newaxis]

                            x_adv2 = x_adv_batch_tanh[active_and_do_doubling]
                            new_x_adv_batch_tanh = (
                                x_adv2 + lr_mult * perturbation_tanh[do_doubling] * activation_batch[do_doubling]
                            )
                            new_x_adv_batch = tanh_to_original(new_x_adv_batch_tanh, clip_min, clip_max)
                            _, l2dist[active_and_do_doubling], loss[active_and_do_doubling] = self._loss(
                                x_batch[active_and_do_doubling],
                                new_x_adv_batch,
                                y_batch[active_and_do_doubling],
                                c_current[active_and_do_doubling],
                            )
                            logger.debug("New Average Loss: %f", np.mean(loss))
                            logger.debug("New Average L2Dist: %f", np.mean(l2dist))
                            logger.debug("New Average Margin Loss: %f", np.mean(loss - l2dist))
                            best_lr[loss < best_loss] = learning_rate[loss < best_loss]
                            best_loss[loss < best_loss] = loss[loss < best_loss]

                        learning_rate[halving == 1] /= 2

                        update_adv = best_lr[active] > 0
                        logger.debug("Number of adversarial samples to be finally updated: %i", int(np.sum(update_adv)))

                        if np.sum(update_adv) > 0:
                            active_and_update_adv = active.copy()
                            active_and_update_adv[active] = update_adv
                            best_lr_mult = best_lr[active_and_update_adv]
                            for _ in range(len(x.shape) - 1):
                                best_lr_mult = best_lr_mult[:, np.newaxis]

                            x_adv4 = x_adv_batch_tanh[active_and_update_adv]
                            best_lr1 = best_lr_mult * perturbation_tanh[update_adv]
                            x_adv_batch_tanh[active_and_update_adv] = (
                                x_adv4 + best_lr1 * activation_batch[active_and_update_adv]
                            )

                            x_adv6 = x_adv_batch_tanh[active_and_update_adv]
                            x_adv_batch[active_and_update_adv] = tanh_to_original(x_adv6, clip_min, clip_max)
                            (
                                z_logits[active_and_update_adv],
                                l2dist[active_and_update_adv],
                                loss[active_and_update_adv],
                            ) = self._loss(
                                x_batch[active_and_update_adv],
                                x_adv_batch[active_and_update_adv],
                                y_batch[active_and_update_adv],
                                c_current[active_and_update_adv],
                            )
                            attack_success = loss - l2dist <= 0
                            overall_attack_success = overall_attack_success | attack_success

                    # Update depending on attack success:
                    l0dist = np.sum(
                        (np.abs(x_batch - x_adv_batch) > self._perturbation_threshold).astype(int), axis=(1, 2, 3)
                    )
                    improved_adv = attack_success & (l0dist < best_l0dist_batch)
                    logger.debug("Number of improved L0 distances: %i", int(np.sum(improved_adv)))
                    if np.sum(improved_adv) > 0:
                        best_l0dist_batch[improved_adv] = l0dist[improved_adv]
                        best_x_adv_batch[improved_adv] = x_adv_batch[improved_adv]

                    c_double[overall_attack_success] = False
                    c_current[overall_attack_success] = (c_lower_bound + c_current)[overall_attack_success] / 2

                    c_old = c_current
                    c_current[~overall_attack_success & c_double] *= 2

                    c_current1 = (c_current - c_lower_bound)[~overall_attack_success & ~c_double]
                    c_current[~overall_attack_success & ~c_double] += c_current1 / 2
                    c_lower_bound[~overall_attack_success] = c_old[~overall_attack_success]

                c_final[batch_index_1:batch_index_2] = c_current
                x_adv[batch_index_1:batch_index_2] = best_x_adv_batch

            logger.info(
                "Success rate of C&W L_2 attack: %.2f%%",
                100 * compute_success(self.estimator, x, y, x_adv, self.targeted, batch_size=self.batch_size),
            )

            # If the L_2 attack can't find any adversarial examples with the new activation, return the last one
            z_logits, l2dist, loss = self._loss(x, x_adv, y, c_final)  # type: ignore
            attack_success = loss - l2dist <= 0
            l0dist = np.sum((np.abs(x - x_adv) > self._perturbation_threshold).astype(int), axis=(1, 2, 3))
            improved_adv = attack_success & (l0dist < best_l0dist)
            if np.sum(improved_adv) > 0:
                final_adversarial_example[improved_adv] = x_adv[improved_adv]
            else:
                return x * (old_activation == 0).astype(int) + final_adversarial_example * old_activation

            # Compute the gradients of the objective function evaluated at the adversarial instance
            x_adv_tanh = original_to_tanh(x_adv, clip_min, clip_max, self._tanh_smoother)
            objective_loss_gradient = -self._loss_gradient(
                z_logits,
                y,
                x,
                x_adv,
                x_adv_tanh,
                c_final,
                clip_min,
                clip_max,
            )
            perturbation_l1_norm = np.abs(x_adv - x)

            # gradient * perturbation tells how much reduction to the objective function we obtain for each attribute
            objective_reduction = np.abs(objective_loss_gradient) * perturbation_l1_norm

            # Assign infinity as the objective_reduction value for fixed feature (in order not to select them again)
            objective_reduction += np.array(np.where(activation == 0, np.inf, 0))

            # Fix the feature with the lowest objective_reduction value (only for the examples that succeeded)
            fix_feature_index = np.argmin(objective_reduction.reshape(objective_reduction.shape[0], -1), axis=1)
            fix_feature = np.ones(x.shape)
            fix_feature = fix_feature.reshape(fix_feature.shape[0], -1)
            fix_feature[np.arange(fix_feature_index.size), fix_feature_index] = 0
            fix_feature = fix_feature.reshape(x.shape)
            old_activation[improved_adv] = activation.copy()[improved_adv]
            activation[improved_adv] *= fix_feature[improved_adv]
            logger.info(
                "L0 norm before fixing :\n%f\nNumber active features :\n%f\nIndex of fixed feature :\n%d",
                np.sum((perturbation_l1_norm > self._perturbation_threshold).astype(int), axis=1),
                np.sum(activation, axis=1),
                fix_feature_index,
            )
        return x_adv

    def _check_params(self):

        if not isinstance(self.binary_search_steps, int) or self.binary_search_steps < 0:
            raise ValueError("The number of binary search steps must be a non-negative integer.")
