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
This module implements the elastic net attack `ElasticNet`. This is a white-box attack.

| Paper link: https://arxiv.org/abs/1709.04114
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np
import six
from tqdm.auto import trange

from art.config import ART_NUMPY_DTYPE
from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassGradientsMixin
from art.utils import (
    compute_success,
    get_labels_np_array,
    check_and_transform_label_format,
)

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE

logger = logging.getLogger(__name__)


class ElasticNet(EvasionAttack):
    """
    The elastic net attack of Pin-Yu Chen et al. (2018).

    | Paper link: https://arxiv.org/abs/1709.04114
    """

    attack_params = EvasionAttack.attack_params + [
        "confidence",
        "targeted",
        "learning_rate",
        "max_iter",
        "beta",
        "binary_search_steps",
        "initial_const",
        "batch_size",
        "decision_rule",
        "verbose",
    ]

    _estimator_requirements = (BaseEstimator, ClassGradientsMixin)

    def __init__(
        self,
        classifier: "CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE",
        confidence: float = 0.0,
        targeted: bool = False,
        learning_rate: float = 1e-2,
        binary_search_steps: int = 9,
        max_iter: int = 100,
        beta: float = 1e-3,
        initial_const: float = 1e-3,
        batch_size: int = 1,
        decision_rule: str = "EN",
        verbose: bool = True,
    ) -> None:
        """
        Create an ElasticNet attack instance.

        :param classifier: A trained classifier.
        :param confidence: Confidence of adversarial examples: a higher value produces examples that are farther
               away, from the original input, but classified with higher confidence as the target class.
        :param targeted: Should the attack target one specific class.
        :param learning_rate: The initial learning rate for the attack algorithm. Smaller values produce better
               results but are slower to converge.
        :param binary_search_steps: Number of times to adjust constant with binary search (positive value).
        :param max_iter: The maximum number of iterations.
        :param beta: Hyperparameter trading off L2 minimization for L1 minimization.
        :param initial_const: The initial trade-off constant `c` to use to tune the relative importance of distance
               and confidence. If `binary_search_steps` is large, the initial constant is not important, as discussed in
               Carlini and Wagner (2016).
        :param batch_size: Internal size of batches on which adversarial samples are generated.
        :param decision_rule: Decision rule. 'EN' means Elastic Net rule, 'L1' means L1 rule, 'L2' means L2 rule.
        :param verbose: Show progress bars.
        """
        super().__init__(estimator=classifier)
        self.confidence = confidence
        self._targeted = targeted
        self.learning_rate = learning_rate
        self.binary_search_steps = binary_search_steps
        self.max_iter = max_iter
        self.beta = beta
        self.initial_const = initial_const
        self.batch_size = batch_size
        self.decision_rule = decision_rule
        self.verbose = verbose
        self._check_params()

    def _loss(self, x: np.ndarray, x_adv: np.ndarray) -> tuple:
        """
        Compute the loss function values.

        :param x: An array with the original input.
        :param x_adv: An array with the adversarial input.
        :return: A tuple of shape `(np.ndarray, float, float, float)` holding the current predictions, l1 distance,
                 l2 distance and elastic net loss.
        """
        l1dist = np.sum(np.abs(x - x_adv).reshape(x.shape[0], -1), axis=1)
        l2dist = np.sum(np.square(x - x_adv).reshape(x.shape[0], -1), axis=1)
        endist = self.beta * l1dist + l2dist
        predictions = self.estimator.predict(np.array(x_adv, dtype=ART_NUMPY_DTYPE), batch_size=self.batch_size)

        return np.argmax(predictions, axis=1), l1dist, l2dist, endist

    def _gradient_of_loss(
        self,
        target: np.ndarray,
        x: np.ndarray,
        x_adv: np.ndarray,
        c_weight: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the gradient of the loss function.

        :param target: An array with the target class (one-hot encoded).
        :param x: An array with the original input.
        :param x_adv: An array with the adversarial input.
        :param c_weight: Weight of the loss term aiming for classification as target.
        :return: An array with the gradient of the loss function.
        """
        # Compute the current predictions
        predictions = self.estimator.predict(np.array(x_adv, dtype=ART_NUMPY_DTYPE), batch_size=self.batch_size)

        if self.targeted:
            i_sub = np.argmax(target, axis=1)
            i_add = np.argmax(
                predictions * (1 - target) + (np.min(predictions, axis=1) - 1)[:, np.newaxis] * target,
                axis=1,
            )
        else:
            i_add = np.argmax(target, axis=1)
            i_sub = np.argmax(
                predictions * (1 - target) + (np.min(predictions, axis=1) - 1)[:, np.newaxis] * target,
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

        # Set gradients where loss is constant to zero
        cond = (
            predictions[np.arange(x.shape[0]), i_add] - predictions[np.arange(x.shape[0]), i_sub] + self.confidence
        ) < 0
        loss_gradient[cond] = 0.0

        return loss_gradient

    def _decay_learning_rate(self, global_step: int, end_learning_rate: float, decay_steps: int) -> float:
        """
        Applies a square-root decay to the learning rate.

        :param global_step: Global step to use for the decay computation.
        :param end_learning_rate: The minimal end learning rate.
        :param decay_steps: Number of decayed steps.
        :return: The decayed learning rate
        """
        learn_rate = self.learning_rate - end_learning_rate
        decayed_learning_rate = learn_rate * (1 - global_step / decay_steps) ** 2 + end_learning_rate

        return decayed_learning_rate

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs to be attacked.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,). If `self.targeted` is true, then `y` represents the target labels. Otherwise, the
                  targets are the original class labels.
        :return: An array holding the adversarial examples.
        """
        y = check_and_transform_label_format(y, self.estimator.nb_classes)
        x_adv = x.astype(ART_NUMPY_DTYPE)

        # Assert that, if attack is targeted, y is provided:
        if self.targeted and y is None:  # pragma: no cover
            raise ValueError("Target labels `y` need to be provided for a targeted attack.")

        # No labels provided, use model prediction as correct class
        if y is None:
            y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))

        if self.estimator.nb_classes == 2 and y.shape[1] == 1:  # pragma: no cover
            raise ValueError(
                "This attack has not yet been tested for binary classification with a single output classifier."
            )

        # Compute adversarial examples with implicit batching
        nb_batches = int(np.ceil(x_adv.shape[0] / float(self.batch_size)))
        for batch_id in trange(nb_batches, desc="EAD", disable=not self.verbose):
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            x_batch = x_adv[batch_index_1:batch_index_2]
            y_batch = y[batch_index_1:batch_index_2]
            x_adv[batch_index_1:batch_index_2] = self._generate_batch(x_batch, y_batch)

        # Apply clip
        if self.estimator.clip_values is not None:
            x_adv = np.clip(x_adv, self.estimator.clip_values[0], self.estimator.clip_values[1])

        # Compute success rate of the EAD attack
        logger.info(
            "Success rate of EAD attack: %.2f%%",
            100 * compute_success(self.estimator, x, y, x_adv, self.targeted, batch_size=self.batch_size),
        )

        return x_adv

    def _generate_batch(self, x_batch: np.ndarray, y_batch: np.ndarray) -> np.ndarray:
        """
        Run the attack on a batch of images and labels.

        :param x_batch: A batch of original examples.
        :param y_batch: A batch of targets (0-1 hot).
        :return: A batch of adversarial examples.
        """
        # Initialize binary search:
        c_current = self.initial_const * np.ones(x_batch.shape[0])
        c_lower_bound = np.zeros(x_batch.shape[0])
        c_upper_bound = 10e10 * np.ones(x_batch.shape[0])

        # Initialize best distortions and best attacks globally
        o_best_dist = np.inf * np.ones(x_batch.shape[0])
        o_best_attack = x_batch.copy()

        # Start with a binary search
        for bss in range(self.binary_search_steps):
            logger.debug(
                "Binary search step %i out of %i (c_mean==%f)",
                bss,
                self.binary_search_steps,
                np.mean(c_current),
            )

            # Run with 1 specific binary search step
            best_dist, best_label, best_attack = self._generate_bss(x_batch, y_batch, c_current)

            # Update best results so far
            o_best_attack[best_dist < o_best_dist] = best_attack[best_dist < o_best_dist]
            o_best_dist[best_dist < o_best_dist] = best_dist[best_dist < o_best_dist]

            # Adjust the constant as needed
            c_current, c_lower_bound, c_upper_bound = self._update_const(
                y_batch, best_label, c_current, c_lower_bound, c_upper_bound
            )

        return o_best_attack

    def _update_const(
        self,
        y_batch: np.ndarray,
        best_label: np.ndarray,
        c_batch: np.ndarray,
        c_lower_bound: np.ndarray,
        c_upper_bound: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Update constants.

        :param y_batch: A batch of targets (0-1 hot).
        :param best_label: A batch of best labels.
        :param c_batch: A batch of constants.
        :param c_lower_bound: A batch of lower bound constants.
        :param c_upper_bound: A batch of upper bound constants.
        :return: A tuple of three batches of updated constants and lower/upper bounds.
        """

        def compare(o_1, o_2):
            if self.targeted:
                return o_1 == o_2
            return o_1 != o_2

        for i in range(c_batch.shape[0]):
            if compare(best_label[i], np.argmax(y_batch[i])) and best_label[i] != -np.inf:
                # Successful attack
                c_upper_bound[i] = min(c_upper_bound[i], c_batch[i])
                if c_upper_bound[i] < 1e9:
                    c_batch[i] = (c_lower_bound[i] + c_upper_bound[i]) / 2.0

            else:
                # Failure attack
                c_lower_bound[i] = max(c_lower_bound[i], c_batch[i])
                if c_upper_bound[i] < 1e9:
                    c_batch[i] = (c_lower_bound[i] + c_upper_bound[i]) / 2.0
                else:
                    c_batch[i] *= 10

        return c_batch, c_lower_bound, c_upper_bound

    def _generate_bss(self, x_batch: np.ndarray, y_batch: np.ndarray, c_batch: np.ndarray) -> tuple:
        """
        Generate adversarial examples for a batch of inputs with a specific batch of constants.

        :param x_batch: A batch of original examples.
        :param y_batch: A batch of targets (0-1 hot).
        :param c_batch: A batch of constants.
        :return: A tuple of best elastic distances, best labels, best attacks
        """

        def compare(o_1, o_2):
            if self.targeted:
                return o_1 == o_2
            return o_1 != o_2

        # Initialize best distortions and best changed labels and best attacks
        best_dist = np.inf * np.ones(x_batch.shape[0])
        best_label = [-np.inf] * x_batch.shape[0]
        best_attack = x_batch.copy()

        # Implement the algorithm 1 in the EAD paper
        x_adv = x_batch.copy()
        y_adv = x_batch.copy()
        for i_iter in range(self.max_iter):
            logger.debug("Iteration step %i out of %i", i_iter, self.max_iter)

            # Update learning rate
            learning_rate = self._decay_learning_rate(
                global_step=i_iter, end_learning_rate=0, decay_steps=self.max_iter
            )

            # Compute adversarial examples
            grad = self._gradient_of_loss(target=y_batch, x=x_batch, x_adv=y_adv, c_weight=c_batch)
            x_adv_next = self._shrinkage_threshold(y_adv - learning_rate * grad, x_batch, self.beta)
            y_adv = x_adv_next + (1.0 * i_iter / (i_iter + 3)) * (x_adv_next - x_adv)
            x_adv = x_adv_next

            # Adjust the best result
            (logits, l1dist, l2dist, endist) = self._loss(x=x_batch, x_adv=x_adv)

            if self.decision_rule == "EN":
                zip_set = zip(endist, logits)
            elif self.decision_rule == "L1":
                zip_set = zip(l1dist, logits)
            elif self.decision_rule == "L2":
                zip_set = zip(l2dist, logits)
            else:  # pragma: no cover
                raise ValueError("The decision rule only supports `EN`, `L1`, `L2`.")

            for j, (distance, label) in enumerate(zip_set):
                if distance < best_dist[j] and compare(label, np.argmax(y_batch[j])):
                    best_dist[j] = distance
                    best_attack[j] = x_adv[j]
                    best_label[j] = label

        return best_dist, best_label, best_attack

    @staticmethod
    def _shrinkage_threshold(z_batch: np.ndarray, x_batch: np.ndarray, beta: float) -> np.ndarray:
        """
        Implement the element-wise projected shrinkage-threshold function.

        :param z_batch: A batch of examples.
        :param x_batch: A batch of original examples.
        :param beta: The shrink parameter.
        :return: A shrinked version of z.
        """
        cond1 = (z_batch - x_batch) > beta
        cond2 = np.abs(z_batch - x_batch) <= beta
        cond3 = (z_batch - x_batch) < -beta

        upper = np.minimum(z_batch - beta, 1.0)
        lower = np.maximum(z_batch + beta, 0.0)
        result = cond1 * upper + cond2 * x_batch + cond3 * lower

        return result

    def _check_params(self) -> None:
        if not isinstance(self.binary_search_steps, int) or self.binary_search_steps < 0:
            raise ValueError("The number of binary search steps must be a non-negative integer.")

        if not isinstance(self.max_iter, int) or self.max_iter < 0:
            raise ValueError("The number of iterations must be a non-negative integer.")

        if not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError("The batch size must be an integer greater than zero.")

        if not isinstance(self.decision_rule, six.string_types) or self.decision_rule not in ["EN", "L1", "L2"]:
            raise ValueError("The decision rule only supports `EN`, `L1`, `L2`.")

        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be of type bool.")
