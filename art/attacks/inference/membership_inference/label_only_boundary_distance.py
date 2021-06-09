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
This module implements the Label-Only Inference Attack based on Decision Boundary.

| Paper link: https://arxiv.org/abs/2007.14321
"""
import logging
from typing import Optional, TYPE_CHECKING

import numpy as np

from art.attacks.attack import MembershipInferenceAttack
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import check_and_transform_label_format

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE

logger = logging.getLogger(__name__)


class LabelOnlyDecisionBoundary(MembershipInferenceAttack):
    """
    Implementation of Label-Only Inference Attack based on Decision Boundary.

    | Paper link: https://arxiv.org/abs/2007.14321
    """

    attack_params = MembershipInferenceAttack.attack_params + [
        "distance_threshold_tau",
    ]
    _estimator_requirements = (BaseEstimator, ClassifierMixin)

    def __init__(self, estimator: "CLASSIFIER_TYPE", distance_threshold_tau: Optional[float] = None):
        """
        Create a `LabelOnlyDecisionBoundary` instance for Label-Only Inference Attack based on Decision Boundary.

        :param estimator: A trained classification estimator.
        :param distance_threshold_tau: Threshold distance for decision boundary. Samples with boundary distances larger
                                       than threshold are considered members of the training dataset.
        """
        super().__init__(estimator=estimator)
        self.distance_threshold_tau = distance_threshold_tau
        self.threshold_bins: list = []
        self._check_params()

    def infer(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Infer membership of input `x` in estimator's training data.

        :param x: Input data.
        :param y: True labels for `x`.
        :param probabilities: a boolean indicating whether to return the predicted probabilities per class, or just
                              the predicted class

        :Keyword Arguments for HopSkipJump:
            * *norm*: Order of the norm. Possible values: "inf", np.inf or 2.
            * *max_iter*: Maximum number of iterations.
            * *max_eval*: Maximum number of evaluations for estimating gradient.
            * *init_eval*: Initial number of evaluations for estimating gradient.
            * *init_size*: Maximum number of trials for initial generation of adversarial examples.
            * *verbose*: Show progress bars.

        :return: An array holding the inferred membership status, 1 indicates a member and 0 indicates non-member,
                 or class probabilities.
        """
        from art.attacks.evasion.hop_skip_jump import HopSkipJump

        if y is None:
            raise ValueError("Argument `y` is None, but this attack requires true labels `y` to be provided.")

        if self.distance_threshold_tau is None:
            raise ValueError(
                "No value for distance threshold `distance_threshold_tau` provided. Please set"
                "`distance_threshold_tau` or run method `calibrate_distance_threshold` on known training and test"
                "dataset."
            )

        if "probabilities" in kwargs.keys():
            probabilities = kwargs.get("probabilities")
            del kwargs["probabilities"]
        else:
            probabilities = False

        if "classifier" in kwargs:
            raise ValueError("Keyword `classifier` in kwargs is not supported.")

        if "targeted" in kwargs:
            raise ValueError("Keyword `targeted` in kwargs is not supported.")

        y = check_and_transform_label_format(y, self.estimator.nb_classes)

        hsj = HopSkipJump(classifier=self.estimator, targeted=False, **kwargs)
        x_adv = hsj.generate(x=x, y=y)

        distance = np.linalg.norm((x_adv - x).reshape((x.shape[0], -1)), ord=2, axis=1)

        y_pred = self.estimator.predict(x=x)

        distance[np.argmax(y_pred, axis=1) != np.argmax(y, axis=1)] = 0
        predicted_class = np.where(distance > self.distance_threshold_tau, 1, 0)
        if probabilities:
            prob_1 = np.zeros_like(distance)
            if self.threshold_bins:
                # bin accuracy is the probability of being a member
                for t_bin in self.threshold_bins:
                    prob_1[distance > t_bin[0]] = t_bin[1]
            else:
                # use sigmoid on distance from threshold
                dist_threshold = distance - self.distance_threshold_tau
                prob_1 = 1 / (1 + np.exp(-dist_threshold))
            prob_0 = np.ones_like(prob_1) - prob_1
            return np.stack((prob_0, prob_1), axis=1)
        return predicted_class

    def calibrate_distance_threshold(
        self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, **kwargs
    ):
        """
        Calibrate distance threshold maximising the membership inference accuracy on `x_train` and `x_test`.

        :param x_train: Training data.
        :param y_train: Labels of training data `x_train`.
        :param x_test: Test data.
        :param y_test: Labels of test data `x_test`.

        :Keyword Arguments for HopSkipJump:
            * *norm*: Order of the norm. Possible values: "inf", np.inf or 2.
            * *max_iter*: Maximum number of iterations.
            * *max_eval*: Maximum number of evaluations for estimating gradient.
            * *init_eval*: Initial number of evaluations for estimating gradient.
            * *init_size*: Maximum number of trials for initial generation of adversarial examples.
            * *verbose*: Show progress bars.
        """
        from art.attacks.evasion.hop_skip_jump import HopSkipJump

        if "classifier" in kwargs:
            raise ValueError("Keyword `classifier` in kwargs is not supported.")

        if "targeted" in kwargs:
            raise ValueError("Keyword `targeted` in kwargs is not supported.")

        y_train = check_and_transform_label_format(y_train, self.estimator.nb_classes)
        y_test = check_and_transform_label_format(y_test, self.estimator.nb_classes)

        hsj = HopSkipJump(classifier=self.estimator, targeted=False, **kwargs)

        x_train_adv = hsj.generate(x=x_train, y=y_train)
        x_test_adv = hsj.generate(x=x_test, y=y_test)

        distance_train = np.linalg.norm((x_train_adv - x_train).reshape((x_train.shape[0], -1)), ord=2, axis=1)
        distance_test = np.linalg.norm((x_test_adv - x_test).reshape((x_test.shape[0], -1)), ord=2, axis=1)

        y_train_pred = self.estimator.predict(x=x_train)
        y_test_pred = self.estimator.predict(x=x_test)

        distance_train[np.argmax(y_train_pred, axis=1) != np.argmax(y_train, axis=1)] = 0
        distance_test[np.argmax(y_test_pred, axis=1) != np.argmax(y_test, axis=1)] = 0

        num_increments = 100
        tau_increment = np.amax([np.amax(distance_train), np.amax(distance_test)]) / num_increments

        acc_max = 0.0
        distance_threshold_tau = 0.0
        self.threshold_bins = []
        for i_tau in range(1, num_increments):
            # searching for threshold that yields the best accuracy in separating between members and non-members
            is_member_train = np.where(distance_train > i_tau * tau_increment, 1, 0)
            is_member_test = np.where(distance_test > i_tau * tau_increment, 1, 0)

            acc = (np.sum(is_member_train) + (is_member_test.shape[0] - np.sum(is_member_test))) / (
                is_member_train.shape[0] + is_member_test.shape[0]
            )
            new_threshold_tau = i_tau * tau_increment
            self.threshold_bins.append((new_threshold_tau, acc))
            if acc > acc_max:
                distance_threshold_tau = new_threshold_tau
                acc_max = acc

        self.distance_threshold_tau = distance_threshold_tau

    def _check_params(self) -> None:
        if self.distance_threshold_tau is not None and (
            not isinstance(self.distance_threshold_tau, (int, float)) or self.distance_threshold_tau <= 0.0
        ):
            raise ValueError("The distance threshold `distance_threshold_tau` needs to be a positive float.")
