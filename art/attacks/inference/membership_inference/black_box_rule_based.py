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
This module implements membership inference attacks.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

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


class MembershipInferenceBlackBoxRuleBased(MembershipInferenceAttack):
    """
    Implementation of a simple, rule-based black-box membership inference attack.

    This implementation uses the simple rule: if the model's prediction for a sample is correct, then it is a
    member. Otherwise, it is not a member.
    """

    attack_params = MembershipInferenceAttack.attack_params
    _estimator_requirements = (BaseEstimator, ClassifierMixin)

    def __init__(self, classifier: "CLASSIFIER_TYPE"):
        """
        Create a MembershipInferenceBlackBoxRuleBased attack instance.

        :param classifier: Target classifier.
        """
        super().__init__(estimator=classifier)

    def infer(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Infer membership in the training set of the target estimator.

        :param x: Input records to attack.
        :param y: True labels for `x`.
        :param probabilities: a boolean indicating whether to return the predicted probabilities per class, or just
                              the predicted class.
        :return: An array holding the inferred membership status, 1 indicates a member and 0 indicates non-member,
                 or class probabilities.
        """
        if y is None:  # pragma: no cover
            raise ValueError("MembershipInferenceBlackBoxRuleBased requires true labels `y`.")

        if self.estimator.input_shape is not None:  # pragma: no cover
            if self.estimator.input_shape[0] != x.shape[1]:
                raise ValueError("Shape of x does not match input_shape of classifier")

        if "probabilities" in kwargs.keys():
            probabilities = kwargs.get("probabilities")
        else:
            probabilities = False

        y = check_and_transform_label_format(y, len(np.unique(y)), return_one_hot=True)
        if y.shape[0] != x.shape[0]:  # pragma: no cover
            raise ValueError("Number of rows in x and y do not match")

        # get model's predictions for x
        y_pred = self.estimator.predict(x=x)
        predicted_class = (np.argmax(y, axis=1) == np.argmax(y_pred, axis=1)).astype(np.int)
        if probabilities:
            # use y_pred as the probability if binary classification, otherwise just use 1
            if y_pred.shape[1] == 2:
                pred_prob = np.max(y_pred, axis=1)
                prob = np.zeros((predicted_class.shape[0], 2))
                prob[:, predicted_class] = pred_prob
                prob[:, np.ones_like(predicted_class) - predicted_class] = np.ones_like(pred_prob) - pred_prob
            else:
                # simply returns probability 1 for the predicted class and 0 for the other class
                prob = check_and_transform_label_format(predicted_class, return_one_hot=True)
            return prob
        return predicted_class
