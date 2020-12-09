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

from art.attacks.attack import InferenceAttack
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import check_and_transform_label_format

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE

logger = logging.getLogger(__name__)


class MembershipInferenceBlackBoxRuleBased(InferenceAttack):
    """
        Implementation of a simple, rule-based black-box membership inference attack.

        This implementation uses the simple rule: if the model's prediction for a sample is correct, then it is a
        member. Otherwise, it is not a member.
    """

    attack_params = InferenceAttack.attack_params
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
        :return: An array holding the inferred membership status, 1 indicates a member and 0 indicates non-member.
        """
        if y is None:
            raise ValueError("MembershipInferenceBlackBoxRuleBased requires true labels `y`.")

        if self.estimator.input_shape[0] != x.shape[1]:
            raise ValueError("Shape of x does not match input_shape of classifier")

        y = check_and_transform_label_format(y, len(np.unique(y)), return_one_hot=True)
        if y.shape[0] != x.shape[0]:
            raise ValueError("Number of rows in x and y do not match")

        # get model's predictions for x
        y_pred = self.estimator.predict(x=x)
        return (np.argmax(y, axis=1) == np.argmax(y_pred, axis=1)).astype(np.int)
