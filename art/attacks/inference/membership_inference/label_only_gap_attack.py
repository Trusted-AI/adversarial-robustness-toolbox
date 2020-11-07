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
This module implements the Label Only Gap Attack `.

| Paper link: https://arxiv.org/abs/2007.14321
"""
import logging
from typing import TYPE_CHECKING

import numpy as np

from art.attacks.attack import InferenceAttack
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE

logger = logging.getLogger(__name__)


class LabelOnlyGapAttack(InferenceAttack):
    """
    Implementation of Label Only Gap Attack.

    | Paper link: https://arxiv.org/abs/2007.14321
    """

    attack_params = InferenceAttack.attack_params
    _estimator_requirements = (BaseEstimator, ClassifierMixin)

    def __init__(self, estimator: "CLASSIFIER_TYPE"):
        """
        Create a `LabelOnlyGapAttack` instance of a Label Only Gap Attack.

        :param estimator: A trained classification estimator.
        """
        super().__init__(estimator=estimator)
        self._check_params()

    def infer(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Infer membership of input `x` in estimator's training data.

        :param x: Input data.
        :param y: True labels for `x`.
        :return: An array holding the inferred membership status, 1 indicates a member and 0 indicates non-member.
        """
        y_pred = self.estimator.predict(x=x)
        return np.argmax(y, axis=1) == np.argmax(y_pred, axis=1)

    def _check_params(self) -> None:
        pass
