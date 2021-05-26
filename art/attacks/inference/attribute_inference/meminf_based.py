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
This module implements attribute inference attacks.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, Union, TYPE_CHECKING

import numpy as np
from sklearn.neural_network import MLPClassifier

from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin
from art.attacks.attack import AttributeInferenceAttack, MembershipInferenceAttack
from art.utils import check_and_transform_label_format, float_to_categorical, floats_to_one_hot

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE

logger = logging.getLogger(__name__)


class AttributeInferenceUsingMembershipInference(AttributeInferenceAttack):
    """
    Implementation of a an attribute inference attack that utilizes a membership inference attack.

    The idea is to find the target feature value that causes the membership inference attack to classify the sample
    as a member with the highest confidence.
    """

    _estimator_requirements = (BaseEstimator, ClassifierMixin)

    def __init__(
        self,
        classifier: "CLASSIFIER_TYPE",
        membership_attack: MembershipInferenceAttack,
        attack_feature: Union[int, slice] = 0,
    ):
        """
        Create an AttributeInferenceUsingMembershipInference attack instance.

        :param classifier: Target classifier.
        :param membership_attack: The membership inference attack to use. Should be fit/callibrated in advance, and
                                  should support returning probabilities.
        :param attack_feature: The index of the feature to be attacked or a slice representing multiple indexes in
                               case of a one-hot encoded feature.
        """
        super().__init__(estimator=classifier, attack_feature=attack_feature)
        self._estimator_requirements = self._estimator_requirements + membership_attack._estimator_requirements

        if isinstance(self.attack_feature, int):
            self.single_index_feature = True
        else:
            self.single_index_feature = False

        self.membership_attack = membership_attack

        self._check_params()

    def infer(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Infer the attacked feature.

        :param x: Input to attack. Includes all features except the attacked feature.
        :param y: The labels expected by the membership attack.
        :param values: Possible values for attacked feature. For a single column feature this should be a simple list
                       containing all possible values, in increasing order (the smallest value in the 0 index and so
                       on). For a multi-column feature (for example 1-hot encoded and then scaled), this should be a
                       list of lists, where each internal list represents a column (in increasing order) and the values
                       represent the possible values for that column (in increasing order).
        :type values: list
        :return: The inferred feature values.
        """
        if self.estimator.input_shape is not None:
            if self.single_index_feature and self.estimator.input_shape[0] != x.shape[1] + 1:
                raise ValueError("Number of features in x + 1 does not match input_shape of classifier")

        if "values" not in kwargs.keys():
            raise ValueError("Missing parameter `values`.")
        values: list = kwargs.get("values")

        if y is not None:
            if y.shape[0] != x.shape[0]:
                raise ValueError("Number of rows in x and y do not match")

        # assumes single index
        if self.single_index_feature:
            first = True
            for value in values:
                v_full = np.full((x.shape[0], 1), value)
                x_value = np.concatenate((x[:, : self.attack_feature], v_full), axis=1)
                x_value = np.concatenate((x_value, x[:, self.attack_feature:]), axis=1)

                predicted = self.membership_attack.infer(x_value, y, probabilities=True)
                if first:
                    probabilities = predicted[:, 1].reshape(-1, 1)
                    first = False
                else:
                    probabilities = np.hstack((probabilities, predicted[:, 1].reshape(-1, 1)))

            # needs to be of type float so we can later replace back the actual values
            value_indexes = np.argmax(probabilities, axis=1).astype(np.float32)
            for index, value in enumerate(values):
                value_indexes[value_indexes == index] = value
            return value_indexes
        else: # 1-hot encoded feature. Can also be scaled.
            first = True
            # assumes that the second value is the "positive" value and that there can only be one positive column
            for index in range(len(values)):
                curr_value = np.zeros(len(values),)
                curr_value[index] = values[index][1]
                for not_index in range(len(values)):
                    if not_index != index:
                        curr_value[not_index] = values[not_index][0]
                x_value = np.concatenate((x[:, : self.attack_feature], curr_value), axis=1)
                x_value = np.concatenate((x_value, x[:, self.attack_feature:]), axis=1)

                predicted = self.membership_attack.infer(x_value, y, probabilities=True)
                if first:
                    probabilities = predicted[:, 1]
                else:
                    probabilities = np.stack((probabilities, predicted[:, 1]), axis=1)
                first = False
            i = 0
            for column in probabilities.T:
                for index in range(len(values[i])):
                    np.place(column, [column == index], values[i][index])
                i += 1
            return probabilities

    def _check_params(self) -> None:
        if not isinstance(self.attack_feature, int) and not isinstance(self.attack_feature, slice):
            raise ValueError("Attack feature must be either an integer or a slice object.")
        if isinstance(self.attack_feature, int) and self.attack_feature < 0:
            raise ValueError("Attack feature index must be positive.")
        if not isinstance(self.membership_attack, MembershipInferenceAttack):
            raise ValueError("membership_attack should be a sub-class of MembershipInferenceAttack")
