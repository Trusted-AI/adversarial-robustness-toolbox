# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2021
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import minmax_scale

from art.estimators.classification.classifier import ClassifierMixin
from art.attacks.attack import AttributeInferenceAttack
from art.utils import (
    check_and_transform_label_format,
    float_to_categorical,
    floats_to_one_hot,
    get_feature_values,
    get_feature_index,
)

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE

logger = logging.getLogger(__name__)


class AttributeInferenceBaselineTrueLabel(AttributeInferenceAttack):
    """
    Implementation of a baseline attribute inference, not using a model.

    The idea is to train a simple neural network to learn the attacked feature from the rest of the features, and the
    true label. Should be used to compare with other attribute inference results.
    """

    _estimator_requirements = ()

    def __init__(
        self,
        attack_model_type: str = "nn",
        attack_model: Optional["CLASSIFIER_TYPE"] = None,
        attack_feature: Union[int, slice] = 0,
        scale_range: Optional[slice] = None,
        prediction_normal_factor: float = 1,
    ):
        """
        Create an AttributeInferenceBaseline attack instance.

        :param attack_model_type: the type of default attack model to train, optional. Should be one of `nn` (for neural
                                  network, default) or `rf` (for random forest). If `attack_model` is supplied, this
                                  option will be ignored.
        :param attack_model: The attack model to train, optional. If none is provided, a default model will be created.
        :param attack_feature: The index of the feature to be attacked or a slice representing multiple indexes in
                               case of a one-hot encoded feature.
                               case of a one-hot encoded feature.
        :param scale_range: If supplied, the class labels (both true and predicted) will be scaled to the given range.
                            Only applicable when `estimator` is a regressor.
        :param prediction_normal_factor: If supplied, the class labels (both true and predicted) are multiplied by the
                                         factor when used as inputs to the attack-model. Only applicable when
                                         `estimator` is a regressor and if `scale_range` is not supplied.
        """
        super().__init__(estimator=None, attack_feature=attack_feature)

        self._values: Optional[list] = None

        if attack_model:
            if ClassifierMixin not in type(attack_model).__mro__:
                raise ValueError("Attack model must be of type Classifier.")
            self.attack_model = attack_model
        elif attack_model_type == "nn":
            self.attack_model = MLPClassifier(
                hidden_layer_sizes=(100,),
                activation="relu",
                solver="adam",
                alpha=0.0001,
                batch_size="auto",
                learning_rate="constant",
                learning_rate_init=0.001,
                power_t=0.5,
                max_iter=2000,
                shuffle=True,
                random_state=None,
                tol=0.0001,
                verbose=False,
                warm_start=False,
                momentum=0.9,
                nesterovs_momentum=True,
                early_stopping=False,
                validation_fraction=0.1,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-08,
                n_iter_no_change=10,
                max_fun=15000,
            )
        elif attack_model_type == "rf":
            self.attack_model = RandomForestClassifier()
        else:
            raise ValueError("Illegal value for parameter `attack_model_type`.")

        self.prediction_normal_factor = prediction_normal_factor
        self.scale_range = scale_range
        self._check_params()
        self.attack_feature = get_feature_index(self.attack_feature)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Train the attack model.

        :param x: Input to training process. Includes all features used to train the original model.
        :param y: True labels of the features.
        """

        # Checks:
        if isinstance(self.attack_feature, int) and self.attack_feature >= x.shape[1]:
            raise ValueError("attack_feature must be a valid index to a feature in x")

        # get vector of attacked feature
        attacked_feature = x[:, self.attack_feature]
        self._values = get_feature_values(attacked_feature, isinstance(self.attack_feature, int))
        if isinstance(self.attack_feature, int):
            y_one_hot = float_to_categorical(attacked_feature)
        else:
            y_one_hot = floats_to_one_hot(attacked_feature)
        y_ready = check_and_transform_label_format(y_one_hot, len(np.unique(attacked_feature)), return_one_hot=True)
        if y_ready is None:
            raise ValueError("None value detected.")

        # create training set for attack model
        if self.scale_range is not None:
            normalized_labels = minmax_scale(y, feature_range=self.scale_range)
        else:
            normalized_labels = y * self.prediction_normal_factor
        normalized_labels = check_and_transform_label_format(normalized_labels, return_one_hot=True)
        x_train = np.concatenate((np.delete(x, self.attack_feature, 1), normalized_labels), axis=1).astype(np.float32)

        # train attack model
        self.attack_model.fit(x_train, y_ready)

    def infer(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Infer the attacked feature.

        :param x: Input to attack. Includes all features except the attacked feature.
        :param y: True labels of the features.
        :param values: Possible values for attacked feature. For a single column feature this should be a simple list
                       containing all possible values, in increasing order (the smallest value in the 0 index and so
                       on). For a multi-column feature (for example 1-hot encoded and then scaled), this should be a
                       list of lists, where each internal list represents a column (in increasing order) and the values
                       represent the possible values for that column (in increasing order).
        :type values: list
        :return: The inferred feature values.
        """
        if y is None:
            raise ValueError("True labels are required")

        values = kwargs.get("values")

        # if provided, override the values computed in fit()
        if values is not None:
            self._values = values

        if self.scale_range is not None:
            normalized_labels = minmax_scale(y, feature_range=self.scale_range)
        else:
            normalized_labels = y * self.prediction_normal_factor
        normalized_labels = check_and_transform_label_format(normalized_labels, return_one_hot=True)
        x_test = np.concatenate((x, normalized_labels), axis=1).astype(np.float32)

        predictions = self.attack_model.predict(x_test).astype(np.float32)

        if self._values is not None:
            if isinstance(self.attack_feature, int):
                predictions = np.array([self._values[np.argmax(arr)] for arr in predictions])
            else:
                i = 0
                for column in predictions.T:
                    for index in range(len(self._values[i])):
                        np.place(column, [column == index], self._values[i][index])
                    i += 1
        return np.array(predictions)

    def _check_params(self) -> None:
        if not isinstance(self.attack_feature, int) and not isinstance(self.attack_feature, slice):
            raise ValueError("Attack feature must be either an integer or a slice object.")

        if isinstance(self.attack_feature, int) and self.attack_feature < 0:
            raise ValueError("Attack feature index must be positive.")
