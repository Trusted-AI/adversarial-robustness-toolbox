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
from typing import Optional, Union, List, TYPE_CHECKING

import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

from art.estimators.classification.classifier import ClassifierMixin
from art.estimators.regression import RegressorMixin
from art.attacks.attack import AttributeInferenceAttack
from art.utils import (
    check_and_transform_label_format,
    float_to_categorical,
    floats_to_one_hot,
    get_feature_values,
    remove_attacked_feature,
)

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE

logger = logging.getLogger(__name__)


class AttributeInferenceBaseline(AttributeInferenceAttack):
    """
    Implementation of a baseline attribute inference, not using a model.

    The idea is to train a simple neural network to learn the attacked feature from the rest of the features. Should
    be used to compare with other attribute inference results.
    """

    attack_params = AttributeInferenceAttack.attack_params + [
        "attack_model_type",
        "is_continuous",
        "non_numerical_features",
        "encoder",
    ]
    _estimator_requirements = ()

    def __init__(
        self,
        attack_model_type: str = "nn",
        attack_model: Optional["CLASSIFIER_TYPE"] = None,
        attack_feature: Union[int, slice] = 0,
        is_continuous: Optional[bool] = False,
        non_numerical_features: Optional[List[int]] = None,
        encoder: Optional[Union[OrdinalEncoder, OneHotEncoder, ColumnTransformer]] = None,
    ):
        """
        Create an AttributeInferenceBaseline attack instance.

        :param attack_model_type: the type of default attack model to train, optional. Should be one of `nn` (for neural
                                  network, default) or `rf` (for random forest). If `attack_model` is supplied, this
                                  option will be ignored.
        :param attack_model: The attack model to train, optional. If none is provided, a default model will be created.
        :param attack_feature: The index of the feature to be attacked or a slice representing multiple indexes in
                               case of a one-hot encoded feature.
        :param is_continuous: Whether the attacked feature is continuous. Default is False (which means categorical).
        :param non_numerical_features: a list of feature indexes that require encoding in order to feed into an ML model
                                       (i.e., strings), not including the attacked feature. Should only be supplied if
                                       non-numeric features exist in the input data not including the attacked feature,
                                       and an encoder is not supplied.
        :param encoder: An already fit encoder that can be applied to the model's input features without the attacked
                        feature (i.e., should be fit for n-1 features).
        """
        super().__init__(estimator=None, attack_feature=attack_feature)

        self._values: Optional[list] = None
        self._encoder = encoder
        self._non_numerical_features = non_numerical_features
        self._is_continuous = is_continuous

        if attack_model:
            if self._is_continuous:
                if RegressorMixin not in type(attack_model).__mro__:
                    raise ValueError("When attacking a continuous feature the attack model must be of type Regressor.")
            elif ClassifierMixin not in type(attack_model).__mro__:
                raise ValueError("When attacking a categorical feature the attack model must be of type Classifier.")
            self.attack_model = attack_model
        elif attack_model_type == "nn":
            if self._is_continuous:
                self.attack_model = MLPRegressor(
                    hidden_layer_sizes=(100,),
                    activation="relu",
                    solver="adam",
                    alpha=0.0001,
                    batch_size="auto",
                    learning_rate="constant",
                    learning_rate_init=0.001,
                    power_t=0.5,
                    max_iter=200,
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
            else:
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
            if self._is_continuous:
                self.attack_model = RandomForestRegressor()
            else:
                self.attack_model = RandomForestClassifier()
        else:
            raise ValueError("Illegal value for parameter `attack_model_type`.")

        self._check_params()
        remove_attacked_feature(self.attack_feature, self._non_numerical_features)

    def fit(self, x: np.ndarray) -> None:
        """
        Train the attack model.

        :param x: Input to training process. Includes all features used to train the original model.
        """

        # Checks:
        if isinstance(self.attack_feature, int) and self.attack_feature >= x.shape[1]:
            raise ValueError("attack_feature must be a valid index to a feature in x")

        # get vector of attacked feature
        y = x[:, self.attack_feature]
        y_ready = y
        if not self._is_continuous:
            self._values = get_feature_values(y, isinstance(self.attack_feature, int))
            # number of values in case of single column, number of columns in case of multi-column feature
            nb_classes = len(self._values)
            if isinstance(self.attack_feature, int):
                y_one_hot = float_to_categorical(y)
            else:
                y_one_hot = floats_to_one_hot(y)
            y_ready = check_and_transform_label_format(y_one_hot, nb_classes=nb_classes, return_one_hot=True)
            if y_ready is None:
                raise ValueError("None value detected.")

        # create training set for attack model
        x_train = np.delete(x, self.attack_feature, 1)

        if self._non_numerical_features and self._encoder is None:
            if isinstance(self.attack_feature, int):
                compare_index = self.attack_feature
                size = 1
            else:
                compare_index = self.attack_feature.start
                size = (self.attack_feature.stop - self.attack_feature.start) // self.attack_feature.step
            new_indexes = [(f - size) if f > compare_index else f for f in self._non_numerical_features]
            categorical_transformer = OrdinalEncoder()
            self._encoder = ColumnTransformer(
                transformers=[
                    ("cat", categorical_transformer, new_indexes),
                ],
                remainder="passthrough",
            )
            self._encoder.fit(x_train)

        # train attack model
        if self._encoder is not None:
            x_train = self._encoder.transform(x_train)
        x_train = x_train.astype(np.float32)
        self.attack_model.fit(x_train, y_ready)

    def infer(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Infer the attacked feature.

        :param x: Input to attack. Includes all features except the attacked feature.
        :param y: Not used in this attack.
        :param values: Possible values for attacked feature. For a single column feature this should be a simple list
                       containing all possible values, in increasing order (the smallest value in the 0 index and so
                       on). For a multi-column feature (for example 1-hot encoded and then scaled), this should be a
                       list of lists, where each internal list represents a column (in increasing order) and the values
                       represent the possible values for that column (in increasing order).
        :type values: list
        :return: The inferred feature values.
        """
        values = kwargs.get("values")

        # if provided, override the values computed in fit()
        if values is not None:
            self._values = values

        x_test = x
        if self._encoder is not None:
            x_test = self._encoder.transform(x)
        x_test = x_test.astype(np.float32)
        predictions = self.attack_model.predict(x_test).astype(np.float32)

        if not self._is_continuous and self._values is not None:
            if isinstance(self.attack_feature, int):
                # replace 1-hot encoded prediction with correct single feature value
                predictions = np.array([self._values[np.argmax(arr)] for arr in predictions])
            else:
                i = 0
                # replace 1-hot encoded prediction with multi-column feature value
                for column in predictions.T:
                    # possible values for ith column
                    for index in range(len(self._values[i])):
                        np.place(column, [column == index], self._values[i][index])
                    i += 1
        return np.array(predictions)

    def _check_params(self) -> None:

        super()._check_params()

        if not isinstance(self._is_continuous, bool):
            raise ValueError("is_continuous must be a boolean.")

        if self._non_numerical_features and (
            (not isinstance(self._non_numerical_features, list))
            or (not all(isinstance(item, int) for item in self._non_numerical_features))
        ):
            raise ValueError("non_numerical_features must be a list of int.")

        if self._encoder is not None and (
            not isinstance(self._encoder, OrdinalEncoder)
            and not isinstance(self._encoder, OneHotEncoder)
            and not isinstance(self._encoder, ColumnTransformer)
        ):
            raise ValueError("encoder must be a OneHotEncoder, OrdinalEncoder or ColumnTransformer object.")
