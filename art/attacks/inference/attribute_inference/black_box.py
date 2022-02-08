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
from sklearn.preprocessing import minmax_scale

from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin
from art.attacks.attack import AttributeInferenceAttack
from art.estimators.regression import RegressorMixin
from art.utils import check_and_transform_label_format, float_to_categorical, floats_to_one_hot

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE, REGRESSOR_TYPE

logger = logging.getLogger(__name__)


class AttributeInferenceBlackBox(AttributeInferenceAttack):
    """
    Implementation of a simple black-box attribute inference attack.

    The idea is to train a simple neural network to learn the attacked feature from the rest of the features and the
    model's predictions. Assumes the availability of the attacked model's predictions for the samples under attack,
    in addition to the rest of the feature values. If this is not available, the true class label of the samples may be
    used as a proxy.
    """

    attack_params = AttributeInferenceAttack.attack_params + ["prediction_normal_factor", "scale_range"]
    _estimator_requirements = (BaseEstimator, (ClassifierMixin, RegressorMixin))

    def __init__(
        self,
        estimator: Union["CLASSIFIER_TYPE", "REGRESSOR_TYPE"],
        attack_model: Optional["CLASSIFIER_TYPE"] = None,
        attack_feature: Union[int, slice] = 0,
        scale_range: Optional[slice] = None,
        prediction_normal_factor: Optional[float] = 1,
    ):
        """
        Create an AttributeInferenceBlackBox attack instance.

        :param estimator: Target estimator.
        :param attack_model: The attack model to train, optional. If none is provided, a default model will be created.
        :param attack_feature: The index of the feature to be attacked or a slice representing multiple indexes in
                               case of a one-hot encoded feature.
        :param scale_range: If supplied, the class labels (both true and predicted) will be scaled to the given range.
                            Only applicable when `estimator` is a regressor.
        :param prediction_normal_factor: If supplied, the class labels (both true and predicted) are multiplied by the
                                         factor when used as inputs to the attack-model. Only applicable when
                                         `estimator` is a regressor and if `scale_range` is not supplied.
        """
        super().__init__(estimator=estimator, attack_feature=attack_feature)
        if isinstance(self.attack_feature, int):
            self.single_index_feature = True
        else:
            self.single_index_feature = False

        self._values = None

        if attack_model:
            if ClassifierMixin not in type(attack_model).__mro__:
                raise ValueError("Attack model must be of type Classifier.")
            self.attack_model = attack_model
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

        self.prediction_normal_factor = prediction_normal_factor
        self.scale_range = scale_range

        self._check_params()

    def fit(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """
        Train the attack model.

        :param x: Input to training process. Includes all features used to train the original model.
        :param y: True labels for x.
        """

        # Checks:
        if self.estimator.input_shape is not None:
            if self.estimator.input_shape[0] != x.shape[1]:
                raise ValueError("Shape of x does not match input_shape of model")
        if self.single_index_feature and self.attack_feature >= x.shape[1]:
            raise ValueError("attack_feature must be a valid index to a feature in x")

        # get model's predictions for x
        if ClassifierMixin in type(self.estimator).__mro__:
            predictions = np.array([np.argmax(arr) for arr in self.estimator.predict(x)]).reshape(-1, 1)
        else:  # Regression model
            if self.scale_range is not None:
                predictions = minmax_scale(self.estimator.predict(x).reshape(-1, 1), feature_range=self.scale_range)
            else:
                predictions = self.estimator.predict(x).reshape(-1, 1) * self.prediction_normal_factor

        # get vector of attacked feature
        y_attack = x[:, self.attack_feature]
        if self.single_index_feature:
            self._values = np.unique(y_attack).tolist()
            y_one_hot = float_to_categorical(y_attack)
        else:
            for column in y_attack.T:
                column_values = np.unique(column)
                if self._values is None:
                    self._values = column_values
                else:
                    self._values = np.vstack((self._values, column_values))
            self._values = self._values.tolist()
            y_one_hot = floats_to_one_hot(y_attack)
        y_attack_ready = check_and_transform_label_format(y_one_hot, len(np.unique(y_attack)), return_one_hot=True)

        # create training set for attack model
        x_train = np.concatenate((np.delete(x, self.attack_feature, 1), predictions), axis=1).astype(np.float32)

        if y is not None:
            y_ready = check_and_transform_label_format(y, return_one_hot=False)
            x_train = np.concatenate((x_train, y_ready), axis=1)

        # train attack model
        self.attack_model.fit(x_train, y_attack_ready)

    def infer(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Infer the attacked feature.

        :param x: Input to attack. Includes all features except the attacked feature.
        :param y: True labels for x.
        :param pred: Original model's predictions for x.
        :type pred: `np.ndarray`
        :param values: Possible values for attacked feature. For a single column feature this should be a simple list
                       containing all possible values, in increasing order (the smallest value in the 0 index and so
                       on). For a multi-column feature (for example 1-hot encoded and then scaled), this should be a
                       list of lists, where each internal list represents a column (in increasing order) and the values
                       represent the possible values for that column (in increasing order). If not provided, is
                       computed from the training data when calling `fit`.
        :type values: list, optional
        :return: The inferred feature values.
        """
        if "pred" not in kwargs.keys():
            raise ValueError(
                "Please provide param `pred` of model predictions."
            )
        pred = kwargs.get("pred")

        if pred.shape[0] != x.shape[0]:
            raise ValueError("Number of rows in x and y do not match")
        if self.estimator.input_shape is not None:
            if self.single_index_feature and self.estimator.input_shape[0] != x.shape[1] + 1:
                raise ValueError("Number of features in x + 1 does not match input_shape of model")

        if RegressorMixin in type(self.estimator).__mro__:
            if self.scale_range is not None:
                x_test = np.concatenate((x, minmax_scale(pred, feature_range=self.scale_range)),
                                        axis=1).astype(np.float32)
            else:
                x_test = np.concatenate((x, pred * self.prediction_normal_factor), axis=1).astype(np.float32)
        else:
            x_test = np.concatenate((x, pred), axis=1).astype(np.float32)

        if y is not None:
            y_ready = check_and_transform_label_format(y, return_one_hot=False)
            x_test = np.concatenate((x_test, y_ready), axis=1)

        # if provided, override the values computed in fit()
        if "values" in kwargs.keys():
            self._values = kwargs.get("values")

        if self.single_index_feature:
            return np.array([self._values[np.argmax(arr)] for arr in self.attack_model.predict(x_test)])

        predictions = self.attack_model.predict(x_test).astype(np.float32)
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

        if RegressorMixin not in type(self.estimator).__mro__:
            if self.prediction_normal_factor != 1:
                raise ValueError("Prediction normal factor is only applicable to regressor models.")
