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
from typing import Optional

import numpy as np
from sklearn.neural_network import MLPClassifier

from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin, Classifier
from art.estimators.classification.scikitlearn import ScikitlearnDecisionTreeClassifier
from art.attacks import AttributeInferenceAttack
from art.utils import check_and_transform_label_format, float_to_categorical


logger = logging.getLogger(__name__)


class AttributeInferenceBlackBox(AttributeInferenceAttack):
    """
    Implementation of a simple black-box attribute inference attack.

    The idea is to train a simple neural network to learn the attacked feature from the rest of the features and the
    model's predictions. Assumes the availability of the attacked model's predictions for the samples under attack,
    in addition to the rest of the feature values. If this is not available, the true class label of the samples may be
    used as a proxy.
    """

    _estimator_requirements = [BaseEstimator]

    def __init__(self, classifier: Classifier, attack_model: Optional[Classifier] = None, attack_feature: int = 0):
        """
        Create an AttributeInferenceBlackBox attack instance.

        :param classifier: Target classifier.
        :param attack_model: The attack model to train, optional. If none is provided, a default model will be created.
        :param attack_feature: The index of the feature to be attacked.
        """
        super(AttributeInferenceBlackBox, self).__init__(estimator=classifier, attack_feature=attack_feature)

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
        self._check_params()

    def fit(self, x: np.ndarray) -> None:
        """
        Train the attack model.

        :param x: Input to training process. Includes all features used to train the original model.
        """

        # Checks:
        if self.estimator.input_shape[0] != x.shape[1]:
            raise ValueError("Shape of x does not match input_shape of classifier")
        if self.attack_feature >= x.shape[1]:
            raise ValueError("attack_feature must be a valid index to a feature in x")

        # get model's predictions for x
        predictions = np.array([np.argmax(arr) for arr in self.estimator.predict(x)]).reshape(-1, 1)

        # get vector of attacked feature
        y = x[:, self.attack_feature]
        y_one_hot = float_to_categorical(y)
        y_ready = check_and_transform_label_format(y_one_hot, len(np.unique(y)), return_one_hot=True)

        # create training set for attack model
        x_train = np.concatenate((np.delete(x, self.attack_feature, 1), predictions), axis=1).astype(np.float32)

        # train attack model
        self.attack_model.fit(x_train, y_ready)

    def infer(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Infer the attacked feature.

        :param x: Input to attack. Includes all features except the attacked feature.
        :param y: Original model's predictions for x.
        :param values: Possible values for attacked feature.
        :type values: `np.ndarray`
        :return: The inferred feature values.
        """
        if y is not None and y.shape[0] != x.shape[0]:
            raise ValueError("Number of rows in x and y do not match")
        if self.estimator.input_shape[0] != x.shape[1] + 1:
            raise ValueError("Number of features in x + 1 does not match input_shape of classifier")

        if "values" not in kwargs.keys():
            raise ValueError("Missing parameter `values`.")
        values: np.ndarray = kwargs.get("values")

        x_test = np.concatenate((x, y), axis=1).astype(np.float32)
        return np.array([values[np.argmax(arr)] for arr in self.attack_model.predict(x_test)])


class AttributeInferenceWhiteBoxLifestyleDecisionTree(AttributeInferenceAttack):
    """
    Implementation of Fredrikson et al. white box inference attack for decision trees.

    Assumes that the attacked feature is discrete or categorical, with limited number of possible values. For example:
    a boolean feature.

    | Paper link: https://dl.acm.org/doi/10.1145/2810103.2813677
    """

    _estimator_requirements = (BaseEstimator, ScikitlearnDecisionTreeClassifier)

    def __init__(self, classifier: Classifier, attack_feature: int = 0):
        """
        Create an AttributeInferenceWhiteBoxLifestyle attack instance.

        :param classifier: Target classifier.
        :param attack_feature: The index of the feature to be attacked.
        """
        super(AttributeInferenceWhiteBoxLifestyleDecisionTree, self).__init__(
            estimator=classifier, attack_feature=attack_feature
        )

    def infer(self, x, y=None, **kwargs):
        """
        Infer the attacked feature.

        :param x: Input to attack. Includes all features except the attacked feature.
        :type x: `np.ndarray`
        :param values: Possible values for attacked feature.
        :type values: `np.ndarray`
        :param priors: Prior distributions of attacked feature values. Same size array as `values`.
        :type priors: `np.ndarray`
        :return: The inferred feature values.
        :rtype: `np.ndarray`
        """
        if "priors" not in kwargs.keys():
            raise ValueError("Missing parameter `priors`.")
        if "values" not in kwargs.keys():
            raise ValueError("Missing parameter `values`.")
        priors = kwargs.get("priors")
        values = kwargs.get("values")

        # Checks:
        if self.estimator.input_shape[0] != x.shape[1] + 1:
            raise ValueError("Number of features in x + 1 does not match input_shape of classifier")
        if len(priors) != len(values):
            raise ValueError("Number of priors does not match number of values")
        if self.attack_feature >= x.shape[1]:
            raise ValueError("attack_feature must be a valid index to a feature in x")

        n_samples = x.shape[0]

        # Calculate phi for each possible value of the attacked feature
        # phi is the total number of samples in all tree leaves corresponding to this value
        phi = self._calculate_phi(x, values, n_samples)

        # Will contain the probability of each value
        prob_values = []

        for i, value in enumerate(values):
            # prepare data with the given value in the attacked feature
            v = np.full((n_samples, 1), value)
            x_value = np.concatenate((x[:, : self.attack_feature], v), axis=1)
            x_value = np.concatenate((x_value, x[:, self.attack_feature :]), axis=1)

            # find the relative probability of this value for all samples being attacked
            prob_value = [
                (
                    (self.estimator.get_samples_at_node(self.estimator.get_decision_path([row])[-1]) / n_samples)
                    * priors[i]
                    / phi[i]
                )
                for row in x_value
            ]
            prob_values.append(prob_value)

        # Choose the value with highest probability for each sample
        return np.array([values[np.argmax(list(prob))] for prob in zip(*prob_values)])

    def _calculate_phi(self, x, values, n_samples):
        phi = []
        for value in values:
            v = np.full((n_samples, 1), value)
            x_value = np.concatenate((x[:, : self.attack_feature], v), axis=1)
            x_value = np.concatenate((x_value, x[:, self.attack_feature :]), axis=1)
            nodes_value = {}

            for row in x_value:
                # get leaf ids (no duplicates)
                node_id = self.estimator.get_decision_path([row])[0]
                nodes_value[node_id] = self.estimator.get_samples_at_node(node_id)
            # sum sample numbers
            num_value = sum(nodes_value.values()) / n_samples
            phi.append(num_value)

        return phi


class AttributeInferenceWhiteBoxDecisionTree(AttributeInferenceAttack):
    """
    A variation of the method proposed by of Fredrikson et al. in:
    https://dl.acm.org/doi/10.1145/2810103.2813677

    Assumes the availability of the attacked model's predictions for the samples under attack, in addition to access to
    the model itself and the rest of the feature values. If this is not available, the true class label of the samples
    may be used as a proxy. Also assumes that the attacked feature is discrete or categorical, with limited number of
    possible values. For example: a boolean feature.

    | Paper link: https://dl.acm.org/doi/10.1145/2810103.2813677
    """

    _estimator_requirements = (BaseEstimator, ScikitlearnDecisionTreeClassifier)

    def __init__(self, classifier, attack_feature=0):
        """
        Create an AttributeInferenceWhiteBox attack instance.

        :param classifier: Target classifier.
        :type classifier: :class:`.Classifier`
        :param attack_feature: The index of the feature to be attacked.
        :type attack_feature: `int`
        """
        super(AttributeInferenceWhiteBoxDecisionTree, self).__init__(
            estimator=classifier, attack_feature=attack_feature
        )

    def infer(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Infer the attacked feature.

        If the model's prediction coincides with the real prediction for the sample for a single value, choose it as the
        predicted value. If not, fall back to the Fredrikson method (without phi)

        :param x: Input to attack. Includes all features except the attacked feature.
        :param y: Original model's predictions for x.
        :param values: Possible values for attacked feature.
        :type values: `np.ndarray`
        :param priors: Prior distributions of attacked feature values. Same size array as `values`.
        :type priors: `np.ndarray`
        :return: The inferred feature values.
        """
        if "priors" not in kwargs.keys():
            raise ValueError("Missing parameter `priors`.")
        if "values" not in kwargs.keys():
            raise ValueError("Missing parameter `values`.")
        priors: np.ndarray = kwargs.get("priors")
        values: np.ndarray = kwargs.get("values")

        if self.estimator.input_shape[0] != x.shape[1] + 1:
            raise ValueError("Number of features in x + 1 does not match input_shape of classifier")
        if len(priors) != len(values):
            raise ValueError("Number of priors does not match number of values")
        if y is not None and y.shape[0] != x.shape[0]:
            raise ValueError("Number of rows in x and y do not match")
        if self.attack_feature >= x.shape[1]:
            raise ValueError("attack_feature must be a valid index to a feature in x")

        n_values = len(values)
        n_samples = x.shape[0]

        # Will contain the model's predictions for each value
        pred_values = []
        # Will contain the probability of each value
        prob_values = []

        for i, value in enumerate(values):
            # prepare data with the given value in the attacked feature
            v = np.full((n_samples, 1), value)
            x_value = np.concatenate((x[:, : self.attack_feature], v), axis=1)
            x_value = np.concatenate((x_value, x[:, self.attack_feature :]), axis=1)

            # Obtain the model's prediction for each possible value of the attacked feature
            pred_value = [np.argmax(arr) for arr in self.estimator.predict(x_value)]
            pred_values.append(pred_value)

            # find the relative probability of this value for all samples being attacked
            prob_value = [
                (
                    (self.estimator.get_samples_at_node(self.estimator.get_decision_path([row])[-1]) / n_samples)
                    * priors[i]
                )
                for row in x_value
            ]
            prob_values.append(prob_value)

        # Find the single value that coincides with the real prediction for the sample (if it exists)
        pred_rows = zip(*pred_values)
        predicted_pred = []
        for row_index, row in enumerate(pred_rows):
            if y is not None:
                matches = [1 if row[value_index] == y[row_index] else 0 for value_index in range(n_values)]
                match_values = [
                    values[value_index] if row[value_index] == y[row_index] else 0 for value_index in range(n_values)
                ]
            else:
                matches = [0 for value_index in range(n_values)]
                match_values = [0 for value_index in range(n_values)]
            predicted_pred.append(sum(match_values) if sum(matches) == 1 else None)

        # Choose the value with highest probability for each sample
        predicted_prob = [np.argmax(list(prob)) for prob in zip(*prob_values)]

        return np.array(
            [
                value if value is not None else values[predicted_prob[index]]
                for index, value in enumerate(predicted_pred)
            ]
        )
