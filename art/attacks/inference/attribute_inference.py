# MIT License
#
# Copyright (C) IBM Corporation 2020
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

import numpy as np

from sklearn.neural_network import MLPClassifier

from art.estimators.estimator import BaseEstimator
from art.estimators.classification.scikitlearn import ScikitlearnDecisionTreeClassifier
from art.attacks import InferenceAttack


logger = logging.getLogger(__name__)


class AttributeInferenceBlackBox(InferenceAttack):
    """
    Implementation of a simple black-box attribute inference attack.

    The idea is to train a simple neural network to learn the attacked
    feature from the rest of the features and the model's predictions.
    Assumes the availability of the attacked model's predictions for the samples under attack,
    in addition to the rest of the feature values.
    If this is not available, the true class label of the samples may be used as a proxy.
    """

    attack_params = InferenceAttack.attack_params + ["attack_feature"]

    _estimator_requirements = (BaseEstimator)

    def __init__(self, classifier, attack_feature=0):
        """
        Create an AttributeInferenceBlackBox attack instance.

        :param classifier: Target classifier.
        :type classifier: :class:`.Classifier`
        :param attack_feature: The index of the feature to be attacked.
        :type attack_feature: `int`
        """
        super(AttributeInferenceBlackBox, self).__init__(estimator=classifier)

        params = {
            "attack_feature": attack_feature
        }
        self.set_params(**params)

    def fit(self, x, **kwargs):
        """
        Train the attack model.

        :param x: Input to training process. Includes all features used to train the original model.
        :type x: `np.ndarray`
        :return: None
        """

        # get model's predictions for x
        predictions = self.estimator.predict(x)

        # get vector of attacked feature
        y = x[:, self.attack_feature]

        # create training set for attack model
        x_train = np.concatenate(np.delete(x, self.attack_feature, 1), predictions, axis=1)

        # train attack model
        self.attack_model = MLPClassifier()
        self.attack_model.fit(x_train, y)

    def infer(self, x, y, **kwargs):
        """
        Infer the attacked feature.

        :param x: Input to attack. Includes all features except the attacked feature.
        :type x: `np.ndarray`
        :param y: Original model's predictions for x.
        :type y: `np.ndarray`
        :return: The inferred feature values.
        :rtype: `np.ndarray`
        """

        x_test = np.concatenate(x, y, axis=1)
        return self.attack_model.predict(x_test)

class AttributeInferenceWhiteBoxLifestyle(InferenceAttack):
    """
    Implementation of Fredrikson et al. white box inference attack for decision trees.

    Paper link: https://dl.acm.org/doi/10.1145/2810103.2813677
    Assumes that the attacked feature is discrete or categorical, with limited number
    of possible values. For example: a boolean feature.
    """

    attack_params = InferenceAttack.attack_params + ["attack_feature"]

    _estimator_requirements = (BaseEstimator, ScikitlearnDecisionTreeClassifier)

    def __init__(self, classifier, attack_feature=0):
        """
        Create an AttributeInferenceWhiteBoxLifestyle attack instance.

        :param classifier: Target classifier.
        :type classifier: :class:`.Classifier`
        :param attack_feature: The index of the feature to be attacked.
        :type attack_feature: `int`
        """
        super(AttributeInferenceWhiteBoxLifestyle, self).__init__(estimator=classifier)

        params = {
            "attack_feature": attack_feature
        }
        self.set_params(**params)


    def infer(self, x, y=None, **kwargs):
        """
        Infer the attacked feature.

        :param x: Input to attack. Includes all features except the attacked feature.
        :type x: `np.ndarray`
        :param values: Possible values for attacked feature.
        :type values: `np.ndarray`
        :param priors: Prior distributions of attacked feature values. Same size array
            as `values`.
        :type priors: `np.ndarray`
        :return: The inferred feature values.
        :rtype: `np.ndarray`
        """
        priors = kwargs.get("priors")
        values = kwargs.get("values")
        n_samples = x.shape[0]

        # Calculate phi for each possible value of the attacked feature
        # phi is the total number of samples in all tree leaves corresponding to this value
        phi = self._calculate_phi(x, values, n_samples)

        # Will contain the probability of each value
        prob_values = []

        for i, value in enumerate(values):
            # prepare data with the given value in the attacked feature
            v = np.full((n_samples, 1), value)
            x_value = np.concatenate(x[:,:self.attack_feature], v, axis=1)
            x_value = np.concatenate(x_value, x[:,self.attack_feature:], axis=1)
            # find the relative probability of this value for all samples being attacked
            prob_value = [((self.estimator.get_samples_at_node(self.estimator.get_decision_path([row])[0]) / n_samples) * priors[i] / phi[i])
                          for row in x_value]
            prob_values.append(prob_value)

        # Choose the value with highest probability for each sample
        return [np.argmax(list(prob)) for prob in zip(prob_values)]

    def _calculate_phi(self, x, values, n_samples):
        phi = []
        for value in values:
            v = np.full((n_samples, 1), value)
            x_value = np.concatenate(x[:,:self.attack_feature], v, axis=1)
            x_value = np.concatenate(x_value, x[:,self.attack_feature:], axis=1)
            nodes_value = {}

            for row in x_value:
                # get leaf ids (no duplicates)
                node_id = self.estimator.get_decision_path([row])[0]
                nodes_value[node_id] = self.estimator.get_samples_at_node(node_id)
            # sum sample numbers
            num_value = sum(nodes_value.values()) / n_samples
            phi.append(num_value)

        return phi

class AttributeInferenceWhiteBox(InferenceAttack):
    """
    A variation of the method proposed by of Fredrikson et al. in: https://dl.acm.org/doi/10.1145/2810103.2813677

    Assumes the availability of the attacked model's predictions for the samples under attack,
    in addition to access to the model itself and the rest of the feature values.
    If this is not available, the true class label of the samples may be used as a proxy.
    Also assumes that the attacked feature is discrete or categorical, with limited number
    of possible values. For example: a boolean feature.
    """

    attack_params = InferenceAttack.attack_params + ["attack_feature"]

    _estimator_requirements = (BaseEstimator, ScikitlearnDecisionTreeClassifier)

    def __init__(self, classifier, attack_feature=0):
        """
        Create an AttributeInferenceWhiteBox attack instance.

        :param classifier: Target classifier.
        :type classifier: :class:`.Classifier`
        :param attack_feature: The index of the feature to be attacked.
        :type attack_feature: `int`
        """
        super(AttributeInferenceWhiteBox, self).__init__(estimator=classifier)

        params = {
            "attack_feature": attack_feature
        }
        self.set_params(**params)


    def infer(self, x, y=None, **kwargs):
        """
        Infer the attacked feature.

        If the model's prediction coincides with the real prediction for the sample for a single
        value, choose it as the predicted value
        If not, fall back to the Fredrikson method (without phi)

        :param x: Input to attack. Includes all features except the attacked feature.
        :type x: `np.ndarray`
        :param y: Original model's predictions for x.
        :type y: `np.ndarray`
        :param values: Possible values for attacked feature.
        :type values: `np.ndarray`
        :param priors: Prior distributions of attacked feature values. Same size array
            as `values`.
        :type priors: `np.ndarray`
        :return: The inferred feature values.
        :rtype: `np.ndarray`
        """
        priors = kwargs.get("priors")
        values = kwargs.get("values")
        n_values = len(values)
        n_samples = x.shape[0]

        # Will contain the model's predictions for each value
        pred_values = []
        # Will contain the probability of each value
        prob_values = []

        for i, value in enumerate(values):
            # prepare data with the given value in the attacked feature
            v = np.full((n_samples, 1), value)
            x_value = np.concatenate(x[:,:self.attack_feature], v, axis=1)
            x_value = np.concatenate(x_value, x[:,self.attack_feature:], axis=1)

            # Obtain the model's prediction for each possible value of the attacked feature
            pred_value = self.estimator.predict(x_value)
            pred_values.append(pred_value)

            # find the relative probability of this value for all samples being attacked
            prob_value = [((self.estimator.get_samples_at_node(self.estimator.get_decision_path([row])[0]) / n_samples) * priors[i])
                          for row in x_value]
            prob_values.append(prob_value)

        # Find the single value that coincides with the real prediction for the sample (if it exists)
        pred_rows = zip(pred_values)
        predicted_pred = []
        for row_index, row in enumerate(pred_rows):
            if y:
                matches = [1 if row[value_index] == y[row_index] else 0 for value_index in range(n_values)]
                match_values = [row[value_index] if row[value_index] == y[row_index] else 0 for value_index in range(n_values)]
            else:
                matches = [0 for value_index in range(n_values)]
                match_values = [0 for value_index in range(n_values)]
            predicted_pred.append(sum(match_values) if sum(matches) == 1 else None)


        # Choose the value with highest probability for each sample
        predicted_prob = [np.argmax(list(prob)) for prob in zip(prob_values)]

        return [value if value is not None else predicted_prob[index] for index, value in enumerate(predicted_pred)]





