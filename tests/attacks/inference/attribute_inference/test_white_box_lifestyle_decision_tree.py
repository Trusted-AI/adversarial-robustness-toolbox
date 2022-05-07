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
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import pytest

import numpy as np

from art.attacks.inference.attribute_inference.white_box_lifestyle_decision_tree import (
    AttributeInferenceWhiteBoxLifestyleDecisionTree,
)
from art.estimators.classification.scikitlearn import ScikitlearnDecisionTreeClassifier
from art.estimators.regression.scikitlearn import ScikitlearnDecisionTreeRegressor

from tests.attacks.utils import backend_test_classifier_type_check_fail
from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.mark.skip_framework("dl_frameworks")
def test_white_box_lifestyle(art_warning, decision_tree_estimator, get_iris_dataset):
    try:
        attack_feature = 2  # petal length
        values = [0.14, 0.42, 0.71]  # rounded down
        priors = [50 / 150, 54 / 150, 46 / 150]

        (x_train_iris, y_train_iris), (x_test_iris, y_test_iris) = get_iris_dataset
        x_train_for_attack = np.delete(x_train_iris, attack_feature, 1)
        x_train_feature = x_train_iris[:, attack_feature]
        x_test_for_attack = np.delete(x_test_iris, attack_feature, 1)
        x_test_feature = x_test_iris[:, attack_feature]

        classifier = decision_tree_estimator()
        attack = AttributeInferenceWhiteBoxLifestyleDecisionTree(classifier, attack_feature=attack_feature)
        x_train_predictions = np.array([np.argmax(arr) for arr in classifier.predict(x_train_iris)]).reshape(-1, 1)
        x_test_predictions = np.array([np.argmax(arr) for arr in classifier.predict(x_test_iris)]).reshape(-1, 1)
        inferred_train = attack.infer(x_train_for_attack, x_train_predictions, values=values, priors=priors)
        inferred_test = attack.infer(x_test_for_attack, x_test_predictions, values=values, priors=priors)
        train_diff = np.abs(inferred_train - x_train_feature.reshape(1, -1))
        test_diff = np.abs(inferred_test - x_test_feature.reshape(1, -1))
        assert np.sum(train_diff) / len(inferred_train) == pytest.approx(0.3357, abs=0.03)
        assert np.sum(test_diff) / len(inferred_test) == pytest.approx(0.3149, abs=0.03)
        # assert np.sum(train_diff) / len(inferred_train) < np.sum(test_diff) / len(inferred_test)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("dl_frameworks")
def test_white_box_lifestyle_regression(art_warning, get_diabetes_dataset):
    try:
        attack_feature = 0  # age
        (x_train_diabetes, y_train_diabetes), (x_test_diabetes, y_test_diabetes) = get_diabetes_dataset

        bins = [
            -0.96838121,
            -0.18102872,
            0.21264752,
            1.0,
        ]

        # need to transform attacked feature into categorical
        def transform_feature(x):
            orig = x.copy()
            for i in range(3):
                x[(orig >= bins[i]) & (orig <= bins[i + 1])] = i / 3

        values = [i / 3 for i in range(3)]
        priors = [154 / 442, 145 / 442, 143 / 442]

        x_train_for_attack = np.delete(x_train_diabetes, attack_feature, 1)
        x_train_feature = x_train_diabetes[:, attack_feature].copy()
        transform_feature(x_train_feature)

        x_test_for_attack = np.delete(x_test_diabetes, attack_feature, 1)
        x_test_feature = x_test_diabetes[:, attack_feature].copy()
        transform_feature(x_test_feature)

        from sklearn import tree

        regr_model = tree.DecisionTreeRegressor(random_state=7)
        regr_model.fit(x_train_diabetes, y_train_diabetes)
        regressor = ScikitlearnDecisionTreeRegressor(regr_model)

        attack = AttributeInferenceWhiteBoxLifestyleDecisionTree(regressor, attack_feature=attack_feature)

        x_train_predictions = regressor.predict(x_train_diabetes).reshape(-1, 1)
        x_test_predictions = regressor.predict(x_test_diabetes).reshape(-1, 1)

        inferred_train = attack.infer(x_train_for_attack, x_train_predictions, values=values, priors=priors)
        inferred_test = attack.infer(x_test_for_attack, x_test_predictions, values=values, priors=priors)

        train_diff = np.abs(inferred_train - x_train_feature.reshape(1, -1))
        test_diff = np.abs(inferred_test - x_test_feature.reshape(1, -1))
        assert np.sum(train_diff) / len(inferred_train) == pytest.approx(0.318, abs=0.1)
        assert np.sum(test_diff) / len(inferred_test) == pytest.approx(0.34, abs=0.12)
        # assert np.sum(train_diff) / len(inferred_train) < np.sum(test_diff) / len(inferred_test)

    except ARTTestException as e:
        art_warning(e)


def test_check_params(art_warning, image_dl_estimator_for_attack):
    try:
        classifier = image_dl_estimator_for_attack(AttributeInferenceWhiteBoxLifestyleDecisionTree)

        with pytest.raises(ValueError):
            _ = AttributeInferenceWhiteBoxLifestyleDecisionTree(classifier, attack_feature=-5)

    except ARTTestException as e:
        art_warning(e)


def test_classifier_type_check_fail():
    backend_test_classifier_type_check_fail(
        AttributeInferenceWhiteBoxLifestyleDecisionTree,
        ((ScikitlearnDecisionTreeClassifier, ScikitlearnDecisionTreeRegressor),),
    )
