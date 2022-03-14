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
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import pytest

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

from art.attacks.inference.attribute_inference.meminf_based import AttributeInferenceMembership
from art.attacks.inference.membership_inference import (
    MembershipInferenceBlackBox,
    MembershipInferenceBlackBoxRuleBased,
    LabelOnlyDecisionBoundary,
)
from art.estimators.classification.scikitlearn import ScikitlearnDecisionTreeClassifier
from art.estimators.regression import ScikitlearnRegressor

from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.mark.skip_framework("dl_frameworks")
def test_meminf_black_box(art_warning, decision_tree_estimator, get_iris_dataset):
    try:
        attack_feature = 2  # petal length

        # need to transform attacked feature into categorical
        def transform_feature(x):
            x[x > 0.5] = 0.6
            x[(x > 0.2) & (x <= 0.5)] = 0.35
            x[x <= 0.2] = 0.1

        values = [0.1, 0.35, 0.6]

        (x_train_iris, y_train_iris), (x_test_iris, y_test_iris) = get_iris_dataset
        # training data without attacked feature
        x_train_for_attack = np.delete(x_train_iris, attack_feature, 1)
        # only attacked feature
        x_train_feature = x_train_iris[:, attack_feature].copy().reshape(-1, 1)
        transform_feature(x_train_feature)
        # training data with attacked feature (after transformation)
        x_train = np.concatenate((x_train_for_attack[:, :attack_feature], x_train_feature), axis=1)
        x_train = np.concatenate((x_train, x_train_for_attack[:, attack_feature:]), axis=1)

        # test data without attacked feature
        x_test_for_attack = np.delete(x_test_iris, attack_feature, 1)
        # only attacked feature
        x_test_feature = x_test_iris[:, attack_feature].copy().reshape(-1, 1)
        transform_feature(x_test_feature)
        # test data with attacked feature (after transformation)
        x_test = np.concatenate((x_test_for_attack[:, :attack_feature], x_test_feature), axis=1)
        x_test = np.concatenate((x_test, x_test_for_attack[:, attack_feature:]), axis=1)

        classifier = decision_tree_estimator()

        meminf_attack = MembershipInferenceBlackBox(classifier, attack_model_type="nn")
        attack_train_ratio = 0.5
        attack_train_size = int(len(x_train) * attack_train_ratio)
        attack_test_size = int(len(x_test) * attack_train_ratio)
        meminf_attack.fit(
            x_train[:attack_train_size],
            y_train_iris[:attack_train_size],
            x_test[:attack_test_size],
            y_test_iris[:attack_test_size],
        )
        attack = AttributeInferenceMembership(classifier, meminf_attack, attack_feature=attack_feature)
        # infer attacked feature
        inferred_train = attack.infer(x_train_for_attack, y_train_iris, values=values)
        inferred_test = attack.infer(x_test_for_attack, y_test_iris, values=values)
        # check accuracy
        train_acc = np.sum(inferred_train == x_train_feature.reshape(1, -1)) / len(inferred_train)
        test_acc = np.sum(inferred_test == x_test_feature.reshape(1, -1)) / len(inferred_test)
        assert 0.1 <= train_acc
        assert 0.1 <= test_acc

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("dl_frameworks")
def test_meminf_black_box_regressor(art_warning, get_diabetes_dataset):
    try:
        attack_feature = 0  # age

        bins = [
            -0.96838121,
            -0.18102872,
            0.21264752,
            1.0,
        ]

        # need to transform attacked feature into categorical
        def transform_feature(x):
            for i in range(len(bins) - 1):
                x[(x >= bins[i]) & (x < bins[i + 1])] = i

        values = list(range(len(bins) - 1))

        (x_train_diabetes, y_train_diabetes), (x_test_diabetes, y_test_diabetes) = get_diabetes_dataset
        # training data without attacked feature
        x_train_for_attack = np.delete(x_train_diabetes, attack_feature, 1)
        # only attacked feature
        x_train_feature = x_train_diabetes[:, attack_feature].copy().reshape(-1, 1)
        transform_feature(x_train_feature)
        # training data with attacked feature (after transformation)
        x_train = np.concatenate((x_train_for_attack[:, :attack_feature], x_train_feature), axis=1)
        x_train = np.concatenate((x_train, x_train_for_attack[:, attack_feature:]), axis=1)

        # test data without attacked feature
        x_test_for_attack = np.delete(x_test_diabetes, attack_feature, 1)
        # only attacked feature
        x_test_feature = x_test_diabetes[:, attack_feature].copy().reshape(-1, 1)
        transform_feature(x_test_feature)
        # test data with attacked feature (after transformation)
        x_test = np.concatenate((x_test_for_attack[:, :attack_feature], x_test_feature), axis=1)
        x_test = np.concatenate((x_test, x_test_for_attack[:, attack_feature:]), axis=1)

        from sklearn import linear_model

        regr_model = linear_model.LinearRegression()
        regr_model.fit(x_train_diabetes, y_train_diabetes)
        regressor = ScikitlearnRegressor(regr_model)

        meminf_attack = MembershipInferenceBlackBox(regressor, attack_model_type="rf", input_type="loss")
        attack_train_ratio = 0.5
        attack_train_size = int(len(x_train) * attack_train_ratio)
        attack_test_size = int(len(x_test) * attack_train_ratio)
        meminf_attack.fit(
            x_train[:attack_train_size],
            y_train_diabetes[:attack_train_size],
            x_test[:attack_test_size],
            y_test_diabetes[:attack_test_size],
        )
        attack = AttributeInferenceMembership(regressor, meminf_attack, attack_feature=attack_feature)
        # infer attacked feature
        inferred_train = attack.infer(x_train_for_attack, y_train_diabetes, values=values)
        inferred_test = attack.infer(x_test_for_attack, y_test_diabetes, values=values)
        # check accuracy
        train_acc = np.sum(inferred_train == x_train_feature.reshape(1, -1)) / len(inferred_train)
        test_acc = np.sum(inferred_test == x_test_feature.reshape(1, -1)) / len(inferred_test)
        assert 0.1 <= train_acc
        assert 0.1 <= test_acc

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("scikitlearn")
def test_meminf_black_box_dl(art_warning, tabular_dl_estimator_for_attack, get_iris_dataset):
    try:
        attack_feature = 2  # petal length

        # need to transform attacked feature into categorical
        def transform_feature(x):
            x[x > 0.5] = 0.6
            x[(x > 0.2) & (x <= 0.5)] = 0.35
            x[x <= 0.2] = 0.1

        values = [0.1, 0.35, 0.6]

        (x_train_iris, y_train_iris), (x_test_iris, y_test_iris) = get_iris_dataset
        # training data without attacked feature
        x_train_for_attack = np.delete(x_train_iris, attack_feature, 1)
        # only attacked feature
        x_train_feature = x_train_iris[:, attack_feature].copy().reshape(-1, 1)
        transform_feature(x_train_feature)
        # training data with attacked feature (after transformation)
        x_train = np.concatenate((x_train_for_attack[:, :attack_feature], x_train_feature), axis=1)
        x_train = np.concatenate((x_train, x_train_for_attack[:, attack_feature:]), axis=1)

        # test data without attacked feature
        x_test_for_attack = np.delete(x_test_iris, attack_feature, 1)
        # only attacked feature
        x_test_feature = x_test_iris[:, attack_feature].copy().reshape(-1, 1)
        transform_feature(x_test_feature)
        # test data with attacked feature (after transformation)
        x_test = np.concatenate((x_test_for_attack[:, :attack_feature], x_test_feature), axis=1)
        x_test = np.concatenate((x_test, x_test_for_attack[:, attack_feature:]), axis=1)

        classifier = tabular_dl_estimator_for_attack(AttributeInferenceMembership)

        meminf_attack = MembershipInferenceBlackBox(classifier, attack_model_type="nn")
        attack_train_ratio = 0.5
        attack_train_size = int(len(x_train) * attack_train_ratio)
        attack_test_size = int(len(x_test) * attack_train_ratio)
        meminf_attack.fit(
            x_train[:attack_train_size],
            y_train_iris[:attack_train_size],
            x_test[:attack_test_size],
            y_test_iris[:attack_test_size],
        )
        attack = AttributeInferenceMembership(classifier, meminf_attack, attack_feature=attack_feature)
        # infer attacked feature
        inferred_train = attack.infer(x_train_for_attack, y_train_iris, values=values)
        inferred_test = attack.infer(x_test_for_attack, y_test_iris, values=values)
        # check accuracy
        train_acc = np.sum(inferred_train == x_train_feature.reshape(1, -1)) / len(inferred_train)
        test_acc = np.sum(inferred_test == x_test_feature.reshape(1, -1)) / len(inferred_test)
        assert 0.085 <= train_acc
        assert 0.04 <= test_acc

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("dl_frameworks")
def test_meminf_rule_based(art_warning, decision_tree_estimator, get_iris_dataset):
    try:
        attack_feature = 2  # petal length

        # need to transform attacked feature into categorical
        def transform_feature(x):
            x[x > 0.5] = 0.6
            x[(x > 0.2) & (x <= 0.5)] = 0.35
            x[x <= 0.2] = 0.1

        values = [0.1, 0.35, 0.6]

        (x_train_iris, y_train_iris), (x_test_iris, y_test_iris) = get_iris_dataset
        # training data without attacked feature
        x_train_for_attack = np.delete(x_train_iris, attack_feature, 1)
        # only attacked feature
        x_train_feature = x_train_iris[:, attack_feature].copy().reshape(-1, 1)
        transform_feature(x_train_feature)

        # test data without attacked feature
        x_test_for_attack = np.delete(x_test_iris, attack_feature, 1)
        # only attacked feature
        x_test_feature = x_test_iris[:, attack_feature].copy().reshape(-1, 1)
        transform_feature(x_test_feature)

        classifier = decision_tree_estimator()

        meminf_attack = MembershipInferenceBlackBoxRuleBased(classifier)
        attack = AttributeInferenceMembership(classifier, meminf_attack, attack_feature=attack_feature)
        # infer attacked feature
        inferred_train = attack.infer(x_train_for_attack, y_train_iris, values=values)
        inferred_test = attack.infer(x_test_for_attack, y_test_iris, values=values)
        # check accuracy
        train_acc = np.sum(inferred_train == x_train_feature) / len(inferred_train)
        test_acc = np.sum(inferred_test == x_test_feature) / len(inferred_test)
        assert 0.1 <= train_acc
        assert 0.1 <= test_acc

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("dl_frameworks")
def test_black_box_one_hot_float(art_warning, get_iris_dataset):
    try:
        attack_feature = 2  # petal length

        # need to transform attacked feature into categorical
        def transform_feature(x):
            x[x > 0.5] = 2
            x[(x > 0.2) & (x <= 0.5)] = 1
            x[x <= 0.2] = 0

        (x_train_iris, y_train_iris), (x_test_iris, y_test_iris) = get_iris_dataset
        # training data without attacked feature
        x_train_for_attack = np.delete(x_train_iris, attack_feature, 1)
        # only attacked feature
        x_train_feature = x_train_iris[:, attack_feature].copy().reshape(-1, 1)
        transform_feature(x_train_feature)
        # transform to one-hot encoding
        num_columns = int(x_train_feature.max()) + 1
        train_one_hot = np.zeros((x_train_feature.size, num_columns))
        train_one_hot[np.arange(x_train_feature.size), x_train_feature.reshape(1, -1).astype(int)] = 1
        # training data with attacked feature (after transformation)
        x_train = np.concatenate((x_train_for_attack[:, :attack_feature], train_one_hot), axis=1)
        x_train = np.concatenate((x_train, x_train_for_attack[:, attack_feature:]), axis=1)

        y_train = np.array([np.argmax(y) for y in y_train_iris]).reshape(-1, 1)

        # test data without attacked feature
        x_test_for_attack = np.delete(x_test_iris, attack_feature, 1)
        # only attacked feature
        x_test_feature = x_test_iris[:, attack_feature].copy().reshape(-1, 1)
        transform_feature(x_test_feature)
        # transform to one-hot encoding
        test_one_hot = np.zeros((x_test_feature.size, int(x_test_feature.max()) + 1))
        test_one_hot[np.arange(x_test_feature.size), x_test_feature.reshape(1, -1).astype(int)] = 1
        # test data with attacked feature (after transformation)
        x_test = np.concatenate((x_test_for_attack[:, :attack_feature], test_one_hot), axis=1)
        x_test = np.concatenate((x_test, x_test_for_attack[:, attack_feature:]), axis=1)

        # scale before training
        scaler = StandardScaler().fit(x_train)
        x_test = scaler.transform(x_test).astype(np.float32)
        x_train = scaler.transform(x_train).astype(np.float32)
        # derive dataset for attack (after scaling)
        attack_feature = slice(attack_feature, attack_feature + 3)
        x_train_for_attack = np.delete(x_train, attack_feature, 1)
        x_test_for_attack = np.delete(x_test, attack_feature, 1)
        train_one_hot = x_train[:, attack_feature]
        test_one_hot = x_test[:, attack_feature]

        tree = DecisionTreeClassifier()
        tree.fit(x_train, y_train)
        classifier = ScikitlearnDecisionTreeClassifier(tree)

        meminf_attack = MembershipInferenceBlackBox(classifier, attack_model_type="nn")
        attack_train_ratio = 0.5
        attack_train_size = int(len(x_train) * attack_train_ratio)
        attack_test_size = int(len(x_test) * attack_train_ratio)
        meminf_attack.fit(
            x_train[:attack_train_size],
            y_train_iris[:attack_train_size],
            x_test[:attack_test_size],
            y_test_iris[:attack_test_size],
        )
        attack = AttributeInferenceMembership(classifier, meminf_attack, attack_feature=attack_feature)
        # infer attacked feature
        values = [[-0.559017, 1.7888544], [-0.47003216, 2.127514], [-1.1774395, 0.84930056]]
        inferred_train = attack.infer(x_train_for_attack, y_train_iris, values=values)
        inferred_test = attack.infer(x_test_for_attack, y_test_iris, values=values)
        # check accuracy
        train_acc = np.sum(
            np.all(np.around(inferred_train, decimals=3) == np.around(train_one_hot, decimals=3), axis=1)
        ) / len(inferred_train)
        test_acc = np.sum(
            np.all(np.around(inferred_test, decimals=3) == np.around(test_one_hot, decimals=3), axis=1)
        ) / len(inferred_test)
        assert 0.1 <= train_acc
        assert 0.1 <= test_acc

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("dl_frameworks")
def test_meminf_label_only(art_warning, decision_tree_estimator, get_iris_dataset):
    try:
        attack_feature = 2  # petal length

        # need to transform attacked feature into categorical
        def transform_feature(x):
            x[x > 0.5] = 0.6
            x[(x > 0.2) & (x <= 0.5)] = 0.35
            x[x <= 0.2] = 0.1

        values = [0.1, 0.35, 0.6]

        (x_train_iris, y_train_iris), (x_test_iris, y_test_iris) = get_iris_dataset
        # training data without attacked feature
        x_train_for_attack = np.delete(x_train_iris, attack_feature, 1)
        # only attacked feature
        x_train_feature = x_train_iris[:, attack_feature].copy().reshape(-1, 1)
        transform_feature(x_train_feature)
        # training data with attacked feature (after transformation)
        x_train = np.concatenate((x_train_for_attack[:, :attack_feature], x_train_feature), axis=1)
        x_train = np.concatenate((x_train, x_train_for_attack[:, attack_feature:]), axis=1)

        # test data without attacked feature
        x_test_for_attack = np.delete(x_test_iris, attack_feature, 1)
        # only attacked feature
        x_test_feature = x_test_iris[:, attack_feature].copy().reshape(-1, 1)
        transform_feature(x_test_feature)
        # test data with attacked feature (after transformation)
        x_test = np.concatenate((x_test_for_attack[:, :attack_feature], x_test_feature), axis=1)
        x_test = np.concatenate((x_test, x_test_for_attack[:, attack_feature:]), axis=1)

        classifier = decision_tree_estimator()

        meminf_attack = LabelOnlyDecisionBoundary(classifier, distance_threshold_tau=0.5)
        kwargs = {
            "norm": 2,
            "max_iter": 2,
            "max_eval": 4,
            "init_eval": 1,
            "init_size": 1,
            "verbose": False,
        }
        attack_train_ratio = 0.5
        attack_train_size = int(len(x_train) * attack_train_ratio)
        attack_test_size = int(len(x_test) * attack_train_ratio)
        # attack without calibration
        attack = AttributeInferenceMembership(classifier, meminf_attack, attack_feature=attack_feature)
        # infer attacked feature
        inferred_train = attack.infer(x_train_for_attack, y_train_iris, values=values)
        inferred_test = attack.infer(x_test_for_attack, y_test_iris, values=values)
        # check accuracy
        train_acc = np.sum(inferred_train == x_train_feature) / len(inferred_train)
        test_acc = np.sum(inferred_test == x_test_feature) / len(inferred_test)
        assert 0.5 <= train_acc
        assert 0.5 <= test_acc

        # attack with calibration
        meminf_attack.calibrate_distance_threshold(
            x_train[:attack_train_size],
            y_train_iris[:attack_train_size],
            x_test[:attack_test_size],
            y_test_iris[:attack_test_size],
            **kwargs
        )
        attack = AttributeInferenceMembership(classifier, meminf_attack, attack_feature=attack_feature)
        # infer attacked feature
        inferred_train = attack.infer(x_train_for_attack, y_train_iris, values=values)
        inferred_test = attack.infer(x_test_for_attack, y_test_iris, values=values)
        # check accuracy
        train_acc = np.sum(inferred_train == x_train_feature) / len(inferred_train)
        test_acc = np.sum(inferred_test == x_test_feature) / len(inferred_test)
        assert 0.1 <= train_acc
        assert 0.1 <= test_acc

    except ARTTestException as e:
        art_warning(e)


def test_errors(art_warning, tabular_dl_estimator_for_attack, get_iris_dataset):
    try:
        classifier = tabular_dl_estimator_for_attack(AttributeInferenceMembership)
        (x_train, y_train), (x_test, y_test) = get_iris_dataset
        meminf_attack = MembershipInferenceBlackBox(classifier, attack_model_type="nn")

        with pytest.raises(ValueError):
            AttributeInferenceMembership(classifier, meminf_attack, attack_feature="a")
        with pytest.raises(ValueError):
            AttributeInferenceMembership(classifier, meminf_attack, attack_feature=-3)

        attack = AttributeInferenceMembership(classifier, meminf_attack)
        with pytest.raises(ValueError):
            AttributeInferenceMembership(classifier, attack, attack_feature=1)
        with pytest.raises(ValueError):
            attack.infer(x_train, y_test, values=[1, 2])
        with pytest.raises(ValueError):
            attack.infer(x_train, y_train)
    except ARTTestException as e:
        art_warning(e)
