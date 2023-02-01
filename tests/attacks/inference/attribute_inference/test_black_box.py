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
import torch.nn as nn
import torch.optim as optim
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from art.attacks.inference.attribute_inference.black_box import AttributeInferenceBlackBox
from art.estimators.classification.pytorch import PyTorchClassifier
from art.estimators.estimator import BaseEstimator
from art.estimators.classification import ClassifierMixin
from art.estimators.classification.scikitlearn import ScikitlearnDecisionTreeClassifier, ScikitlearnClassifier
from art.estimators.regression import ScikitlearnRegressor, RegressorMixin
from art.utils import check_and_transform_label_format

from tests.attacks.utils import backend_test_classifier_type_check_fail
from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.mark.skip_framework("dl_frameworks")
@pytest.mark.parametrize("model_type", ["nn", "rf"])
def test_black_box(art_warning, decision_tree_estimator, get_iris_dataset, model_type):
    try:
        attack_feature = 2  # petal length

        # need to transform attacked feature into categorical
        def transform_feature(x):
            x[x > 0.5] = 2.0
            x[(x > 0.2) & (x <= 0.5)] = 1.0
            x[x <= 0.2] = 0.0

        values = [0.0, 1.0, 2.0]

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

        classifier = decision_tree_estimator()

        attack = AttributeInferenceBlackBox(classifier, attack_feature=attack_feature, attack_model_type=model_type)
        # get original model's predictions
        x_train_predictions = np.array([np.argmax(arr) for arr in classifier.predict(x_train_iris)]).reshape(-1, 1)
        x_test_predictions = np.array([np.argmax(arr) for arr in classifier.predict(x_test_iris)]).reshape(-1, 1)
        # train attack model
        attack.fit(x_train)
        # infer attacked feature
        inferred_train = attack.infer(x_train_for_attack, pred=x_train_predictions, values=values)
        inferred_test = attack.infer(x_test_for_attack, pred=x_test_predictions, values=values)
        # check accuracy
        train_acc = np.sum(inferred_train == x_train_feature.reshape(1, -1)) / len(inferred_train)
        test_acc = np.sum(inferred_test == x_test_feature.reshape(1, -1)) / len(inferred_test)
        assert pytest.approx(0.8285, abs=0.2) == train_acc
        assert pytest.approx(0.8888, abs=0.18) == test_acc

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("dl_frameworks")
@pytest.mark.parametrize("model_type", ["nn", "rf"])
def test_black_box_continuous(art_warning, decision_tree_estimator, get_iris_dataset, model_type):
    try:
        attack_feature = 2  # petal length

        (x_train_iris, y_train_iris), (x_test_iris, y_test_iris) = get_iris_dataset
        # training data without attacked feature
        x_train_for_attack = np.delete(x_train_iris, attack_feature, 1)
        # only attacked feature
        x_train_feature = x_train_iris[:, attack_feature].copy().reshape(-1, 1)

        # test data without attacked feature
        x_test_for_attack = np.delete(x_test_iris, attack_feature, 1)
        # only attacked feature
        x_test_feature = x_test_iris[:, attack_feature].copy().reshape(-1, 1)

        classifier = decision_tree_estimator()

        attack = AttributeInferenceBlackBox(
            classifier, attack_feature=attack_feature, attack_model_type=model_type, is_continuous=True
        )
        # get original model's predictions
        x_train_predictions = np.array([np.argmax(arr) for arr in classifier.predict(x_train_iris)]).reshape(-1, 1)
        x_test_predictions = np.array([np.argmax(arr) for arr in classifier.predict(x_test_iris)]).reshape(-1, 1)
        # train attack model
        attack.fit(x_train_iris)
        # infer attacked feature
        inferred_train = attack.infer(x_train_for_attack, pred=x_train_predictions)
        inferred_test = attack.infer(x_test_for_attack, pred=x_test_predictions)
        # check accuracy
        assert np.allclose(inferred_train, x_train_feature.reshape(1, -1), atol=0.4)
        assert np.allclose(inferred_test, x_test_feature.reshape(1, -1), atol=0.4)

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("dl_frameworks")
@pytest.mark.parametrize("model_type", ["nn", "rf"])
def test_black_box_slice(art_warning, decision_tree_estimator, get_iris_dataset, model_type):
    try:
        attack_feature = 2  # petal length

        # need to transform attacked feature into categorical
        def transform_feature(x):
            x[x > 0.5] = 2.0
            x[(x > 0.2) & (x <= 0.5)] = 1.0
            x[x <= 0.2] = 0.0

        values = [0.0, 1.0, 2.0]

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

        classifier = decision_tree_estimator()

        attack = AttributeInferenceBlackBox(
            classifier, attack_feature=slice(attack_feature, attack_feature + 1), attack_model_type=model_type
        )
        # get original model's predictions
        x_train_predictions = np.array([np.argmax(arr) for arr in classifier.predict(x_train_iris)]).reshape(-1, 1)
        x_test_predictions = np.array([np.argmax(arr) for arr in classifier.predict(x_test_iris)]).reshape(-1, 1)
        # train attack model
        attack.fit(x_train)
        # infer attacked feature
        inferred_train = attack.infer(x_train_for_attack, pred=x_train_predictions, values=values)
        inferred_test = attack.infer(x_test_for_attack, pred=x_test_predictions, values=values)
        # check accuracy
        train_acc = np.sum(inferred_train == x_train_feature.reshape(1, -1)) / len(inferred_train)
        test_acc = np.sum(inferred_test == x_test_feature.reshape(1, -1)) / len(inferred_test)
        assert pytest.approx(0.8285, abs=0.12) == train_acc
        assert pytest.approx(0.8888, abs=0.18) == test_acc

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("dl_frameworks")
@pytest.mark.parametrize("model_type", ["nn", "rf"])
def test_black_box_with_label(art_warning, decision_tree_estimator, get_iris_dataset, model_type):
    try:
        attack_feature = 2  # petal length

        # need to transform attacked feature into categorical
        def transform_feature(x):
            x[x > 0.5] = 2.0
            x[(x > 0.2) & (x <= 0.5)] = 1.0
            x[x <= 0.2] = 0.0

        values = [0.0, 1.0, 2.0]

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

        classifier = decision_tree_estimator()

        attack = AttributeInferenceBlackBox(classifier, attack_feature=attack_feature, attack_model_type=model_type)
        # get original model's predictions
        x_train_predictions = np.array([np.argmax(arr) for arr in classifier.predict(x_train_iris)]).reshape(-1, 1)
        x_test_predictions = np.array([np.argmax(arr) for arr in classifier.predict(x_test_iris)]).reshape(-1, 1)
        # train attack model
        attack.fit(x_train, y=y_train_iris)
        # infer attacked feature
        inferred_train = attack.infer(x_train_for_attack, y=y_train_iris, pred=x_train_predictions, values=values)
        inferred_test = attack.infer(x_test_for_attack, y=y_test_iris, pred=x_test_predictions, values=values)
        # check accuracy
        train_acc = np.sum(inferred_train == x_train_feature.reshape(1, -1)) / len(inferred_train)
        test_acc = np.sum(inferred_test == x_test_feature.reshape(1, -1)) / len(inferred_test)
        assert pytest.approx(0.8285, abs=0.12) == train_acc
        assert pytest.approx(0.8888, abs=0.18) == test_acc

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("dl_frameworks")
@pytest.mark.parametrize("model_type", ["nn", "rf"])
def test_black_box_no_values(art_warning, decision_tree_estimator, get_iris_dataset, model_type):
    try:
        attack_feature = 2  # petal length

        # need to transform attacked feature into categorical
        def transform_feature(x):
            x[x > 0.5] = 2.0
            x[(x > 0.2) & (x <= 0.5)] = 1.0
            x[x <= 0.2] = 0.0

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

        classifier = decision_tree_estimator()

        attack = AttributeInferenceBlackBox(classifier, attack_feature=attack_feature, attack_model_type=model_type)
        # get original model's predictions
        x_train_predictions = np.array([np.argmax(arr) for arr in classifier.predict(x_train_iris)]).reshape(-1, 1)
        x_test_predictions = np.array([np.argmax(arr) for arr in classifier.predict(x_test_iris)]).reshape(-1, 1)
        # train attack model
        attack.fit(x_train)
        # infer attacked feature
        inferred_train = attack.infer(x_train_for_attack, pred=x_train_predictions)
        inferred_test = attack.infer(x_test_for_attack, pred=x_test_predictions)
        # check accuracy
        train_acc = np.sum(inferred_train == x_train_feature.reshape(1, -1)) / len(inferred_train)
        test_acc = np.sum(inferred_test == x_test_feature.reshape(1, -1)) / len(inferred_test)
        assert pytest.approx(0.8285, abs=0.12) == train_acc
        assert pytest.approx(0.8888, abs=0.18) == test_acc

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("dl_frameworks")
@pytest.mark.parametrize("model_type", ["nn", "rf"])
def test_black_box_regressor(art_warning, get_diabetes_dataset, model_type):
    try:
        attack_feature = 0  # age

        bins = [
            -0.96838121,
            -0.77154309,
            -0.57470497,
            -0.37786684,
            -0.18102872,
            0.0158094,
            0.21264752,
            0.40948564,
            0.60632376,
            0.80316188,
            1.0,
        ]

        # need to transform attacked feature into categorical
        def transform_feature(x):
            for i in range(len(bins) - 1):
                x[(x >= bins[i]) & (x <= bins[i + 1])] = i

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

        from sklearn import linear_model

        regr_model = linear_model.LinearRegression()
        regr_model.fit(x_train_diabetes, y_train_diabetes)
        regressor = ScikitlearnRegressor(regr_model)

        attack = AttributeInferenceBlackBox(
            regressor, attack_feature=attack_feature, prediction_normal_factor=1 / 250, attack_model_type=model_type
        )
        # get original model's predictions
        x_train_predictions = regressor.predict(x_train_diabetes).reshape(-1, 1)
        x_test_predictions = regressor.predict(x_test_diabetes).reshape(-1, 1)
        # train attack model
        attack.fit(x_train)
        # infer attacked feature
        inferred_train = attack.infer(x_train_for_attack, pred=x_train_predictions, values=values)
        inferred_test = attack.infer(x_test_for_attack, pred=x_test_predictions, values=values)
        # check accuracy
        train_acc = np.sum(inferred_train == x_train_feature.reshape(1, -1)) / len(inferred_train)
        test_acc = np.sum(inferred_test == x_test_feature.reshape(1, -1)) / len(inferred_test)

        assert pytest.approx(0.0258, abs=0.12) == train_acc
        assert pytest.approx(0.0375, abs=0.12) == test_acc

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("dl_frameworks")
@pytest.mark.parametrize("model_type", ["nn", "rf"])
def test_black_box_regressor_label(art_warning, get_diabetes_dataset, model_type):
    try:
        attack_feature = 0  # age

        bins = [
            -0.96838121,
            -0.77154309,
            -0.57470497,
            -0.37786684,
            -0.18102872,
            0.0158094,
            0.21264752,
            0.40948564,
            0.60632376,
            0.80316188,
            1.0,
        ]

        # need to transform attacked feature into categorical
        def transform_feature(x):
            for i in range(len(bins) - 1):
                x[(x >= bins[i]) & (x <= bins[i + 1])] = i

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

        from sklearn import linear_model

        regr_model = linear_model.LinearRegression()
        regr_model.fit(x_train_diabetes, y_train_diabetes)
        regressor = ScikitlearnRegressor(regr_model)

        attack = AttributeInferenceBlackBox(
            regressor, attack_feature=attack_feature, prediction_normal_factor=1 / 250, attack_model_type=model_type
        )
        # get original model's predictions
        x_train_predictions = regressor.predict(x_train_diabetes).reshape(-1, 1)
        x_test_predictions = regressor.predict(x_test_diabetes).reshape(-1, 1)
        # train attack model
        attack.fit(x_train, y=y_train_diabetes)
        # infer attacked feature
        inferred_train = attack.infer(x_train_for_attack, pred=x_train_predictions, values=values, y=y_train_diabetes)
        inferred_test = attack.infer(x_test_for_attack, pred=x_test_predictions, values=values, y=y_test_diabetes)
        # check accuracy
        train_acc = np.sum(inferred_train == x_train_feature.reshape(1, -1)) / len(inferred_train)
        test_acc = np.sum(inferred_test == x_test_feature.reshape(1, -1)) / len(inferred_test)

        assert pytest.approx(0.0258, abs=0.12) == train_acc
        assert pytest.approx(0.0375, abs=0.12) == test_acc

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("dl_frameworks")
def test_black_box_with_model(art_warning, decision_tree_estimator, get_iris_dataset):
    try:
        attack_feature = 2  # petal length

        # need to transform attacked feature into categorical
        def transform_feature(x):
            x[x > 0.5] = 2.0
            x[(x > 0.2) & (x <= 0.5)] = 1.0
            x[x <= 0.2] = 0.0

        values = [0.0, 1.0, 2.0]

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

        model = nn.Linear(4, 3)

        # Define a loss function and optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        attack_model = PyTorchClassifier(
            model=model, clip_values=(0, 1), loss=loss_fn, optimizer=optimizer, input_shape=(4,), nb_classes=3
        )

        classifier = decision_tree_estimator()

        attack = AttributeInferenceBlackBox(classifier, attack_model=attack_model, attack_feature=attack_feature)
        # get original model's predictions
        x_train_predictions = np.array([np.argmax(arr) for arr in classifier.predict(x_train_iris)]).reshape(-1, 1)
        x_test_predictions = np.array([np.argmax(arr) for arr in classifier.predict(x_test_iris)]).reshape(-1, 1)
        # train attack model
        attack.fit(x_train)
        # infer attacked feature
        inferred_train = attack.infer(x_train_for_attack, pred=x_train_predictions, values=values)
        inferred_test = attack.infer(x_test_for_attack, pred=x_test_predictions, values=values)
        # check accuracy
        # train_acc
        _ = np.sum(inferred_train == x_train_feature.reshape(1, -1)) / len(inferred_train)
        # test_acc
        _ = np.sum(inferred_test == x_test_feature.reshape(1, -1)) / len(inferred_test)
        # assert train_acc == pytest.approx(0.5523, abs=0.03)
        # assert test_acc == pytest.approx(0.5777, abs=0.03)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("dl_frameworks")
@pytest.mark.parametrize("model_type", ["nn", "rf"])
def test_black_box_one_hot(art_warning, get_iris_dataset, model_type):
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
        train_one_hot = np.zeros((x_train_feature.size, int(x_train_feature.max()) + 1))
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

        tree = DecisionTreeClassifier()
        tree.fit(x_train, y_train)
        classifier = ScikitlearnDecisionTreeClassifier(tree)

        attack = AttributeInferenceBlackBox(
            classifier, attack_feature=slice(attack_feature, attack_feature + 3), attack_model_type=model_type
        )
        # get original model's predictions
        x_train_predictions = np.array([np.argmax(arr) for arr in classifier.predict(x_train)]).reshape(-1, 1)
        x_test_predictions = np.array([np.argmax(arr) for arr in classifier.predict(x_test)]).reshape(-1, 1)
        # train attack model
        attack.fit(x_train)
        # infer attacked feature
        inferred_train = attack.infer(x_train_for_attack, pred=x_train_predictions)
        inferred_test = attack.infer(x_test_for_attack, pred=x_test_predictions)
        # check accuracy
        train_acc = np.sum(np.all(inferred_train == train_one_hot, axis=1)) / len(inferred_train)
        test_acc = np.sum(np.all(inferred_test == test_one_hot, axis=1)) / len(inferred_test)
        assert pytest.approx(0.8666, abs=0.12) == train_acc
        assert pytest.approx(0.8888, abs=0.7) == test_acc

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("dl_frameworks")
@pytest.mark.parametrize("model_type", ["nn", "rf"])
def test_black_box_one_hot_float(art_warning, get_iris_dataset, model_type):
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

        attack = AttributeInferenceBlackBox(classifier, attack_feature=attack_feature, attack_model_type=model_type)
        # get original model's predictions
        x_train_predictions = np.array([np.argmax(arr) for arr in classifier.predict(x_train)]).reshape(-1, 1)
        x_test_predictions = np.array([np.argmax(arr) for arr in classifier.predict(x_test)]).reshape(-1, 1)
        # train attack model
        attack.fit(x_train)
        # infer attacked feature
        values = [[-0.559017, 1.7888544], [-0.47003216, 2.127514], [-1.1774395, 0.84930056]]
        inferred_train = attack.infer(x_train_for_attack, pred=x_train_predictions, values=values)
        inferred_test = attack.infer(x_test_for_attack, pred=x_test_predictions, values=values)
        # check accuracy
        train_acc = np.sum(
            np.all(np.around(inferred_train, decimals=3) == np.around(train_one_hot, decimals=3), axis=1)
        ) / len(inferred_train)
        test_acc = np.sum(
            np.all(np.around(inferred_test, decimals=3) == np.around(test_one_hot, decimals=3), axis=1)
        ) / len(inferred_test)
        assert pytest.approx(0.8666, abs=0.12) == train_acc
        assert pytest.approx(0.8666, abs=0.1) == test_acc

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("dl_frameworks")
@pytest.mark.parametrize("model_type", ["nn", "rf"])
def test_black_box_one_hot_float_no_values(art_warning, get_iris_dataset, model_type):
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

        attack = AttributeInferenceBlackBox(classifier, attack_feature=attack_feature, attack_model_type=model_type)
        # get original model's predictions
        x_train_predictions = np.array([np.argmax(arr) for arr in classifier.predict(x_train)]).reshape(-1, 1)
        x_test_predictions = np.array([np.argmax(arr) for arr in classifier.predict(x_test)]).reshape(-1, 1)
        # train attack model
        attack.fit(x_train)
        # infer attacked feature
        inferred_train = attack.infer(x_train_for_attack, pred=x_train_predictions)
        inferred_test = attack.infer(x_test_for_attack, pred=x_test_predictions)
        # check accuracy
        train_acc = np.sum(
            np.all(np.around(inferred_train, decimals=3) == np.around(train_one_hot, decimals=3), axis=1)
        ) / len(inferred_train)
        test_acc = np.sum(
            np.all(np.around(inferred_test, decimals=3) == np.around(test_one_hot, decimals=3), axis=1)
        ) / len(inferred_test)
        assert pytest.approx(0.8666, abs=0.12) == train_acc
        assert pytest.approx(0.8666, abs=0.1) == test_acc

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("dl_frameworks")
@pytest.mark.parametrize("model_type", ["nn", "rf"])
def test_black_box_baseline_encoder(art_warning, get_iris_dataset, model_type):
    try:
        attack_feature = 2  # petal length

        # need to transform attacked feature into categorical
        def transform_attacked_feature(x):
            x[x > 0.5] = 2.0
            x[(x > 0.2) & (x <= 0.5)] = 1.0
            x[x <= 0.2] = 0.0

        def transform_other_feature(x):
            x[x > 0.5] = 2.0
            x[(x > 0.2) & (x <= 0.5)] = 1.0
            x[x <= 0.2] = 0.0
            x[x == 2.0] = "A"
            x[x == 1.0] = "B"
            x[x == 0.0] = "C"

        values = [0.0, 1.0, 2.0]

        (x_train_iris, y_train_iris), (x_test_iris, y_test_iris) = get_iris_dataset

        # training data without attacked feature
        x_train_for_attack = np.delete(x_train_iris, attack_feature, 1)
        # transform attacked feature
        x_train_feature = x_train_iris[:, attack_feature].copy().reshape(-1, 1)
        transform_attacked_feature(x_train_feature)
        # training data with attacked feature (after transformation)
        x_train = np.concatenate((x_train_for_attack[:, :attack_feature], x_train_feature), axis=1)
        x_train = np.concatenate((x_train, x_train_for_attack[:, attack_feature:]), axis=1)

        # transform other feature
        other_feature = 1
        x_without_feature = np.delete(x_train, other_feature, 1)
        x_other_feature = x_train_iris[:, other_feature].copy().reshape(-1, 1).astype(object)
        transform_other_feature(x_other_feature)
        # training data with other feature (after transformation)
        x_train = np.concatenate((x_without_feature[:, :other_feature], x_other_feature), axis=1)
        x_train = np.concatenate((x_train, x_without_feature[:, other_feature:]), axis=1)

        x_train_for_attack_without_feature = np.delete(x_train_for_attack, other_feature, 1)
        x_train_for_attack = np.concatenate(
            (x_train_for_attack_without_feature[:, :other_feature], x_other_feature), axis=1
        )
        x_train_for_attack = np.concatenate(
            (x_train_for_attack, x_train_for_attack_without_feature[:, other_feature:]), axis=1
        )

        # test data without attacked feature
        x_test_for_attack = np.delete(x_test_iris, attack_feature, 1)
        # only attacked feature
        x_test_feature = x_test_iris[:, attack_feature].copy().reshape(-1, 1)
        transform_attacked_feature(x_test_feature)

        # transform other feature
        x_test_without_feature = np.delete(x_test_for_attack, other_feature, 1)
        x_test_other_feature = x_test_iris[:, other_feature].copy().reshape(-1, 1).astype(object)
        transform_other_feature(x_test_other_feature)
        # training data with other feature (after transformation)
        x_test_for_attack = np.concatenate((x_test_without_feature[:, :other_feature], x_test_other_feature), axis=1)
        x_test_for_attack = np.concatenate((x_test_for_attack, x_test_without_feature[:, other_feature:]), axis=1)

        # transform other feature for full test data
        x_test = np.delete(x_test_iris, other_feature, 1)
        # test data with other feature (after transformation)
        x_test_for_pred = np.concatenate((x_test[:, :other_feature], x_test_other_feature), axis=1)
        x_test_for_pred = np.concatenate((x_test_for_pred, x_test[:, other_feature:]), axis=1)

        categorical_transformer = OrdinalEncoder()
        encoder = ColumnTransformer(
            transformers=[
                ("cat", categorical_transformer, [other_feature]),
            ],
            remainder="passthrough",
        )
        encoder.fit(x_train_for_attack)

        encoder_for_pipeline = ColumnTransformer(
            transformers=[
                ("cat", categorical_transformer, [other_feature]),
            ],
            remainder="passthrough",
        )
        model = DecisionTreeClassifier()
        pipeline = Pipeline([("encoder", encoder_for_pipeline), ("model", model)])
        pipeline.fit(x_train, np.argmax(y_train_iris, axis=1))
        classifier = ScikitlearnClassifier(pipeline, preprocessing=None)

        baseline_attack = AttributeInferenceBlackBox(
            classifier, attack_feature=attack_feature, attack_model_type=model_type, encoder=encoder
        )
        # train attack model
        baseline_attack.fit(x_train, y_train_iris)
        # infer attacked feature
        x_train_predictions = np.array([np.argmax(arr) for arr in classifier.predict(x_train)]).reshape(-1, 1)
        x_test_predictions = np.array([np.argmax(arr) for arr in classifier.predict(x_test_for_pred)]).reshape(-1, 1)
        baseline_inferred_train = baseline_attack.infer(
            x_train_for_attack, y_train_iris, pred=x_train_predictions, values=values
        )
        baseline_inferred_test = baseline_attack.infer(
            x_test_for_attack, y_test_iris, pred=x_test_predictions, values=values
        )
        # check accuracy
        baseline_train_acc = np.sum(baseline_inferred_train == x_train_feature.reshape(1, -1)) / len(
            baseline_inferred_train
        )
        baseline_test_acc = np.sum(baseline_inferred_test == x_test_feature.reshape(1, -1)) / len(
            baseline_inferred_test
        )

        assert 0.6 <= baseline_train_acc
        assert 0.6 <= baseline_test_acc

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("dl_frameworks")
@pytest.mark.parametrize("model_type", ["nn", "rf"])
def test_black_box_baseline_no_encoder(art_warning, get_iris_dataset, model_type):
    try:
        attack_feature = 2  # petal length

        # need to transform attacked feature into categorical
        def transform_attacked_feature(x):
            x[x > 0.5] = 2.0
            x[(x > 0.2) & (x <= 0.5)] = 1.0
            x[x <= 0.2] = 0.0

        def transform_other_feature(x):
            x[x > 0.5] = 2.0
            x[(x > 0.2) & (x <= 0.5)] = 1.0
            x[x <= 0.2] = 0.0
            x[x == 2.0] = "A"
            x[x == 1.0] = "B"
            x[x == 0.0] = "C"

        values = [0.0, 1.0, 2.0]

        (x_train_iris, y_train_iris), (x_test_iris, y_test_iris) = get_iris_dataset

        # training data without attacked feature
        x_train_for_attack = np.delete(x_train_iris, attack_feature, 1)
        # transform attacked feature
        x_train_feature = x_train_iris[:, attack_feature].copy().reshape(-1, 1)
        transform_attacked_feature(x_train_feature)
        # training data with attacked feature (after transformation)
        x_train = np.concatenate((x_train_for_attack[:, :attack_feature], x_train_feature), axis=1)
        x_train = np.concatenate((x_train, x_train_for_attack[:, attack_feature:]), axis=1)

        # transform other feature
        other_feature = 1
        x_without_feature = np.delete(x_train, other_feature, 1)
        x_other_feature = x_train_iris[:, other_feature].copy().reshape(-1, 1).astype(object)
        transform_other_feature(x_other_feature)
        # training data with other feature (after transformation)
        x_train = np.concatenate((x_without_feature[:, :other_feature], x_other_feature), axis=1)
        x_train = np.concatenate((x_train, x_without_feature[:, other_feature:]), axis=1)

        x_train_for_attack_without_feature = np.delete(x_train_for_attack, other_feature, 1)
        x_train_for_attack = np.concatenate(
            (x_train_for_attack_without_feature[:, :other_feature], x_other_feature), axis=1
        )
        x_train_for_attack = np.concatenate(
            (x_train_for_attack, x_train_for_attack_without_feature[:, other_feature:]), axis=1
        )

        # test data without attacked feature
        x_test_for_attack = np.delete(x_test_iris, attack_feature, 1)
        # only attacked feature
        x_test_feature = x_test_iris[:, attack_feature].copy().reshape(-1, 1)
        transform_attacked_feature(x_test_feature)

        # transform other feature
        x_test_without_feature = np.delete(x_test_for_attack, other_feature, 1)
        x_test_other_feature = x_test_iris[:, other_feature].copy().reshape(-1, 1).astype(object)
        transform_other_feature(x_test_other_feature)
        # training data with other feature (after transformation)
        x_test_for_attack = np.concatenate((x_test_without_feature[:, :other_feature], x_test_other_feature), axis=1)
        x_test_for_attack = np.concatenate((x_test_for_attack, x_test_without_feature[:, other_feature:]), axis=1)

        # transform other feature for full test data
        x_test = np.delete(x_test_iris, other_feature, 1)
        # test data with other feature (after transformation)
        x_test_for_pred = np.concatenate((x_test[:, :other_feature], x_test_other_feature), axis=1)
        x_test_for_pred = np.concatenate((x_test_for_pred, x_test[:, other_feature:]), axis=1)

        categorical_transformer = OrdinalEncoder()
        encoder = ColumnTransformer(
            transformers=[
                ("cat", categorical_transformer, [other_feature]),
            ],
            remainder="passthrough",
        )
        encoder.fit(x_train_for_attack)

        encoder_for_pipeline = ColumnTransformer(
            transformers=[
                ("cat", categorical_transformer, [other_feature]),
            ],
            remainder="passthrough",
        )
        model = DecisionTreeClassifier()
        pipeline = Pipeline([("encoder", encoder_for_pipeline), ("model", model)])
        pipeline.fit(x_train, np.argmax(y_train_iris, axis=1))
        classifier = ScikitlearnClassifier(pipeline, preprocessing=None)

        baseline_attack = AttributeInferenceBlackBox(
            classifier,
            attack_feature=attack_feature,
            attack_model_type=model_type,
            non_numerical_features=[other_feature],
        )
        # train attack model
        baseline_attack.fit(x_train, y_train_iris)
        # infer attacked feature
        x_train_predictions = np.array([np.argmax(arr) for arr in classifier.predict(x_train)]).reshape(-1, 1)
        x_test_predictions = np.array([np.argmax(arr) for arr in classifier.predict(x_test_for_pred)]).reshape(-1, 1)
        baseline_inferred_train = baseline_attack.infer(
            x_train_for_attack, y_train_iris, pred=x_train_predictions, values=values
        )
        baseline_inferred_test = baseline_attack.infer(
            x_test_for_attack, y_test_iris, pred=x_test_predictions, values=values
        )
        # check accuracy
        baseline_train_acc = np.sum(baseline_inferred_train == x_train_feature.reshape(1, -1)) / len(
            baseline_inferred_train
        )
        baseline_test_acc = np.sum(baseline_inferred_test == x_test_feature.reshape(1, -1)) / len(
            baseline_inferred_test
        )

        assert 0.6 <= baseline_train_acc
        assert 0.6 <= baseline_test_acc

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("dl_frameworks")
@pytest.mark.parametrize("model_type", ["nn", "rf"])
def test_black_box_baseline_no_encoder_after_feature(art_warning, get_iris_dataset, model_type):
    try:
        attack_feature = 2  # petal length

        # need to transform attacked feature into categorical
        def transform_attacked_feature(x):
            x[x > 0.5] = 2.0
            x[(x > 0.2) & (x <= 0.5)] = 1.0
            x[x <= 0.2] = 0.0

        def transform_other_feature(x):
            x[x > 0.3] = 2.0
            x[(x > 0.2) & (x <= 0.3)] = 1.0
            x[x <= 0.2] = 0.0
            x[x == 2.0] = "A"
            x[x == 1.0] = "B"
            x[x == 0.0] = "C"

        values = [0.0, 1.0, 2.0]

        (x_train_iris, y_train_iris), (x_test_iris, y_test_iris) = get_iris_dataset

        # training data without attacked feature
        x_train_for_attack = np.delete(x_train_iris, attack_feature, 1)
        # transform attacked feature
        x_train_feature = x_train_iris[:, attack_feature].copy().reshape(-1, 1)
        transform_attacked_feature(x_train_feature)
        # training data with attacked feature (after transformation)
        x_train = np.concatenate((x_train_for_attack[:, :attack_feature], x_train_feature), axis=1)
        x_train = np.concatenate((x_train, x_train_for_attack[:, attack_feature:]), axis=1)

        # transform other feature
        other_feature = 3
        x_without_feature = np.delete(x_train, other_feature, 1)
        x_other_feature = x_train_iris[:, other_feature].copy().reshape(-1, 1).astype(object)
        transform_other_feature(x_other_feature)
        # training data with other feature (after transformation)
        x_train = np.concatenate((x_without_feature[:, :other_feature], x_other_feature), axis=1)
        x_train = np.concatenate((x_train, x_without_feature[:, other_feature:]), axis=1)

        new_other_feature = other_feature - 1
        x_train_for_attack_without_feature = np.delete(x_train_for_attack, new_other_feature, 1)
        x_train_for_attack = np.concatenate(
            (x_train_for_attack_without_feature[:, :new_other_feature], x_other_feature), axis=1
        )
        x_train_for_attack = np.concatenate(
            (x_train_for_attack, x_train_for_attack_without_feature[:, new_other_feature:]), axis=1
        )

        # test data without attacked feature
        x_test_for_attack = np.delete(x_test_iris, attack_feature, 1)
        # only attacked feature
        x_test_feature = x_test_iris[:, attack_feature].copy().reshape(-1, 1)
        transform_attacked_feature(x_test_feature)

        # transform other feature
        x_test_without_feature = np.delete(x_test_for_attack, new_other_feature, 1)
        x_test_other_feature = x_test_iris[:, new_other_feature].copy().reshape(-1, 1).astype(object)
        transform_other_feature(x_test_other_feature)
        # training data with other feature (after transformation)
        x_test_for_attack = np.concatenate(
            (x_test_without_feature[:, :new_other_feature], x_test_other_feature), axis=1
        )
        x_test_for_attack = np.concatenate((x_test_for_attack, x_test_without_feature[:, new_other_feature:]), axis=1)

        # transform other feature for full test data
        x_test = np.delete(x_test_iris, other_feature, 1)
        # test data with other feature (after transformation)
        x_test_for_pred = np.concatenate((x_test[:, :other_feature], x_test_other_feature), axis=1)
        x_test_for_pred = np.concatenate((x_test_for_pred, x_test[:, other_feature:]), axis=1)

        categorical_transformer = OrdinalEncoder()

        encoder_for_pipeline = ColumnTransformer(
            transformers=[
                ("cat", categorical_transformer, [other_feature]),
            ],
            remainder="passthrough",
        )
        model = DecisionTreeClassifier()
        pipeline = Pipeline([("encoder", encoder_for_pipeline), ("model", model)])
        pipeline.fit(x_train, np.argmax(y_train_iris, axis=1))
        classifier = ScikitlearnClassifier(pipeline, preprocessing=None)

        baseline_attack = AttributeInferenceBlackBox(
            classifier,
            attack_feature=attack_feature,
            attack_model_type=model_type,
            non_numerical_features=[other_feature],
        )
        # train attack model
        baseline_attack.fit(x_train, y_train_iris)
        # infer attacked feature
        x_train_predictions = np.array([np.argmax(arr) for arr in classifier.predict(x_train)]).reshape(-1, 1)
        x_test_predictions = np.array([np.argmax(arr) for arr in classifier.predict(x_test_for_pred)]).reshape(-1, 1)
        baseline_inferred_train = baseline_attack.infer(
            x_train_for_attack, y_train_iris, pred=x_train_predictions, values=values
        )
        baseline_inferred_test = baseline_attack.infer(
            x_test_for_attack, y_test_iris, pred=x_test_predictions, values=values
        )
        # check accuracy
        baseline_train_acc = np.sum(baseline_inferred_train == x_train_feature.reshape(1, -1)) / len(
            baseline_inferred_train
        )
        baseline_test_acc = np.sum(baseline_inferred_test == x_test_feature.reshape(1, -1)) / len(
            baseline_inferred_test
        )

        assert 0.5 <= baseline_train_acc
        assert 0.5 <= baseline_test_acc

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("dl_frameworks")
@pytest.mark.parametrize("model_type", ["nn", "rf"])
def test_black_box_baseline_no_encoder_after_feature_slice(art_warning, get_iris_dataset, model_type):
    try:
        orig_attack_feature = 1  # petal length
        new_attack_feature = slice(1, 4)  # petal length

        # need to transform attacked feature into categorical
        def transform_attacked_feature(x):
            x[x > 0.5] = 2.0
            x[(x > 0.2) & (x <= 0.5)] = 1.0
            x[x <= 0.2] = 0.0

        def transform_other_feature(x):
            x[x > 0.3] = 2.0
            x[(x > 0.2) & (x <= 0.3)] = 1.0
            x[x <= 0.2] = 0.0
            x[x == 2.0] = "A"
            x[x == 1.0] = "B"
            x[x == 0.0] = "C"

        (x_train_iris, y_train_iris), (x_test_iris, y_test_iris) = get_iris_dataset

        # training data without attacked feature
        x_train_for_attack = np.delete(x_train_iris, orig_attack_feature, 1)
        # transform attacked feature
        x_train_feature = x_train_iris[:, orig_attack_feature].copy()
        transform_attacked_feature(x_train_feature)
        x_train_feature = check_and_transform_label_format(x_train_feature, nb_classes=3, return_one_hot=True)
        # training data with attacked feature (after transformation)
        x_train = np.concatenate((x_train_for_attack[:, :orig_attack_feature], x_train_feature), axis=1)
        x_train = np.concatenate((x_train, x_train_for_attack[:, orig_attack_feature:]), axis=1)

        # transform other feature
        other_feature = 5  # was 3 before 1-hot encoding of attacked feature
        x_without_feature = np.delete(x_train, other_feature, 1)
        x_other_feature = x_train[:, other_feature].copy().reshape(-1, 1).astype(object)
        transform_other_feature(x_other_feature)
        # training data with other feature (after transformation)
        x_train = np.concatenate((x_without_feature[:, :other_feature], x_other_feature), axis=1)
        x_train = np.concatenate((x_train, x_without_feature[:, other_feature:]), axis=1)

        new_other_feature = other_feature - 3
        x_train_for_attack_without_feature = np.delete(x_train_for_attack, new_other_feature, 1)
        x_train_for_attack = np.concatenate(
            (x_train_for_attack_without_feature[:, :new_other_feature], x_other_feature), axis=1
        )
        x_train_for_attack = np.concatenate(
            (x_train_for_attack, x_train_for_attack_without_feature[:, new_other_feature:]), axis=1
        )

        # test data without attacked feature
        x_test_for_attack = np.delete(x_test_iris, orig_attack_feature, 1)
        # only attacked feature
        x_test_feature = x_test_iris[:, orig_attack_feature].copy()
        transform_attacked_feature(x_test_feature)
        x_test_feature = check_and_transform_label_format(x_test_feature, nb_classes=3, return_one_hot=True)

        # transform other feature
        x_test_without_feature = np.delete(x_test_for_attack, new_other_feature, 1)
        x_test_other_feature = x_test_for_attack[:, new_other_feature].copy().reshape(-1, 1).astype(object)
        transform_other_feature(x_test_other_feature)
        # training data with other feature (after transformation)
        x_test_for_attack = np.concatenate(
            (x_test_without_feature[:, :new_other_feature], x_test_other_feature), axis=1
        )
        x_test_for_attack = np.concatenate((x_test_for_attack, x_test_without_feature[:, new_other_feature:]), axis=1)

        # transform features for full test data
        x_test_without_feature = np.delete(x_test_iris, orig_attack_feature, 1)
        # test data with attacked feature (after transformation)
        x_test_with_feature = np.concatenate((x_test_without_feature[:, :orig_attack_feature], x_test_feature), axis=1)
        x_test_with_feature = np.concatenate(
            (x_test_with_feature, x_test_without_feature[:, orig_attack_feature:]), axis=1
        )
        x_test_without_feature = np.delete(x_test_with_feature, other_feature, 1)
        # test data with other feature (after transformation)
        x_test_for_pred = np.concatenate((x_test_without_feature[:, :other_feature], x_test_other_feature), axis=1)
        x_test_for_pred = np.concatenate((x_test_for_pred, x_test_without_feature[:, other_feature:]), axis=1)

        categorical_transformer = OrdinalEncoder()
        encoder_for_pipeline = ColumnTransformer(
            transformers=[
                ("cat", categorical_transformer, [other_feature]),
            ],
            remainder="passthrough",
        )
        model = DecisionTreeClassifier()
        pipeline = Pipeline([("encoder", encoder_for_pipeline), ("model", model)])
        pipeline.fit(x_train, np.argmax(y_train_iris, axis=1))
        classifier = ScikitlearnClassifier(pipeline, preprocessing=None)

        baseline_attack = AttributeInferenceBlackBox(
            classifier,
            attack_feature=new_attack_feature,
            attack_model_type=model_type,
            non_numerical_features=[other_feature],
        )
        # train attack model
        baseline_attack.fit(x_train, y_train_iris)
        # infer attacked feature
        x_train_predictions = np.array([np.argmax(arr) for arr in classifier.predict(x_train)]).reshape(-1, 1)
        x_test_predictions = np.array([np.argmax(arr) for arr in classifier.predict(x_test_for_pred)]).reshape(-1, 1)
        baseline_inferred_train = baseline_attack.infer(x_train_for_attack, y_train_iris, pred=x_train_predictions)
        baseline_inferred_test = baseline_attack.infer(x_test_for_attack, y_test_iris, pred=x_test_predictions)
        # check accuracy
        baseline_train_acc = np.sum(baseline_inferred_train == x_train_feature) / len(baseline_inferred_train)
        baseline_test_acc = np.sum(baseline_inferred_test == x_test_feature) / len(baseline_inferred_test)

        assert 0.0 <= baseline_train_acc
        assert 0.0 <= baseline_test_acc

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("dl_frameworks")
@pytest.mark.parametrize("model_type", ["nn", "rf"])
def test_black_box_baseline_no_encoder_remove_attack_feature(art_warning, get_iris_dataset, model_type):
    try:
        attack_feature = 2  # petal length

        # need to transform attacked feature into categorical
        def transform_attacked_feature(x):
            x[x > 0.5] = 2.0
            x[(x > 0.2) & (x <= 0.5)] = 1.0
            x[x <= 0.2] = 0.0

        def transform_other_feature(x):
            x[x > 0.5] = 2.0
            x[(x > 0.2) & (x <= 0.5)] = 1.0
            x[x <= 0.2] = 0.0
            x[x == 2.0] = "A"
            x[x == 1.0] = "B"
            x[x == 0.0] = "C"

        values = [0.0, 1.0, 2.0]

        (x_train_iris, y_train_iris), (x_test_iris, y_test_iris) = get_iris_dataset

        # training data without attacked feature
        x_train_for_attack = np.delete(x_train_iris, attack_feature, 1)
        # transform attacked feature
        x_train_feature = x_train_iris[:, attack_feature].copy().reshape(-1, 1)
        transform_attacked_feature(x_train_feature)
        # training data with attacked feature (after transformation)
        x_train = np.concatenate((x_train_for_attack[:, :attack_feature], x_train_feature), axis=1)
        x_train = np.concatenate((x_train, x_train_for_attack[:, attack_feature:]), axis=1)

        # transform other feature
        other_feature = 1
        x_without_feature = np.delete(x_train, other_feature, 1)
        x_other_feature = x_train_iris[:, other_feature].copy().reshape(-1, 1).astype(object)
        transform_other_feature(x_other_feature)
        # training data with other feature (after transformation)
        x_train = np.concatenate((x_without_feature[:, :other_feature], x_other_feature), axis=1)
        x_train = np.concatenate((x_train, x_without_feature[:, other_feature:]), axis=1)

        x_train_for_attack_without_feature = np.delete(x_train_for_attack, other_feature, 1)
        x_train_for_attack = np.concatenate(
            (x_train_for_attack_without_feature[:, :other_feature], x_other_feature), axis=1
        )
        x_train_for_attack = np.concatenate(
            (x_train_for_attack, x_train_for_attack_without_feature[:, other_feature:]), axis=1
        )

        # test data without attacked feature
        x_test_for_attack = np.delete(x_test_iris, attack_feature, 1)
        # only attacked feature
        x_test_feature = x_test_iris[:, attack_feature].copy().reshape(-1, 1)
        transform_attacked_feature(x_test_feature)

        # transform other feature
        x_test_without_feature = np.delete(x_test_for_attack, other_feature, 1)
        x_test_other_feature = x_test_iris[:, other_feature].copy().reshape(-1, 1).astype(object)
        transform_other_feature(x_test_other_feature)
        # training data with other feature (after transformation)
        x_test_for_attack = np.concatenate((x_test_without_feature[:, :other_feature], x_test_other_feature), axis=1)
        x_test_for_attack = np.concatenate((x_test_for_attack, x_test_without_feature[:, other_feature:]), axis=1)

        # transform other feature for full test data
        x_test = np.delete(x_test_iris, other_feature, 1)
        # test data with other feature (after transformation)
        x_test_for_pred = np.concatenate((x_test[:, :other_feature], x_test_other_feature), axis=1)
        x_test_for_pred = np.concatenate((x_test_for_pred, x_test[:, other_feature:]), axis=1)

        categorical_transformer = OrdinalEncoder()
        encoder = ColumnTransformer(
            transformers=[
                ("cat", categorical_transformer, [other_feature]),
            ],
            remainder="passthrough",
        )
        encoder.fit(x_train_for_attack)

        encoder_for_pipeline = ColumnTransformer(
            transformers=[
                ("cat", categorical_transformer, [other_feature]),
            ],
            remainder="passthrough",
        )
        model = DecisionTreeClassifier()
        pipeline = Pipeline([("encoder", encoder_for_pipeline), ("model", model)])
        pipeline.fit(x_train, np.argmax(y_train_iris, axis=1))
        classifier = ScikitlearnClassifier(pipeline, preprocessing=None)

        baseline_attack = AttributeInferenceBlackBox(
            classifier,
            attack_feature=attack_feature,
            attack_model_type=model_type,
            non_numerical_features=[other_feature, attack_feature],
        )
        # train attack model
        baseline_attack.fit(x_train, y_train_iris)
        # infer attacked feature
        x_train_predictions = np.array([np.argmax(arr) for arr in classifier.predict(x_train)]).reshape(-1, 1)
        x_test_predictions = np.array([np.argmax(arr) for arr in classifier.predict(x_test_for_pred)]).reshape(-1, 1)
        baseline_inferred_train = baseline_attack.infer(
            x_train_for_attack, y_train_iris, pred=x_train_predictions, values=values
        )
        baseline_inferred_test = baseline_attack.infer(
            x_test_for_attack, y_test_iris, pred=x_test_predictions, values=values
        )
        # check accuracy
        baseline_train_acc = np.sum(baseline_inferred_train == x_train_feature.reshape(1, -1)) / len(
            baseline_inferred_train
        )
        baseline_test_acc = np.sum(baseline_inferred_test == x_test_feature.reshape(1, -1)) / len(
            baseline_inferred_test
        )

        assert 0.6 <= baseline_train_acc
        assert 0.6 <= baseline_test_acc

    except ARTTestException as e:
        art_warning(e)


def test_errors(art_warning, tabular_dl_estimator_for_attack, get_iris_dataset):
    try:
        classifier = tabular_dl_estimator_for_attack(AttributeInferenceBlackBox)
        (x_train, y_train), (x_test, y_test) = get_iris_dataset

        with pytest.raises(ValueError):
            AttributeInferenceBlackBox(classifier, attack_feature="a")
        with pytest.raises(ValueError):
            AttributeInferenceBlackBox(classifier, attack_feature=-3)
        attack = AttributeInferenceBlackBox(classifier, attack_feature=8)
        with pytest.raises(ValueError):
            attack.fit(x_train)
        attack = AttributeInferenceBlackBox(classifier)
        with pytest.raises(ValueError):
            attack.fit(np.delete(x_train, 1, 1))
        with pytest.raises(ValueError):
            attack.infer(x_train, y_test)
        with pytest.raises(ValueError):
            attack.infer(x_train, y_train)
    except ARTTestException as e:
        art_warning(e)


def test_classifier_type_check_fail():
    backend_test_classifier_type_check_fail(
        AttributeInferenceBlackBox, (BaseEstimator, (ClassifierMixin, RegressorMixin))
    )


def test_check_params(art_warning, tabular_dl_estimator_for_attack):
    try:
        classifier = tabular_dl_estimator_for_attack(AttributeInferenceBlackBox)

        with pytest.raises(ValueError):
            AttributeInferenceBlackBox(classifier, attack_feature="a")

        with pytest.raises(ValueError):
            AttributeInferenceBlackBox(classifier, attack_feature=-1)

        with pytest.raises(ValueError):
            AttributeInferenceBlackBox(classifier, prediction_normal_factor=-1)

        with pytest.raises(ValueError):
            AttributeInferenceBlackBox(classifier, non_numerical_features=["a"])

        with pytest.raises(ValueError):
            AttributeInferenceBlackBox(classifier, encoder="a")

        with pytest.raises(ValueError):
            AttributeInferenceBlackBox(classifier, is_continuous="a")

    except ARTTestException as e:
        art_warning(e)
