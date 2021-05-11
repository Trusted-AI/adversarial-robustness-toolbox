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
from sklearn.preprocessing import StandardScaler

from art.attacks.inference.attribute_inference.black_box import AttributeInferenceBlackBox
from art.estimators.classification.pytorch import PyTorchClassifier
from art.estimators.estimator import BaseEstimator
from art.estimators.classification import ClassifierMixin
from art.estimators.classification.scikitlearn import ScikitlearnDecisionTreeClassifier

from tests.attacks.utils import backend_test_classifier_type_check_fail
from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.mark.skip_framework("dl_frameworks")
def test_black_box(art_warning, decision_tree_estimator, get_iris_dataset):
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

        attack = AttributeInferenceBlackBox(classifier, attack_feature=attack_feature)
        # get original model's predictions
        x_train_predictions = np.array([np.argmax(arr) for arr in classifier.predict(x_train_iris)]).reshape(-1, 1)
        x_test_predictions = np.array([np.argmax(arr) for arr in classifier.predict(x_test_iris)]).reshape(-1, 1)
        # train attack model
        attack.fit(x_train)
        # infer attacked feature
        inferred_train = attack.infer(x_train_for_attack, x_train_predictions, values=values)
        inferred_test = attack.infer(x_test_for_attack, x_test_predictions, values=values)
        # check accuracy
        train_acc = np.sum(inferred_train == x_train_feature.reshape(1, -1)) / len(inferred_train)
        test_acc = np.sum(inferred_test == x_test_feature.reshape(1, -1)) / len(inferred_test)
        assert train_acc == pytest.approx(0.8285, abs=0.12)
        assert test_acc == pytest.approx(0.8888, abs=0.12)

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
        inferred_train = attack.infer(x_train_for_attack, x_train_predictions, values=values)
        inferred_test = attack.infer(x_test_for_attack, x_test_predictions, values=values)
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
def test_black_box_one_hot(art_warning, get_iris_dataset):
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

        attack = AttributeInferenceBlackBox(classifier, attack_feature=slice(attack_feature, attack_feature + 3))
        # get original model's predictions
        x_train_predictions = np.array([np.argmax(arr) for arr in classifier.predict(x_train)]).reshape(-1, 1)
        x_test_predictions = np.array([np.argmax(arr) for arr in classifier.predict(x_test)]).reshape(-1, 1)
        # train attack model
        attack.fit(x_train)
        # infer attacked feature
        inferred_train = attack.infer(x_train_for_attack, x_train_predictions)
        inferred_test = attack.infer(x_test_for_attack, x_test_predictions)
        # check accuracy
        train_acc = np.sum(np.all(inferred_train == train_one_hot, axis=1)) / len(inferred_train)
        test_acc = np.sum(np.all(inferred_test == test_one_hot, axis=1)) / len(inferred_test)
        assert pytest.approx(0.8666, abs=0.03) == train_acc
        assert pytest.approx(0.8888, abs=0.03) == test_acc

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

        attack = AttributeInferenceBlackBox(classifier, attack_feature=attack_feature)
        # get original model's predictions
        x_train_predictions = np.array([np.argmax(arr) for arr in classifier.predict(x_train)]).reshape(-1, 1)
        x_test_predictions = np.array([np.argmax(arr) for arr in classifier.predict(x_test)]).reshape(-1, 1)
        # train attack model
        attack.fit(x_train)
        # infer attacked feature
        values = [[-0.559017, 1.7888544], [-0.47003216, 2.127514], [-1.1774395, 0.84930056]]
        inferred_train = attack.infer(x_train_for_attack, x_train_predictions, values=values)
        inferred_test = attack.infer(x_test_for_attack, x_test_predictions, values=values)
        # check accuracy
        train_acc = np.sum(
            np.all(np.around(inferred_train, decimals=3) == np.around(train_one_hot, decimals=3), axis=1)
        ) / len(inferred_train)
        test_acc = np.sum(
            np.all(np.around(inferred_test, decimals=3) == np.around(test_one_hot, decimals=3), axis=1)
        ) / len(inferred_test)
        assert pytest.approx(0.8666, abs=0.05) == train_acc
        assert pytest.approx(0.8666, abs=0.05) == test_acc

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
    backend_test_classifier_type_check_fail(AttributeInferenceBlackBox, (BaseEstimator, ClassifierMixin))
