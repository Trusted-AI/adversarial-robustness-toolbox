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

import keras

from art.attacks.inference import (
    MembershipInferenceBlackBoxRuleBased,
    MembershipInferenceBlackBox,
)
from art.estimators.classification.pytorch import PyTorchClassifier
from art.estimators.classification.keras import KerasClassifier
from art.estimators.estimator import BaseEstimator

from tests.attacks.utils import backend_test_classifier_type_check_fail


logger = logging.getLogger(__name__)
attack_train_ratio = 0.5
num_classes_iris = 3
num_classes_mnist = 10


@pytest.fixture()
def fix_get_mnist_subset(get_mnist_dataset, get_iris_dataset):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_mnist_dataset
    n_train = 1000
    n_test = 200
    yield x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test]


def test_rule_based_image(fix_get_mnist_subset, get_image_classifier_list_for_attack):
    classifier_list = get_image_classifier_list_for_attack(MembershipInferenceBlackBoxRuleBased)
    if not classifier_list:
        logging.warning("Couldn't perform this test because no classifier is defined")
        return

    x_train, y_train, x_test, y_test = fix_get_mnist_subset

    for classifier in classifier_list:
        # print(type(classifier).__name__)
        attack = MembershipInferenceBlackBoxRuleBased(classifier)
        # infer attacked feature
        inferred_train = attack.infer(x_train, y_train)
        inferred_test = attack.infer(x_test, y_test)
        # check accuracy
        # print(inferred_train)
        # print(inferred_test)
        train_pos = sum(inferred_train) / len(inferred_train)
        test_pos = sum(inferred_test) / len(inferred_test)
        assert (train_pos > test_pos or
                train_pos == pytest.approx(test_pos, abs=0.03) or
                test_pos == 1)


def test_rule_based_tabular(get_iris_dataset, get_tabular_classifier_list):
    classifier_list = get_tabular_classifier_list(MembershipInferenceBlackBoxRuleBased)
    if not classifier_list:
        logging.warning("Couldn't perform this test because no classifier is defined")
        return

    (x_train, y_train), (x_test, y_test) = get_iris_dataset

    for classifier in classifier_list:
        # print(type(classifier).__name__)
        attack = MembershipInferenceBlackBoxRuleBased(classifier)
        # infer attacked feature
        inferred_train = attack.infer(x_train, y_train)
        inferred_test = attack.infer(x_test, y_test)
        # check accuracy
        train_pos = sum(inferred_train) / len(inferred_train)
        test_pos = sum(inferred_test) / len(inferred_test)
        assert (train_pos > test_pos or
                train_pos == pytest.approx(test_pos, abs=0.06) or
                test_pos == 1)


def test_black_box_image(fix_get_mnist_subset, get_image_classifier_list_for_attack):
    classifier_list = get_image_classifier_list_for_attack(MembershipInferenceBlackBox)
    if not classifier_list:
        logging.warning("Couldn't perform this test because no classifier is defined")
        return

    x_train, y_train, x_test, y_test = fix_get_mnist_subset
    attack_train_size = int(len(x_train) * attack_train_ratio)
    attack_test_size = int(len(x_test) * attack_train_ratio)

    for classifier in classifier_list:
        attack = MembershipInferenceBlackBox(classifier)
        # train attack model using only attack_train_ratio of data
        attack.fit(x_train[:attack_train_size], y_train[:attack_train_size],
                   x_test[:attack_test_size], y_test[:attack_test_size])
        # infer attacked feature on remainder of data
        inferred_train = attack.infer(x_train[attack_train_size:], y_train[attack_train_size:])
        inferred_test = attack.infer(x_test[attack_test_size:], y_test[attack_test_size:])
        # check accuracy
        train_pos = sum(inferred_train) / len(inferred_train)
        test_pos = sum(inferred_test) / len(inferred_test)
        assert (train_pos > test_pos or
                train_pos == pytest.approx(test_pos, abs=0.03) or
                test_pos == 1)


def test_black_box_tabular(get_tabular_classifier_list, get_iris_dataset):
    classifier_list = get_tabular_classifier_list(MembershipInferenceBlackBox)
    if not classifier_list:
        logging.warning("Couldn't perform this test because no classifier is defined")
        return

    (x_train, y_train), (x_test, y_test) = get_iris_dataset
    attack_train_size = int(len(x_train) * attack_train_ratio)
    attack_test_size = int(len(x_test) * attack_train_ratio)

    model_types = ['nn', 'rf', 'gb']

    for classifier in classifier_list:
        for t in model_types:
            attack = MembershipInferenceBlackBox(classifier, attack_model_type=t)
            # train attack model using only attack_train_ratio of data
            attack.fit(x_train[:attack_train_size], y_train[:attack_train_size],
                       x_test[:attack_test_size], y_test[:attack_test_size])
            # infer attacked feature on remainder of data
            inferred_train = attack.infer(x_train[attack_train_size:], y_train[attack_train_size:])
            inferred_test = attack.infer(x_test[attack_test_size:], y_test[attack_test_size:])
            # check accuracy
            train_pos = sum(inferred_train) / len(inferred_train)
            test_pos = sum(inferred_test) / len(inferred_test)
            assert (train_pos > test_pos or
                    train_pos == pytest.approx(test_pos, abs=0.08) or
                    test_pos == 1)

def test_black_box_loss_tabular(get_tabular_classifier_list, get_iris_dataset):
    classifier_list = get_tabular_classifier_list(MembershipInferenceBlackBox)
    if not classifier_list:
        logging.warning("Couldn't perform this test because no classifier is defined")
        return

    (x_train, y_train), (x_test, y_test) = get_iris_dataset
    attack_train_size = int(len(x_train) * attack_train_ratio)
    attack_test_size = int(len(x_test) * attack_train_ratio)

    model_types = ['nn', 'rf', 'gb']

    for classifier in classifier_list:
        if type(classifier).__name__ == "PyTorchClassifier" or \
           type(classifier).__name__ == "TensorFlowV2Classifier":
            for t in model_types:
                attack = MembershipInferenceBlackBox(classifier, input_type='loss', attack_model_type=t)
                # train attack model using only attack_train_ratio of data
                attack.fit(x_train[:attack_train_size], y_train[:attack_train_size],
                           x_test[:attack_test_size], y_test[:attack_test_size])
                # infer attacked feature on remainder of data
                inferred_train = attack.infer(x_train[attack_train_size:], y_train[attack_train_size:])
                inferred_test = attack.infer(x_test[attack_test_size:], y_test[attack_test_size:])
                # check accuracy
                train_pos = sum(inferred_train) / len(inferred_train)
                test_pos = sum(inferred_test) / len(inferred_test)
                assert (train_pos > test_pos or
                        train_pos == pytest.approx(test_pos, abs=0.15) or
                        test_pos == 1)


@pytest.mark.only_with_platform("keras")
def test_black_box_keras_loss(get_iris_dataset):
    (x_train, y_train), (x_test, y_test) = get_iris_dataset
    attack_train_size = int(len(x_train) * attack_train_ratio)
    attack_test_size = int(len(x_test) * attack_train_ratio)

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(8, input_dim=4, activation='relu'))
    model.add(keras.layers.Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=150, batch_size=10)

    classifier = KerasClassifier(model)
    attack = MembershipInferenceBlackBox(classifier, input_type='loss')

    with pytest.raises(NotImplementedError):
        attack.fit(x_train[:attack_train_size], y_train[:attack_train_size],
                   x_test[:attack_test_size], y_test[:attack_test_size])

    model2 = keras.models.Sequential()
    model2.add(keras.layers.Dense(12, input_dim=4, activation='relu'))
    model2.add(keras.layers.Dense(3, activation='softmax'))
    model2.compile(loss=keras.losses.CategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])
    model2.fit(x_train, y_train, epochs=150, batch_size=10)

    classifier = KerasClassifier(model2)
    attack = MembershipInferenceBlackBox(classifier, input_type='loss')

    # train attack model using only attack_train_ratio of data
    attack.fit(x_train[:attack_train_size], y_train[:attack_train_size],
               x_test[:attack_test_size], y_test[:attack_test_size])
    # infer attacked feature on remainder of data
    inferred_train = attack.infer(x_train[attack_train_size:], y_train[attack_train_size:])
    inferred_test = attack.infer(x_test[attack_test_size:], y_test[attack_test_size:])
    # check accuracy
    train_pos = sum(inferred_train) / len(inferred_train)
    test_pos = sum(inferred_test) / len(inferred_test)
    assert (train_pos > test_pos or
            train_pos == pytest.approx(test_pos, abs=0.15) or
            test_pos == 1)


def test_black_box_tabular_rf(get_tabular_classifier_list, get_iris_dataset):
    classifier_list = get_tabular_classifier_list(MembershipInferenceBlackBox)
    if not classifier_list:
        logging.warning("Couldn't perform this test because no classifier is defined")
        return

    (x_train, y_train), (x_test, y_test) = get_iris_dataset
    attack_train_size = int(len(x_train) * attack_train_ratio)
    attack_test_size = int(len(x_test) * attack_train_ratio)

    for classifier in classifier_list:
        attack = MembershipInferenceBlackBox(classifier, attack_model_type='rf')
        # train attack model using only attack_train_ratio of data
        attack.fit(x_train[:attack_train_size], y_train[:attack_train_size],
                   x_test[:attack_test_size], y_test[:attack_test_size])
        # infer attacked feature on remainder of data
        inferred_train = attack.infer(x_train[attack_train_size:], y_train[attack_train_size:])
        inferred_test = attack.infer(x_test[attack_test_size:], y_test[attack_test_size:])
        # check accuracy
        train_pos = sum(inferred_train) / len(inferred_train)
        test_pos = sum(inferred_test) / len(inferred_test)
        assert (train_pos > test_pos or
                train_pos == pytest.approx(test_pos, abs=0.08) or
                test_pos == 1)


def test_black_box_tabular_gb(get_tabular_classifier_list, get_iris_dataset):
    classifier_list = get_tabular_classifier_list(MembershipInferenceBlackBox)
    if not classifier_list:
        logging.warning("Couldn't perform this test because no classifier is defined")
        return

    (x_train, y_train), (x_test, y_test) = get_iris_dataset
    attack_train_size = int(len(x_train) * attack_train_ratio)
    attack_test_size = int(len(x_test) * attack_train_ratio)

    for classifier in classifier_list:
        attack = MembershipInferenceBlackBox(classifier, attack_model_type='gb')
        # train attack model using only attack_train_ratio of data
        attack.fit(x_train[:attack_train_size], y_train[:attack_train_size],
                   x_test[:attack_test_size], y_test[:attack_test_size])
        # infer attacked feature on remainder of data
        inferred_train = attack.infer(x_train[attack_train_size:], y_train[attack_train_size:])
        inferred_test = attack.infer(x_test[attack_test_size:], y_test[attack_test_size:])
        # check accuracy
        train_pos = sum(inferred_train) / len(inferred_train)
        test_pos = sum(inferred_test) / len(inferred_test)
        assert (train_pos > test_pos or
                train_pos == pytest.approx(test_pos, abs=0.03) or
                test_pos == 1)


class AttackModel(nn.Module):
    def __init__(self, num_features):
        super(AttackModel, self).__init__()
        self.layer = nn.Linear(num_features, 1)
        self.output = nn.Sigmoid()

    def forward(self, x):
        return self.output(self.layer(x))


# @pytest.mark.skipMlFramework("scikitlearn")
def test_black_box_with_model(get_tabular_classifier_list, get_iris_dataset):
    classifier_list = get_tabular_classifier_list(MembershipInferenceBlackBox)
    if not classifier_list:
        logging.warning("Couldn't perform this test because no classifier is defined")
        return

    (x_train, y_train), (x_test, y_test) = get_iris_dataset
    attack_train_size = int(len(x_train) * attack_train_ratio)
    attack_test_size = int(len(x_test) * attack_train_ratio)

    model = AttackModel(2*num_classes_iris)

    # Define a loss function and optimizer
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    attack_model = PyTorchClassifier(
        model=model, loss=loss_fn, optimizer=optimizer, input_shape=(2*num_classes_iris,), nb_classes=1
    )

    for classifier in classifier_list:
        print(type(classifier).__name__)
        attack = MembershipInferenceBlackBox(classifier, attack_model=attack_model)
        # train attack model using only attack_train_ratio of data
        attack.fit(x_train[:attack_train_size], y_train[:attack_train_size],
                   x_test[:attack_test_size], y_test[:attack_test_size])
        # infer attacked feature on remainder of data
        inferred_train = attack.infer(x_train[attack_train_size:], y_train[attack_train_size:])
        inferred_test = attack.infer(x_test[attack_test_size:], y_test[attack_test_size:])
        # check accuracy
        train_pos = sum(inferred_train) / len(inferred_train)
        test_pos = sum(inferred_test) / len(inferred_test)
        assert (train_pos > test_pos or
                train_pos == pytest.approx(test_pos, abs=0.03) or
                test_pos == 1)


def test_errors(get_tabular_classifier_list, get_iris_dataset):
    classifier_list = get_tabular_classifier_list(MembershipInferenceBlackBox)
    if not classifier_list:
        logging.warning("Couldn't perform this test because no classifier is defined")
        return
    (x_train, y_train), (x_test, y_test) = get_iris_dataset

    with pytest.raises(ValueError):
        MembershipInferenceBlackBox(classifier_list[0], attack_model_type='a')
    with pytest.raises(ValueError):
        MembershipInferenceBlackBox(classifier_list[0], input_type='a')
    attack = MembershipInferenceBlackBox(classifier_list[0])
    with pytest.raises(ValueError):
        attack.fit(x_train, y_test, x_test, y_test)
    with pytest.raises(ValueError):
        attack.fit(x_train, y_train, x_test, y_train)
    with pytest.raises(ValueError):
        attack.infer(x_train, y_test)


def test_classifier_type_check_fail():
    backend_test_classifier_type_check_fail(MembershipInferenceBlackBoxRuleBased, [BaseEstimator])
    backend_test_classifier_type_check_fail(MembershipInferenceBlackBox, [BaseEstimator])


if __name__ == "__main__":
    pytest.cmdline.main("-q -s {} --mlFramework=tensorflow --durations=0".format(__file__).split(" "))
