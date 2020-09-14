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

import keras

from art.attacks.inference.membership_inference.black_box import MembershipInferenceBlackBox
from art.attacks.inference.membership_inference.black_box_rule_based import MembershipInferenceBlackBoxRuleBased
from art.estimators.classification.keras import KerasClassifier
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin

from tests.attacks.utils import backend_test_classifier_type_check_fail


logger = logging.getLogger(__name__)
attack_train_ratio = 0.5
num_classes_iris = 3
num_classes_mnist = 10


def test_rule_based_image(get_default_mnist_subset, image_dl_estimator_for_attack):
    classifier_list = image_dl_estimator_for_attack(MembershipInferenceBlackBoxRuleBased)
    if not classifier_list:
        logging.warning("Couldn't perform this test because no classifier is defined")
        return

    for classifier in classifier_list:
        attack = MembershipInferenceBlackBoxRuleBased(classifier)
        backend_check_membership_accuracy_no_fit(attack, get_default_mnist_subset, 0.8)


def test_rule_based_tabular(get_iris_dataset, get_tabular_classifier_list):
    classifier_list = get_tabular_classifier_list(MembershipInferenceBlackBoxRuleBased)
    if not classifier_list:
        logging.warning("Couldn't perform this test because no classifier is defined")
        return

    for classifier in classifier_list:
        attack = MembershipInferenceBlackBoxRuleBased(classifier)
        backend_check_membership_accuracy_no_fit(attack, get_iris_dataset, 0.06)


def test_black_box_image(get_default_mnist_subset, image_dl_estimator_for_attack):
    classifier_list = image_dl_estimator_for_attack(MembershipInferenceBlackBox)
    if not classifier_list:
        logging.warning("Couldn't perform this test because no classifier is defined")
        return

    for classifier in classifier_list:
        attack = MembershipInferenceBlackBox(classifier)
        backend_check_membership_accuracy(attack, get_default_mnist_subset, attack_train_ratio, 0.03)


@pytest.mark.parametrize("model_type", ["nn", "rf", "gb"])
def test_black_box_tabular(model_type, get_tabular_classifier_list, get_iris_dataset):
    classifier_list = get_tabular_classifier_list(MembershipInferenceBlackBox)
    if not classifier_list:
        logging.warning("Couldn't perform this test because no classifier is defined")
        return

    for classifier in classifier_list:
        attack = MembershipInferenceBlackBox(classifier, attack_model_type=model_type)
        backend_check_membership_accuracy(attack, get_iris_dataset, attack_train_ratio, 0.08)


@pytest.mark.parametrize("model_type", ["nn", "rf", "gb"])
def test_black_box_loss_tabular(model_type, get_tabular_classifier_list, get_iris_dataset):
    classifier_list = get_tabular_classifier_list(MembershipInferenceBlackBox)
    if not classifier_list:
        logging.warning("Couldn't perform this test because no classifier is defined")
        return

    for classifier in classifier_list:
        if type(classifier).__name__ == "PyTorchClassifier" or type(classifier).__name__ == "TensorFlowV2Classifier":
            attack = MembershipInferenceBlackBox(classifier, input_type="loss", attack_model_type=model_type)
            backend_check_membership_accuracy(attack, get_iris_dataset, attack_train_ratio, 0.15)


@pytest.mark.only_with_platform("keras")
@pytest.mark.skipif(keras.__version__.startswith("2.2"), reason="requires Keras 2.3.0 or higher")
def test_black_box_keras_loss(get_iris_dataset):
    (x_train, y_train), (_, _) = get_iris_dataset

    # This test creates a framework-specific (keras) model because it needs to check both the case of a string-based
    # loss and a class-based loss, and therefore cannot use the generic fixture get_tabular_classifier_list
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(8, input_dim=4, activation="relu"))
    model.add(keras.layers.Dense(3, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=150, batch_size=10)

    classifier = KerasClassifier(model)
    attack = MembershipInferenceBlackBox(classifier, input_type="loss")
    backend_check_membership_accuracy(attack, get_iris_dataset, attack_train_ratio, 0.15)

    model2 = keras.models.Sequential()
    model2.add(keras.layers.Dense(12, input_dim=4, activation="relu"))
    model2.add(keras.layers.Dense(3, activation="softmax"))
    model2.compile(loss=keras.losses.CategoricalCrossentropy(), optimizer="adam", metrics=["accuracy"])
    model2.fit(x_train, y_train, epochs=150, batch_size=10)

    classifier = KerasClassifier(model2)
    attack = MembershipInferenceBlackBox(classifier, input_type="loss")
    backend_check_membership_accuracy(attack, get_iris_dataset, attack_train_ratio, 0.15)


def test_black_box_tabular_rf(get_tabular_classifier_list, get_iris_dataset):
    classifier_list = get_tabular_classifier_list(MembershipInferenceBlackBox)
    if not classifier_list:
        logging.warning("Couldn't perform this test because no classifier is defined")
        return

    for classifier in classifier_list:
        attack = MembershipInferenceBlackBox(classifier, attack_model_type="rf")
        backend_check_membership_accuracy(attack, get_iris_dataset, attack_train_ratio, 0.1)


def test_black_box_tabular_gb(get_tabular_classifier_list, get_iris_dataset):
    classifier_list = get_tabular_classifier_list(MembershipInferenceBlackBox)
    if not classifier_list:
        logging.warning("Couldn't perform this test because no classifier is defined")
        return

    for classifier in classifier_list:
        attack = MembershipInferenceBlackBox(classifier, attack_model_type="gb")
        # train attack model using only attack_train_ratio of data
        backend_check_membership_accuracy(attack, get_iris_dataset, attack_train_ratio, 0.03)


@pytest.mark.only_with_platform("pytorch")
def test_black_box_with_model(get_tabular_classifier_list, get_attack_classifier_list, get_iris_dataset):
    classifier_list = get_tabular_classifier_list(MembershipInferenceBlackBox)
    if not classifier_list:
        logging.warning("Couldn't perform this test because no classifier is defined")
        return

    attack_model_list = get_attack_classifier_list(num_features=2 * num_classes_iris)
    if not attack_model_list:
        logging.warning("Couldn't perform this test because no attack model is defined")
        return

    for classifier in classifier_list:
        for attack_model in attack_model_list:
            print(type(attack_model).__name__)
            attack = MembershipInferenceBlackBox(classifier, attack_model=attack_model)
            backend_check_membership_accuracy(attack, get_iris_dataset, attack_train_ratio, 0.03)


def test_errors(get_tabular_classifier_list, get_iris_dataset):
    classifier_list = get_tabular_classifier_list(MembershipInferenceBlackBox)
    if not classifier_list:
        logging.warning("Couldn't perform this test because no classifier is defined")
        return
    (x_train, y_train), (x_test, y_test) = get_iris_dataset

    with pytest.raises(ValueError):
        MembershipInferenceBlackBox(classifier_list[0], attack_model_type="a")
    with pytest.raises(ValueError):
        MembershipInferenceBlackBox(classifier_list[0], input_type="a")
    attack = MembershipInferenceBlackBox(classifier_list[0])
    with pytest.raises(ValueError):
        attack.fit(x_train, y_test, x_test, y_test)
    with pytest.raises(ValueError):
        attack.fit(x_train, y_train, x_test, y_train)
    with pytest.raises(ValueError):
        attack.infer(x_train, y_test)


def test_classifier_type_check_fail():
    backend_test_classifier_type_check_fail(MembershipInferenceBlackBoxRuleBased, [BaseEstimator, ClassifierMixin])
    backend_test_classifier_type_check_fail(MembershipInferenceBlackBox, [BaseEstimator, ClassifierMixin])


def backend_check_membership_accuracy_no_fit(attack, dataset, approx):
    (x_train, y_train), (x_test, y_test) = dataset
    # infer attacked feature
    inferred_train = attack.infer(x_train, y_train)
    inferred_test = attack.infer(x_test, y_test)
    # check accuracy
    backend_check_accuracy(inferred_train, inferred_test, approx)


def backend_check_membership_accuracy(attack, dataset, attack_train_ratio, approx):
    (x_train, y_train), (x_test, y_test) = dataset
    attack_train_size = int(len(x_train) * attack_train_ratio)
    attack_test_size = int(len(x_test) * attack_train_ratio)

    # train attack model using only attack_train_ratio of data
    attack.fit(
        x_train[:attack_train_size], y_train[:attack_train_size], x_test[:attack_test_size], y_test[:attack_test_size]
    )

    # infer attacked feature on remainder of data
    inferred_train = attack.infer(x_train[attack_train_size:], y_train[attack_train_size:])
    inferred_test = attack.infer(x_test[attack_test_size:], y_test[attack_test_size:])

    # check accuracy
    backend_check_accuracy(inferred_train, inferred_test, approx)


def backend_check_accuracy(inferred_train, inferred_test, approx):
    train_pos = sum(inferred_train) / len(inferred_train)
    test_pos = sum(inferred_test) / len(inferred_test)
    assert train_pos > test_pos or train_pos == pytest.approx(test_pos, abs=approx) or test_pos == 1


if __name__ == "__main__":
    pytest.cmdline.main("-q -s {} --mlFramework=tensorflow --durations=0".format(__file__).split(" "))
