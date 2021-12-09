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

import keras

from art.attacks.inference.membership_inference.black_box import MembershipInferenceBlackBox
from art.estimators.classification.keras import KerasClassifier
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin
from art.estimators.regression import ScikitlearnRegressor, RegressorMixin

from tests.attacks.utils import backend_test_classifier_type_check_fail
from tests.utils import ARTTestException

logger = logging.getLogger(__name__)
attack_train_ratio = 0.5
num_classes_iris = 3
num_classes_mnist = 10


def test_black_box_image(art_warning, get_default_mnist_subset, image_dl_estimator_for_attack):
    try:
        classifier = image_dl_estimator_for_attack(MembershipInferenceBlackBox)
        attack = MembershipInferenceBlackBox(classifier)
        backend_check_membership_accuracy(attack, get_default_mnist_subset, attack_train_ratio, 0.25)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.parametrize("model_type", ["nn", "rf", "gb"])
def test_black_box_tabular(art_warning, model_type, tabular_dl_estimator_for_attack, get_iris_dataset):
    try:
        classifier = tabular_dl_estimator_for_attack(MembershipInferenceBlackBox)
        attack = MembershipInferenceBlackBox(classifier, attack_model_type=model_type)
        backend_check_membership_accuracy(attack, get_iris_dataset, attack_train_ratio, 0.25)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.parametrize("model_type", ["nn", "rf", "gb"])
def test_black_box_loss_tabular(art_warning, model_type, tabular_dl_estimator_for_attack, get_iris_dataset):
    try:
        classifier = tabular_dl_estimator_for_attack(MembershipInferenceBlackBox)
        if type(classifier).__name__ == "PyTorchClassifier" or type(classifier).__name__ == "TensorFlowV2Classifier":
            attack = MembershipInferenceBlackBox(classifier, input_type="loss", attack_model_type=model_type)
            backend_check_membership_accuracy(attack, get_iris_dataset, attack_train_ratio, 0.25)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.parametrize("model_type", ["nn", "rf", "gb"])
def test_black_box_loss_regression(art_warning, model_type, get_diabetes_dataset):
    try:
        from sklearn import linear_model

        (x_train_diabetes, y_train_diabetes), _ = get_diabetes_dataset
        regr_model = linear_model.LinearRegression()
        regr_model.fit(x_train_diabetes, y_train_diabetes)
        regressor = ScikitlearnRegressor(regr_model)

        attack = MembershipInferenceBlackBox(regressor, input_type="loss", attack_model_type=model_type)
        backend_check_membership_accuracy(attack, get_diabetes_dataset, attack_train_ratio, 0.25)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("tensorflow", "pytorch", "scikitlearn", "mxnet", "kerastf")
@pytest.mark.skipif(keras.__version__.startswith("2.2"), reason="requires Keras 2.3.0 or higher")
def test_black_box_keras_loss(art_warning, get_iris_dataset):
    try:
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
        backend_check_membership_accuracy(attack, get_iris_dataset, attack_train_ratio, 0.25)

        model2 = keras.models.Sequential()
        model2.add(keras.layers.Dense(12, input_dim=4, activation="relu"))
        model2.add(keras.layers.Dense(3, activation="softmax"))
        model2.compile(loss=keras.losses.CategoricalCrossentropy(), optimizer="adam", metrics=["accuracy"])
        model2.fit(x_train, y_train, epochs=150, batch_size=10)

        classifier = KerasClassifier(model2)
        attack = MembershipInferenceBlackBox(classifier, input_type="loss")
        backend_check_membership_accuracy(attack, get_iris_dataset, attack_train_ratio, 0.25)
    except ARTTestException as e:
        art_warning(e)


def test_black_box_tabular_rf(art_warning, tabular_dl_estimator_for_attack, get_iris_dataset):
    try:
        classifier = tabular_dl_estimator_for_attack(MembershipInferenceBlackBox)
        attack = MembershipInferenceBlackBox(classifier, attack_model_type="rf")
        backend_check_membership_accuracy(attack, get_iris_dataset, attack_train_ratio, 0.2)
    except ARTTestException as e:
        art_warning(e)


def test_black_box_tabular_gb(art_warning, tabular_dl_estimator_for_attack, get_iris_dataset):
    try:
        classifier = tabular_dl_estimator_for_attack(MembershipInferenceBlackBox)
        attack = MembershipInferenceBlackBox(classifier, attack_model_type="gb")
        # train attack model using only attack_train_ratio of data
        backend_check_membership_accuracy(attack, get_iris_dataset, attack_train_ratio, 0.25)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("tensorflow", "keras", "scikitlearn", "mxnet", "kerastf")
def test_black_box_with_model(art_warning, tabular_dl_estimator_for_attack, estimator_for_attack, get_iris_dataset):
    try:
        classifier = tabular_dl_estimator_for_attack(MembershipInferenceBlackBox)
        attack_model = estimator_for_attack(num_features=2 * num_classes_iris)
        attack = MembershipInferenceBlackBox(classifier, attack_model=attack_model)
        backend_check_membership_accuracy(attack, get_iris_dataset, attack_train_ratio, 0.25)
    except ARTTestException as e:
        art_warning(e)


def test_black_box_tabular_prob_rf(art_warning, tabular_dl_estimator_for_attack, get_iris_dataset):
    try:
        classifier = tabular_dl_estimator_for_attack(MembershipInferenceBlackBox)
        attack = MembershipInferenceBlackBox(classifier, attack_model_type="rf")
        backend_check_membership_probabilities(attack, get_iris_dataset, attack_train_ratio)
    except ARTTestException as e:
        art_warning(e)


def test_black_box_tabular_prob_nn(art_warning, tabular_dl_estimator_for_attack, get_iris_dataset):
    try:
        classifier = tabular_dl_estimator_for_attack(MembershipInferenceBlackBox)
        attack = MembershipInferenceBlackBox(classifier, attack_model_type="nn")
        backend_check_membership_probabilities(attack, get_iris_dataset, attack_train_ratio)
    except ARTTestException as e:
        art_warning(e)


def test_black_box_with_model_prob(
    art_warning, tabular_dl_estimator_for_attack, estimator_for_attack, get_iris_dataset
):
    try:
        classifier = tabular_dl_estimator_for_attack(MembershipInferenceBlackBox)
        attack_model = estimator_for_attack(num_features=2 * num_classes_iris)
        attack = MembershipInferenceBlackBox(classifier, attack_model=attack_model)
        backend_check_membership_probabilities(attack, get_iris_dataset, attack_train_ratio)
    except ARTTestException as e:
        art_warning(e)


def test_errors(art_warning, tabular_dl_estimator_for_attack, get_iris_dataset):
    try:
        classifier = tabular_dl_estimator_for_attack(MembershipInferenceBlackBox)
        (x_train, y_train), (x_test, y_test) = get_iris_dataset

        with pytest.raises(ValueError):
            MembershipInferenceBlackBox(classifier, attack_model_type="a")
        with pytest.raises(ValueError):
            MembershipInferenceBlackBox(classifier, input_type="a")
        attack = MembershipInferenceBlackBox(classifier)
        with pytest.raises(ValueError):
            attack.fit(x_train, y_test, x_test, y_test)
        with pytest.raises(ValueError):
            attack.fit(x_train, y_train, x_test, y_train)
        with pytest.raises(ValueError):
            attack.infer(x_train, y_test)
    except ARTTestException as e:
        art_warning(e)


def test_classifier_type_check_fail(art_warning):
    try:
        backend_test_classifier_type_check_fail(
            MembershipInferenceBlackBox, [BaseEstimator, (ClassifierMixin, RegressorMixin)]
        )
    except ARTTestException as e:
        art_warning(e)


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


def backend_check_membership_probabilities(attack, dataset, attack_train_ratio):
    (x_train, y_train), (x_test, y_test) = dataset
    attack_train_size = int(len(x_train) * attack_train_ratio)
    attack_test_size = int(len(x_test) * attack_train_ratio)

    # train attack model using only attack_train_ratio of data
    attack.fit(
        x_train[:attack_train_size], y_train[:attack_train_size], x_test[:attack_test_size], y_test[:attack_test_size]
    )

    # infer attacked feature on remainder of data
    inferred_train_pred = attack.infer(x_train[attack_train_size:], y_train[attack_train_size:])
    inferred_train_prob = attack.infer(x_train[attack_train_size:], y_train[attack_train_size:], probabilities=True)

    # check accuracy
    backend_check_probabilities(inferred_train_pred, inferred_train_prob)


def backend_check_probabilities(pred, prob):
    assert prob.shape[1] == 1
    assert np.all(np.round(prob) == pred.astype(int))
