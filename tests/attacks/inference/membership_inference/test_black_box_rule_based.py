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

from art.attacks.inference.membership_inference.black_box_rule_based import MembershipInferenceBlackBoxRuleBased
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin

from tests.attacks.utils import backend_test_classifier_type_check_fail
from tests.utils import ARTTestException

logger = logging.getLogger(__name__)
attack_train_ratio = 0.5
num_classes_iris = 3
num_classes_mnist = 10


def test_rule_based_image(art_warning, get_default_mnist_subset, image_dl_estimator_for_attack):
    try:
        classifier = image_dl_estimator_for_attack(MembershipInferenceBlackBoxRuleBased)
        attack = MembershipInferenceBlackBoxRuleBased(classifier)
        backend_check_membership_accuracy_no_fit(attack, get_default_mnist_subset, 0.8)
    except ARTTestException as e:
        art_warning(e)


def test_rule_based_tabular(art_warning, get_iris_dataset, tabular_dl_estimator_for_attack):
    try:
        classifier = tabular_dl_estimator_for_attack(MembershipInferenceBlackBoxRuleBased)
        attack = MembershipInferenceBlackBoxRuleBased(classifier)
        backend_check_membership_accuracy_no_fit(attack, get_iris_dataset, 0.06)
    except ARTTestException as e:
        art_warning(e)


def test_rule_based_tabular_prob(art_warning, get_iris_dataset, tabular_dl_estimator_for_attack):
    try:
        classifier = tabular_dl_estimator_for_attack(MembershipInferenceBlackBoxRuleBased)
        attack = MembershipInferenceBlackBoxRuleBased(classifier)
        backend_check_membership_probabilities(attack, get_iris_dataset)
    except ARTTestException as e:
        art_warning(e)


def test_classifier_type_check_fail(art_warning):
    try:
        backend_test_classifier_type_check_fail(MembershipInferenceBlackBoxRuleBased, [BaseEstimator, ClassifierMixin])
    except ARTTestException as e:
        art_warning(e)


def backend_check_membership_accuracy_no_fit(attack, dataset, approx):
    (x_train, y_train), (x_test, y_test) = dataset
    # infer attacked feature
    inferred_train = attack.infer(x_train, y_train)
    inferred_test = attack.infer(x_test, y_test)
    # check accuracy
    backend_check_accuracy(inferred_train, inferred_test, approx)


def backend_check_accuracy(inferred_train, inferred_test, approx):
    train_pos = sum(inferred_train) / len(inferred_train)
    test_pos = sum(inferred_test) / len(inferred_test)
    assert train_pos > test_pos or train_pos == pytest.approx(test_pos, abs=approx) or test_pos == 1


def backend_check_membership_probabilities(attack, dataset):
    (x_train, y_train), _ = dataset

    # infer attacked feature on remainder of data
    inferred_train_pred = attack.infer(x_train, y_train)
    inferred_train_prob = attack.infer(x_train, y_train, probabilities=True)

    # check accuracy
    backend_check_probabilities(inferred_train_pred, inferred_train_prob)


def backend_check_probabilities(pred, prob):
    assert prob.shape[1] == 2
    assert np.all(np.sum(prob, axis=1) == 1)
    assert np.all(np.argmax(prob, axis=1) == pred.astype(int))
