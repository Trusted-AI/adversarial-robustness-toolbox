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

from art.attacks.inference.membership_inference.label_only_boundary_distance import LabelOnlyDecisionBoundary
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin

from tests.attacks.utils import backend_test_classifier_type_check_fail
from tests.utils import ARTTestException

logger = logging.getLogger(__name__)

attack_train_ratio = 0.5


def test_label_only_boundary_distance_image(art_warning, get_default_mnist_subset, image_dl_estimator_for_attack):
    try:
        classifier = image_dl_estimator_for_attack(LabelOnlyDecisionBoundary)
        attack = LabelOnlyDecisionBoundary(classifier, distance_threshold_tau=0.5)
        backend_check_membership_accuracy(attack, get_default_mnist_subset, attack_train_ratio, 0.05)
    except ARTTestException as e:
        art_warning(e)


def test_label_only_boundary_distance_prob(art_warning, get_default_mnist_subset, image_dl_estimator_for_attack):
    try:
        classifier = image_dl_estimator_for_attack(LabelOnlyDecisionBoundary)
        attack = LabelOnlyDecisionBoundary(classifier, distance_threshold_tau=0.5)
        backend_check_membership_probabilities(attack, get_default_mnist_subset, attack_train_ratio)
    except ARTTestException as e:
        art_warning(e)


def test_label_only_boundary_distance_prob_calib(art_warning, get_default_mnist_subset, image_dl_estimator_for_attack):
    try:
        classifier = image_dl_estimator_for_attack(LabelOnlyDecisionBoundary)
        attack = LabelOnlyDecisionBoundary(classifier)
        (x_train, y_train), (x_test, y_test) = get_default_mnist_subset
        kwargs = {
            "norm": 2,
            "max_iter": 2,
            "max_eval": 4,
            "init_eval": 1,
            "init_size": 1,
            "verbose": False,
        }
        attack.calibrate_distance_threshold(x_train, y_train, x_test, y_test, **kwargs)
        backend_check_membership_probabilities(attack, get_default_mnist_subset, attack_train_ratio)
    except ARTTestException as e:
        art_warning(e)


def test_label_only_boundary_distance_prob_calib_unsup(
    art_warning, get_default_mnist_subset, image_dl_estimator_for_attack
):
    try:
        classifier = image_dl_estimator_for_attack(LabelOnlyDecisionBoundary)
        attack = LabelOnlyDecisionBoundary(classifier)
        kwargs = {
            "norm": 2,
            "max_iter": 2,
            "max_eval": 4,
            "init_eval": 1,
            "init_size": 1,
            "verbose": False,
        }
        attack.calibrate_distance_threshold_unsupervised(50, 100, 1, **kwargs)
        backend_check_membership_probabilities(attack, get_default_mnist_subset, attack_train_ratio)
    except ARTTestException as e:
        art_warning(e)


def test_classifier_type_check_fail(art_warning):
    try:
        backend_test_classifier_type_check_fail(LabelOnlyDecisionBoundary, [BaseEstimator, ClassifierMixin])
    except ARTTestException as e:
        art_warning(e)


def backend_check_membership_accuracy(attack, dataset, attack_train_ratio, approx):
    (x_train, y_train), (x_test, y_test) = dataset
    attack_train_size = int(len(x_train) * attack_train_ratio)
    attack_test_size = int(len(x_test) * attack_train_ratio)

    # infer attacked feature on remainder of data
    kwargs = {
        "norm": 2,
        "max_iter": 2,
        "max_eval": 4,
        "init_eval": 1,
        "init_size": 1,
        "verbose": False,
    }
    inferred_train = attack.infer(x_train[attack_train_size:], y_train[attack_train_size:], **kwargs)
    inferred_test = attack.infer(x_test[attack_test_size:], y_test[attack_test_size:], **kwargs)

    # check accuracy
    backend_check_accuracy(inferred_train, inferred_test, approx)


def backend_check_accuracy(inferred_train, inferred_test, approx):
    train_pos = sum(inferred_train) / len(inferred_train)
    test_pos = sum(inferred_test) / len(inferred_test)
    assert train_pos > test_pos or train_pos == pytest.approx(test_pos, abs=approx) or test_pos == 1


def backend_check_membership_probabilities(attack, dataset, attack_train_ratio):
    (x_train, y_train), _ = dataset
    attack_train_size = int(len(x_train) * attack_train_ratio)

    kwargs = {
        "norm": 2,
        "max_iter": 2,
        "max_eval": 4,
        "init_eval": 1,
        "init_size": 1,
        "verbose": False,
    }
    # infer attacked feature on remainder of data
    inferred_train_prob = attack.infer(
        x_train[attack_train_size:], y_train[attack_train_size:], probabilities=True, **kwargs
    )

    # check accuracy
    backend_check_probabilities(inferred_train_prob)


def backend_check_probabilities(prob):
    assert prob.shape[1] == 2
    assert np.all(np.sum(prob, axis=1) == 1)


def test_check_params(art_warning, image_dl_estimator_for_attack):
    try:
        classifier = image_dl_estimator_for_attack(LabelOnlyDecisionBoundary)

        with pytest.raises(ValueError):
            _ = LabelOnlyDecisionBoundary(classifier, distance_threshold_tau=-0.5)

    except ARTTestException as e:
        art_warning(e)
