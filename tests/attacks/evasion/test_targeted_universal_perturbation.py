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

import numpy as np
import pytest

from art.attacks.evasion.targeted_universal_perturbation import TargetedUniversalPerturbation
from art.estimators.classification.classifier import ClassifierMixin
from art.estimators.estimator import BaseEstimator
from tests.attacks.utils import backend_test_classifier_type_check_fail


logger = logging.getLogger(__name__)


@pytest.fixture()
def fix_get_mnist_subset(get_mnist_dataset):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_mnist_dataset
    n_train = 150
    n_test = 10
    yield x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test]


@pytest.mark.skipMlFramework("mxnet", "scikitlearn", "tensorflow2v1")
def test_mnist(fix_get_mnist_subset, image_dl_estimator):
    (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset

    x_test_original = x_test_mnist.copy()

    # Build TensorFlowClassifier
    estimator, _ = image_dl_estimator(from_logits=False)

    # set target label
    target = 0
    y_target = np.zeros([len(x_train_mnist), 10])
    for i in range(len(x_train_mnist)):
        y_target[i, target] = 1.0

    # Attack
    up = TargetedUniversalPerturbation(
        estimator, max_iter=1, attacker="fgsm", attacker_params={"eps": 0.3, "targeted": True}
    )
    x_train_adv = up.generate(x_train_mnist, y=y_target)
    assert (up.fooling_rate >= 0.2) or not up.converged

    x_test_adv = x_test_mnist + up.noise
    np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, x_test_mnist, x_test_adv)

    train_y_pred = np.argmax(estimator.predict(x_train_adv), axis=1)
    test_y_pred = np.argmax(estimator.predict(x_test_adv), axis=1)
    assert bool((np.argmax(y_test_mnist, axis=1) == test_y_pred).all()) is False
    assert bool((np.argmax(y_train_mnist, axis=1) == train_y_pred).all()) is False

    # Check that x_test has not been modified by attack and classifier
    np.testing.assert_array_almost_equal(float(np.max(np.abs(x_test_original - x_test_mnist))), 0, decimal=5)


def test_classifier_type_check_fail():
    backend_test_classifier_type_check_fail(TargetedUniversalPerturbation, (BaseEstimator, ClassifierMixin))
