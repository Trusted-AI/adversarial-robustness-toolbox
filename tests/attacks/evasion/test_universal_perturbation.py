# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
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

from art.attacks.evasion.universal_perturbation import UniversalPerturbation

from tests.utils import ARTTestException
from art.estimators.classification.classifier import ClassifierMixin
from art.estimators.estimator import BaseEstimator
from tests.attacks.utils import backend_test_classifier_type_check_fail

logger = logging.getLogger(__name__)


@pytest.fixture()
def fix_get_mnist_subset(get_mnist_dataset):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_mnist_dataset
    n_train = 500
    n_test = 10
    yield x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test]


def test_image(art_warning, fix_get_mnist_subset, image_dl_estimator):
    try:
        (x_train, y_train, x_test, y_test) = fix_get_mnist_subset

        x_test_original = x_test.copy()

        # Build TensorFlowClassifier
        estimator, _ = image_dl_estimator()

        # Attack
        up = UniversalPerturbation(estimator, max_iter=1, attacker="newtonfool", attacker_params={"max_iter": 5})
        x_train_adv = up.generate(x_train)
        assert (up.fooling_rate >= 0.2) or not up.converged

        x_test_adv = x_test + up.noise
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, x_test, x_test_adv)

        train_y_pred = np.argmax(estimator.predict(x_train_adv), axis=1)
        test_y_pred = np.argmax(estimator.predict(x_test_adv), axis=1)
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, np.argmax(y_test, axis=1), test_y_pred)
        np.testing.assert_raises(
            AssertionError, np.testing.assert_array_equal, np.argmax(y_train, axis=1), train_y_pred
        )

        # Check that x_test has not been modified by attack and classifier
        np.testing.assert_array_almost_equal(float(np.max(np.abs(x_test_original - x_test))), 0, decimal=5)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.parametrize("clipped", [True, False])
def test_tabular(art_warning, get_iris_dataset, tabular_dl_estimator, clipped):
    try:
        (_, _), (x_test, y_test) = get_iris_dataset

        estimator = tabular_dl_estimator(clipped)

        # Test untargeted attack
        attack_params = {"max_iter": 1, "attacker": "ead", "attacker_params": {"max_iter": 5, "targeted": False}}
        attack = UniversalPerturbation(estimator)
        attack.set_params(**attack_params)
        x_test_iris_adv = attack.generate(x_test)
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, x_test, x_test_iris_adv)

        if clipped:
            np.testing.assert_array_less(x_test_iris_adv, 1)
            # Note: the np version of the assert doesn't seem to pass
            # np.testing.assert_array_less(0, x_test_iris_adv)
            assert bool((x_test_iris_adv >= 0).all())

        preds_adv = np.argmax(estimator.predict(x_test_iris_adv), axis=1)
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, np.argmax(y_test, axis=1), preds_adv)
        acc = np.sum(preds_adv == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info("Accuracy on Iris with universal adversarial examples: %.2f%%", (acc * 100))
    except ARTTestException as e:
        art_warning(e)


def test_classifier_type_check_fail():
    backend_test_classifier_type_check_fail(UniversalPerturbation, [BaseEstimator, ClassifierMixin])
