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

from art.attacks.evasion.zoo import ZooAttack
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import random_targets

from tests.attacks.utils import backend_test_classifier_type_check_fail, assert_less_or_equal, assert_within_range
from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.fixture()
def fix_get_mnist_subset(get_mnist_dataset):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_mnist_dataset
    n_train = 1
    n_test = 1
    yield x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test]


def test_failure_attack(fix_get_mnist_subset, image_dl_estimator, art_warning):
    """
    Test the corner case when attack fails.
    :return:
    """
    try:
        (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset

        x_test_original = x_test_mnist.copy()

        estimator, _ = image_dl_estimator()

        # Failure attack
        zoo = ZooAttack(classifier=estimator, max_iter=0, binary_search_steps=0, learning_rate=0)
        x_test_mnist_adv = zoo.generate(x_test_mnist)

        assert_within_range(x_test_mnist_adv, 0.0, 1.0)

        np.testing.assert_almost_equal(x_test_mnist, x_test_mnist_adv, 3)

        # Check that x_test has not been modified by attack and classifier
        np.testing.assert_array_almost_equal(float(np.max(np.abs(x_test_original - x_test_mnist))), 0, decimal=5)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.parametrize("targeted", [True, False])
def test_mnist(fix_get_mnist_subset, image_dl_estimator, targeted, art_warning):
    '''
    NOTE: in the original legacy non pytest version of this test, the Keras and Pytorch tests were only running
    the untargeted part of the test (without any assertions for keras). The current code seems to work for both
    frameworks but, if not, this test should be skipped for keras or/and pytorch until reasons for this are found
    :param fix_get_mnist_subset:
    :param image_dl_estimator:
    :return:
    '''
    try:
        (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset
        x_test_original = x_test_mnist.copy()

        estimator, _ = image_dl_estimator()

        params = {}
        if targeted:
            params = {"y": random_targets(y_test_mnist, estimator.nb_classes)}
            # Target
            expectation = np.argmax(params["y"], axis=1)
        else:
            # y_pred
            expectation = np.argmax(estimator.predict(x_test_mnist), axis=1)

        zoo = ZooAttack(classifier=estimator, targeted=targeted, max_iter=30, binary_search_steps=8, batch_size=128)
        x_test_mnist_adv = zoo.generate(x_test_mnist, **params)
        assert bool((x_test_mnist == x_test_mnist_adv).all()) is False
        assert_within_range(x_test_mnist_adv, 0.0, 1.0)
        logger.debug("ZOO target: %s", expectation)
        y_pred_adv = np.argmax(estimator.predict(x_test_mnist_adv), axis=1)
        logger.info("ZOO success rate on MNIST: %.2f", (sum(expectation == y_pred_adv) / float(len(expectation))))
        logger.debug("ZOO actual: %s", y_pred_adv)

        # Check that x_test has not been modified by attack and classifier
        np.testing.assert_array_almost_equal(float(np.max(np.abs(x_test_original - x_test_mnist))), 0, decimal=5)
    except ARTTestException as e:
        art_warning(e)


def test_classifier_type_check_fail(art_warning):
    try:
        backend_test_classifier_type_check_fail(ZooAttack, [BaseEstimator, ClassifierMixin])
    except ARTTestException as e:
        art_warning(e)
