# MIT License
#
# Copyright (C) IBM Corporation 2018
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
import unittest
import sys
import numpy as np
import os
from art.attacks import FastGradientMethod
from art.utils import get_labels_np_array, random_targets
from tests.utils_test import TestBase
from tests import utils_test
import pytest

logger = logging.getLogger(__name__)
# or pytest -q tests/attacks/evasion/test_pytest_fast_gradient.py --mlFramework=pytorch --durations=0

@pytest.fixture()
def fix_get_mnist_subset(fix_get_mnist):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = fix_get_mnist
    n_train = 100
    n_test = 11
    yield (x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test])

def test_minimal_perturbations_images(fix_get_mnist_subset, image_classifier_list):
    (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset

    # TODO this if statement must be removed once we have a classifier for both image and tabular data
    if image_classifier_list is None:
        logging.warning("Couldn't perform  this test because no classifier is defined")
        return

    for classifier in image_classifier_list:
        attack = FastGradientMethod(classifier, eps=1.0, batch_size=11)
        attack_params = {"minimal": True, "eps_step": 0.1, "eps": 5.0}
        attack.set_params(**attack_params)

        x_test_adv_min = attack.generate(x_test_mnist)

        np.testing.assert_array_almost_equal(float(np.mean(x_test_adv_min - x_test_mnist)), 0.03896513, decimal=0.01)
        np.testing.assert_array_almost_equal(float(np.min(x_test_adv_min - x_test_mnist)), -0.30000000, decimal=0.00001)
        np.testing.assert_array_almost_equal(float(np.max(x_test_adv_min - x_test_mnist)), 0.30000000, decimal=0.00001)

        y_test_pred = classifier.predict(x_test_adv_min)

        y_test_pred_expected = np.asarray([4, 2, 4, 7, 0, 4, 7, 2, 0, 7, 0])

        np.testing.assert_array_equal(np.argmax(y_test_pred, axis=1), y_test_pred_expected)


if __name__ == '__main__':
    unittest.main()
