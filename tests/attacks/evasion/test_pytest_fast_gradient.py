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


@pytest.fixture(scope="function")
def fix_get_mnist_subset(fix_get_mnist_dataset):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = fix_get_mnist_dataset
    n_train = 100
    n_test = 11
    yield (x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test])
    print("tmp")

def test_minimal_perturbations_images(fix_get_mnist_subset):
    print("tmp1")
    tmp = fix_get_mnist_subset
    print("tmp2")

def test_minimal_perturbations_images_2(fix_get_mnist_subset):
    print("tmp1")
    tmp = fix_get_mnist_subset
    print("tmp2")

# class TestFastGradientMethodImages(TestBase):
#
#     @classmethod
#     def setUpClass(cls):
#         super().setUpClass()
#         cls.n_train = 100
#         cls.n_test = 11
#         cls.create_image_dataset(n_train=cls.n_train, n_test=cls.n_test)
#
#     def setUp(self):
#         super().setUp()
#         self.x_test_original = self.x_test_mnist.copy()
#         self.x_test_potentially_modified = self.x_test_mnist
#
#     def tearDown(self):
#         super().tearDown()
#
#         # Check that x_test has not been modified by attack and classifier
#         self.assertAlmostEqual(float(np.max(np.abs(self.x_test_original - self.x_test_potentially_modified))), 0.0, delta=0.00001)
#
#     def test_minimal_perturbations_images(self):
#
#         classifier_list = utils_test.get_image_classifiers()
#
#         # TODO this if statement must be removed once we have a classifier for both image and tabular data
#         if classifier_list is None:
#             logging.warning("Couldn't perform  this test because no classifier is defined")
#             return
#
#         for classifier in classifier_list:
#
#
#             attack = FastGradientMethod(classifier, eps=1.0, batch_size=11)
#             attack_params = {"minimal": True, "eps_step": 0.1, "eps": 5.0}
#             attack.set_params(**attack_params)
#
#             x_test_adv_min = attack.generate(self.x_test_mnist)
#
#             self.assertAlmostEqual(float(np.mean(x_test_adv_min - self.x_test_mnist)), 0.03896513, delta=0.01)
#             self.assertAlmostEqual(float(np.min(x_test_adv_min - self.x_test_mnist)), -0.30000000, delta=0.00001)
#             self.assertAlmostEqual(float(np.max(x_test_adv_min - self.x_test_mnist)), 0.30000000, delta=0.00001)
#
#             y_test_pred = classifier.predict(x_test_adv_min)
#
#             y_test_pred_expected = np.asarray([4, 2, 4, 7, 0, 4, 7, 2, 0, 7, 0])
#
#             np.testing.assert_array_equal(np.argmax(y_test_pred, axis=1), y_test_pred_expected)



if __name__ == '__main__':
    unittest.main()
