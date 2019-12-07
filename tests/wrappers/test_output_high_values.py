# MIT License
#
# Copyright (C) IBM Corporation 2019
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

import numpy as np

from art.utils import load_dataset, master_seed
from art.utils_test import get_classifier_kr
from art.wrappers.output_high_values import OutputHighValues

logger = logging.getLogger(__name__)


class TestOutputHighValues(unittest.TestCase):
    """
    A unittest class for testing the High Values Output wrapper.
    """

    @classmethod
    def setUpClass(cls):
        (x_train, y_train), (x_test, y_test), _, _ = load_dataset('mnist')
        cls.mnist = (x_train, y_train), (x_test, y_test)
        cls.classifier = get_classifier_kr()

    def setUp(self):
        master_seed(1234)

    def test_decimals_0_1(self):
        """
        Test with cutoff of 0.1.
        """
        (_, _), (x_test, _) = self.mnist
        wrapper = OutputHighValues(classifier=self.classifier, cutoff=0.1)
        expected_predictions = np.asarray([[0.12109935, 0.0, 0.0, 0.0, 0.11366928, 0.0, 0.0, 0.30685693, 0.0, 0.0]],
                                          dtype=np.float32)
        np.testing.assert_array_almost_equal(wrapper.predict(x_test[0:1]), expected_predictions, decimal=4)

    def test_decimals_0_2(self):
        """
        Test with cutoff of 0.2.
        """
        (_, _), (x_test, _) = self.mnist
        wrapper = OutputHighValues(classifier=self.classifier, cutoff=0.2)
        expected_predictions = np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.30685693, 0.0, 0.0]],
                                          dtype=np.float32)
        np.testing.assert_array_almost_equal(wrapper.predict(x_test[0:1]), expected_predictions, decimal=4)


if __name__ == '__main__':
    unittest.main()
