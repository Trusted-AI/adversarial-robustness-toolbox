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
import unittest

import numpy as np

from art.defences.preprocessor import GaussianAugmentation

from tests.utils import master_seed

logger = logging.getLogger(__name__)


class TestGaussianAugmentation(unittest.TestCase):
    def setUp(self):
        # Set master seed
        master_seed(seed=1234)

    def test_small_size(self):
        x = np.arange(15).reshape((5, 3))
        ga = GaussianAugmentation(ratio=0.4, clip_values=(0, 15))
        x_new, _ = ga(x)
        self.assertEqual(x_new.shape, (7, 3))

    def test_double_size(self):
        x = np.arange(12).reshape((4, 3))
        x_original = x.copy()
        ga = GaussianAugmentation()
        x_new, _ = ga(x)
        self.assertEqual(x_new.shape[0], 2 * x.shape[0])
        # Check that x has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_original - x))), 0.0, delta=0.00001)

    def test_multiple_size(self):
        x = np.arange(12).reshape((4, 3))
        x_original = x.copy()
        ga = GaussianAugmentation(ratio=3.5)
        x_new, _ = ga(x)
        self.assertEqual(int(4.5 * x.shape[0]), x_new.shape[0])
        # Check that x has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_original - x))), 0.0, delta=0.00001)

    def test_labels(self):
        x = np.arange(12).reshape((4, 3))
        y = np.arange(8).reshape((4, 2))

        ga = GaussianAugmentation()
        x_new, new_y = ga(x, y)
        self.assertTrue(x_new.shape[0] == new_y.shape[0] == 8)
        self.assertEqual(x_new.shape[1:], x.shape[1:])
        self.assertEqual(new_y.shape[1:], y.shape[1:])

    def test_no_augmentation(self):
        x = np.arange(12).reshape((4, 3))
        ga = GaussianAugmentation(augmentation=False)
        x_new, _ = ga(x)
        self.assertEqual(x.shape, x_new.shape)
        self.assertFalse((x == x_new).all())

    def test_failure_augmentation_fit_predict(self):
        # Assert that value error is raised
        with self.assertRaises(ValueError) as context:
            _ = GaussianAugmentation(augmentation=True, apply_fit=False, apply_predict=True)

        self.assertTrue(
            "If `augmentation` is `True`, then `apply_fit` must be `True` and `apply_predict`"
            " must be `False`." in str(context.exception)
        )
        with self.assertRaises(ValueError) as context:
            _ = GaussianAugmentation(augmentation=True, apply_fit=False, apply_predict=False)

        self.assertIn(
            "If `augmentation` is `True`, then `apply_fit` and `apply_predict` can't be both `False`.",
            str(context.exception),
        )

    def test_check_params(self):
        with self.assertRaises(ValueError):
            _ = GaussianAugmentation(augmentation=True, ratio=-1)

        with self.assertRaises(ValueError):
            _ = GaussianAugmentation(clip_values=(0, 1, 2))

        with self.assertRaises(ValueError):
            _ = GaussianAugmentation(clip_values=(1, 0))


if __name__ == "__main__":
    unittest.main()
