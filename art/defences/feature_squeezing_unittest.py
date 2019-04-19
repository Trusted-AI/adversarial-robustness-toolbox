from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

import numpy as np

from art.defences.feature_squeezing import FeatureSqueezing
from art.utils import master_seed

logger = logging.getLogger('testLogger')


class TestFeatureSqueezing(unittest.TestCase):
    def setUp(self):
        # Set master seed
        master_seed(1234)

    def test_ones(self):
        m, n = 10, 2
        x = np.ones((m, n))

        for depth in range(1, 50):
            preproc = FeatureSqueezing()
            squeezed_x, _ = preproc(x, bit_depth=depth)
            self.assertTrue((squeezed_x == 1).all())

    def test_random(self):
        m, n = 1000, 20
        x = np.random.rand(m, n)
        x_zero = np.where(x < 0.5)
        x_one = np.where(x >= 0.5)

        preproc = FeatureSqueezing()
        squeezed_x, _ = preproc(x, bit_depth=1)
        self.assertTrue((squeezed_x[x_zero] == 0.).all())
        self.assertTrue((squeezed_x[x_one] == 1.).all())

        squeezed_x, _ = preproc(x, bit_depth=2)
        self.assertFalse(np.logical_and(0. < squeezed_x, squeezed_x < 0.33).any())
        self.assertFalse(np.logical_and(0.34 < squeezed_x, squeezed_x < 0.66).any())
        self.assertFalse(np.logical_and(0.67 < squeezed_x, squeezed_x < 1.).any())

    def test_data_range(self):
        x = np.arange(5)
        preproc = FeatureSqueezing()
        squeezed_x, _ = preproc(x, bit_depth=2, clip_values=(0, 4))
        self.assertTrue(np.array_equal(x, np.arange(5)))
        self.assertTrue(np.allclose(squeezed_x, [0, 1.33, 2.67, 2.67, 4], atol=1e-1))


if __name__ == '__main__':
    unittest.main()
