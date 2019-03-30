from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

import numpy as np

from art.defences.gaussian_augmentation import GaussianAugmentation
from art.utils import master_seed

logger = logging.getLogger('testLogger')


class TestGaussianAugmentation(unittest.TestCase):
    def setUp(self):
        # Set master seed
        master_seed(1234)

    def test_small_size(self):
        x = np.arange(15).reshape((5, 3))
        ga = GaussianAugmentation()
        new_x, _ = ga(x, ratio=.4)
        self.assertTrue(new_x.shape == (7, 3))

    def test_double_size(self):
        x = np.arange(12).reshape((4, 3))
        ga = GaussianAugmentation()
        new_x, _ = ga(x)
        self.assertTrue(new_x.shape[0] == 2 * x.shape[0])

    def test_multiple_size(self):
        x = np.arange(12).reshape((4, 3))
        ga = GaussianAugmentation(ratio=3.5)
        new_x, _ = ga(x)
        self.assertTrue(int(4.5 * x.shape[0]) == new_x.shape[0])

    def test_labels(self):
        x = np.arange(12).reshape((4, 3))
        y = np.arange(8).reshape((4, 2))

        ga = GaussianAugmentation()
        new_x, new_y = ga(x, y)
        self.assertTrue(new_x.shape[0] == new_y.shape[0] == 8)
        self.assertTrue(new_x.shape[1:] == x.shape[1:])
        self.assertTrue(new_y.shape[1:] == y.shape[1:])

    def test_no_augmentation(self):
        x = np.arange(12).reshape((4, 3))
        ga = GaussianAugmentation(augmentation=False)
        new_x, _ = ga(x)
        self.assertTrue(x.shape == new_x.shape)
        self.assertFalse((x == new_x).all())


if __name__ == '__main__':
    unittest.main()
