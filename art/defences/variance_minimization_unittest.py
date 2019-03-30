from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

import numpy as np

from art.defences.variance_minimization import TotalVarMin
from art.utils import master_seed

logger = logging.getLogger('testLogger')


class TestTotalVarMin(unittest.TestCase):
    def setUp(self):
        # Set master seed
        master_seed(1234)

    def test_one_channel(self):
        x = np.random.rand(2, 28, 28, 1)
        preprocess = TotalVarMin()
        preprocessed_x, _ = preprocess(x)
        self.assertTrue((preprocessed_x.shape == x.shape))
        self.assertTrue((preprocessed_x <= 1.0).all())
        self.assertTrue((preprocessed_x >= 0.0).all())
        self.assertFalse((preprocessed_x == x).all())

    def test_three_channels(self):
        x = np.random.rand(2, 32, 32, 3)
        preprocess = TotalVarMin()
        preprocessed_x, _ = preprocess(x)
        self.assertTrue((preprocessed_x.shape == x.shape))
        self.assertTrue((preprocessed_x <= 1.0).all())
        self.assertTrue((preprocessed_x >= 0.0).all())
        self.assertFalse((preprocessed_x == x).all())


if __name__ == '__main__':
    unittest.main()
