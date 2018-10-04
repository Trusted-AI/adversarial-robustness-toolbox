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

import numpy as np

from art.defences.feature_squeezing import FeatureSqueezing

logger = logging.getLogger('testLogger')


class TestFeatureSqueezing(unittest.TestCase):
    def test_ones(self):
        m, n = 10, 2
        x = np.ones((m, n))

        for depth in range(1, 50):
            preproc = FeatureSqueezing()
            squeezed_x = preproc(x, bit_depth=depth)
            self.assertTrue((squeezed_x == 1).all())

    def test_random(self):
        m, n = 1000, 20
        x = np.random.rand(m, n)
        x_zero = np.where(x < 0.5)
        x_one = np.where(x >= 0.5)

        preproc = FeatureSqueezing()
        squeezed_x = preproc(x, bit_depth=1)
        self.assertTrue((squeezed_x[x_zero] == 0.).all())
        self.assertTrue((squeezed_x[x_one] == 1.).all())

        squeezed_x = preproc(x, bit_depth=2)
        self.assertFalse(np.logical_and(0. < squeezed_x, squeezed_x < 0.33).any())
        self.assertFalse(np.logical_and(0.34 < squeezed_x, squeezed_x < 0.66).any())
        self.assertFalse(np.logical_and(0.67 < squeezed_x, squeezed_x < 1.).any())

    def test_data_range(self):
        x = np.arange(5)
        preproc = FeatureSqueezing()
        squeezed_x = preproc(x, bit_depth=2, clip_values=(0, 4))
        self.assertTrue(np.array_equal(x, np.arange(5)))
        self.assertTrue(np.allclose(squeezed_x, [0, 1.33, 2.67, 2.67, 4], atol=1e-1))


if __name__ == '__main__':
    unittest.main()
