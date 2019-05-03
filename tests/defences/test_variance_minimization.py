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
