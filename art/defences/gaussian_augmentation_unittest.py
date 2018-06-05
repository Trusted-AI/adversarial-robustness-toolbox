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

import unittest

import numpy as np

from art.defences.gaussian_augmentation import GaussianAugmentation


class TestGaussianAugmentation(unittest.TestCase):
    def test_small_size(self):
        x = np.arange(15).reshape((5, 3))
        ga = GaussianAugmentation()
        new_x = ga(x, ratio=.4)
        self.assertTrue(new_x.shape == (7, 3))

    def test_double_size(self):
        x = np.arange(12).reshape((4, 3))
        ga = GaussianAugmentation()
        new_x = ga(x)
        self.assertTrue(new_x.shape[0] == 2 * x.shape[0])

    def test_multiple_size(self):
        x = np.arange(12).reshape((4, 3))
        ga = GaussianAugmentation(ratio=3.5)
        new_x = ga(x)
        self.assertTrue(int(4.5 * x.shape[0]) == new_x.shape[0])

    def test_labels(self):
        x = np.arange(12).reshape((4, 3))
        y = np.arange(8).reshape((4, 2))

        ga = GaussianAugmentation()
        new_x, new_y = ga(x, y)
        self.assertTrue(new_x.shape[0] == new_y.shape[0] == 8)
        self.assertTrue(new_x.shape[1:] == x.shape[1:])
        self.assertTrue(new_y.shape[1:] == y.shape[1:])


if __name__ == '__main__':
    unittest.main()
