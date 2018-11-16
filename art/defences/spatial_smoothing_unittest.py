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

from art.defences.spatial_smoothing import SpatialSmoothing

logger = logging.getLogger('testLogger')


class TestLocalSpatialSmoothing(unittest.TestCase):
    def test_ones(self):
        m, n = 10, 2
        x = np.ones((1, m, n, 3))

        # Start to test
        for window_size in range(1, 20):
            preprocess = SpatialSmoothing()
            smoothed_x = preprocess(x, window_size)
            self.assertTrue((smoothed_x == 1).all())

    def test_fix(self):
        x = np.array([[[[0.1], [0.2], [0.3]], [[0.7], [0.8], [0.9]], [[0.4], [0.5], [0.6]]]]).astype(np.float32)

        # Start to test
        preprocess = SpatialSmoothing()
        smooth_x = preprocess(x, window_size=3)
        self.assertTrue((smooth_x == np.array(
            [[[[0.2], [0.3], [0.3]], [[0.4], [0.5], [0.6]], [[0.5], [0.6], [0.6]]]]).astype(np.float32)).all())

        smooth_x = preprocess(x, window_size=1)
        self.assertTrue((smooth_x == x).all())

        smooth_x = preprocess(x, window_size=2)
        self.assertTrue((smooth_x == np.array(
            [[[[0.1], [0.2], [0.3]], [[0.7], [0.7], [0.8]], [[0.7], [0.7], [0.8]]]]).astype(np.float32)).all())

    def test_channels(self):
        x = np.arange(9).reshape(1, 1, 3, 3)
        preprocess = SpatialSmoothing(channel_index=1)
        smooth_x = preprocess(x)

        new_x = np.arange(9).reshape(1, 3, 3, 1)
        preprocess = SpatialSmoothing()
        new_smooth_x = preprocess(new_x)

        self.assertTrue((smooth_x[0, 0] == new_smooth_x[0, :, :, 0]).all())


if __name__ == '__main__':
    unittest.main()

