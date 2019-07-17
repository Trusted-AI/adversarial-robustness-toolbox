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
from art.utils import master_seed

logger = logging.getLogger('testLogger')


class TestLocalSpatialSmoothing(unittest.TestCase):
    def setUp(self):
        # Set master seed
        master_seed(1234)

    def test_ones(self):
        m, n = 10, 2
        x = np.ones((1, m, n, 3))

        # Start to test
        for window_size in range(1, 20):
            preprocess = SpatialSmoothing(window_size=window_size)
            smoothed_x, _ = preprocess(x)
            self.assertTrue((smoothed_x == 1).all())

    def test_fix(self):
        x = np.array([[[[0.1], [0.2], [0.3]], [[0.7], [0.8], [0.9]], [[0.4], [0.5], [0.6]]]]).astype(np.float32)

        # Start to test
        preprocess = SpatialSmoothing(window_size=3)
        x_smooth, _ = preprocess(x)
        self.assertTrue((x_smooth == np.array(
            [[[[0.2], [0.3], [0.3]], [[0.4], [0.5], [0.6]], [[0.5], [0.6], [0.6]]]]).astype(np.float32)).all())

        preprocess = SpatialSmoothing(window_size=1)
        x_smooth, _ = preprocess(x)
        self.assertTrue((x_smooth == x).all())

        preprocess = SpatialSmoothing(window_size=2)
        x_smooth, _ = preprocess(x)
        self.assertTrue((x_smooth == np.array(
            [[[[0.1], [0.2], [0.3]], [[0.7], [0.7], [0.8]], [[0.7], [0.7], [0.8]]]]).astype(np.float32)).all())

    def test_channels(self):
        x = np.arange(9).reshape(1, 1, 3, 3)
        preprocess = SpatialSmoothing(channel_index=1)
        x_smooth, _ = preprocess(x)

        x_new = np.arange(9).reshape(1, 3, 3, 1)
        preprocess = SpatialSmoothing()
        x_new_smooth, _ = preprocess(x_new)

        self.assertTrue((x_smooth[0, 0] == x_new_smooth[0, :, :, 0]).all())

    def test_failure(self):
        x = np.arange(10).reshape(5, 2)
        preprocess = SpatialSmoothing(channel_index=1)
        with self.assertRaises(ValueError) as context:
            preprocess(x)

        self.assertIn('Feature vectors detected.', str(context.exception))


if __name__ == '__main__':
    unittest.main()
