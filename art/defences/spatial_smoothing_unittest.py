from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import numpy as np

from art.defences.spatial_smoothing import SpatialSmoothing


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
        x = np.array([[[[1], [2], [3]], [[7], [8], [9]], [[4], [5], [6]]]])

        # Start to test
        preprocess = SpatialSmoothing()
        smooth_x = preprocess(x, window_size=3)
        self.assertTrue((smooth_x == np.array(
            [[[[2], [3], [3]], [[4], [5], [6]], [[5], [6], [6]]]])).all())

        smooth_x = preprocess(x, window_size=1)
        self.assertTrue((smooth_x == x).all())

        smooth_x = preprocess(x, window_size=2)
        self.assertTrue((smooth_x == np.array(
            [[[[1], [2], [3]], [[7], [7], [8]], [[7], [7], [8]]]])).all())

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
