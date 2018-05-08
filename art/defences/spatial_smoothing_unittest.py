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
        smoothed_x = preprocess(x, 3)
        self.assertTrue((smoothed_x == np.array(
            [[[[2], [3], [3]], [[4], [5], [6]], [[5], [6], [6]]]])).all())

        smoothed_x = preprocess(x, 1)
        self.assertTrue((smoothed_x == x).all())

        smoothed_x = preprocess(x, 2)
        self.assertTrue((smoothed_x == np.array(
            [[[[1], [2], [3]], [[7], [7], [8]], [[7], [7], [8]]]])).all())


if __name__ == '__main__':
    unittest.main()




