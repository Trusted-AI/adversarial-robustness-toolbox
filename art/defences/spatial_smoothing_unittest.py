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
            preprocess = SpatialSmoothing()
            smoothed_x, _ = preprocess(x, window_size)
            self.assertTrue((smoothed_x == 1).all())

    def test_fix(self):
        x = np.array([[[[0.1], [0.2], [0.3]], [[0.7], [0.8], [0.9]], [[0.4], [0.5], [0.6]]]]).astype(np.float32)

        # Start to test
        preprocess = SpatialSmoothing()
        smooth_x, _ = preprocess(x, window_size=3)
        self.assertTrue((smooth_x == np.array(
            [[[[0.2], [0.3], [0.3]], [[0.4], [0.5], [0.6]], [[0.5], [0.6], [0.6]]]]).astype(np.float32)).all())

        smooth_x, _ = preprocess(x, window_size=1)
        self.assertTrue((smooth_x == x).all())

        smooth_x, _ = preprocess(x, window_size=2)
        self.assertTrue((smooth_x == np.array(
            [[[[0.1], [0.2], [0.3]], [[0.7], [0.7], [0.8]], [[0.7], [0.7], [0.8]]]]).astype(np.float32)).all())

    def test_channels(self):
        x = np.arange(9).reshape(1, 1, 3, 3)
        preprocess = SpatialSmoothing(channel_index=1)
        smooth_x, _ = preprocess(x)

        new_x = np.arange(9).reshape(1, 3, 3, 1)
        preprocess = SpatialSmoothing()
        new_smooth_x, _ = preprocess(new_x)

        self.assertTrue((smooth_x[0, 0] == new_smooth_x[0, :, :, 0]).all())


if __name__ == '__main__':
    unittest.main()
