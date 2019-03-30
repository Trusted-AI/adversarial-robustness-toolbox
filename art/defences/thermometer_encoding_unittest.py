from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

import numpy as np

from art.defences.thermometer_encoding import ThermometerEncoding
from art.utils import master_seed

logger = logging.getLogger('testLogger')


class TestThermometerEncoding(unittest.TestCase):
    def setUp(self):
        # Set master seed
        master_seed(1234)

    def test_all(self):
        # Test data
        x = np.array([[[[0.2, 0.6, 0.8], [0.9, 0.4, 0.3], [0.2, 0.8, 0.5]],
                      [[0.2, 0.6, 0.8], [0.9, 0.4, 0.3], [0.2, 0.8, 0.5]]],
                      [[[0.2, 0.6, 0.8], [0.9, 0.4, 0.3], [0.2, 0.8, 0.5]],
                      [[0.2, 0.6, 0.8], [0.9, 0.4, 0.3], [0.2, 0.8, 0.5]]]])

        # Create an instance of ThermometerEncoding
        th_encoder = ThermometerEncoding(num_space=4)

        # Preprocess
        x_preproc, _ = th_encoder(x)

        # Test
        self.assertTrue(x_preproc.shape == (2, 2, 3, 12))

        true_value = np.array([[[[1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1],
                                [1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1]], [[1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1],
                                [0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1]]],
                                [[[1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1],
                                [1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1]], [[1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1],
                                [0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1]]]])
        self.assertTrue((x_preproc == true_value).all())


if __name__ == '__main__':
    unittest.main()
