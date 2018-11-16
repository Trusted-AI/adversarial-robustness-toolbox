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

from art.defences.thermometer_encoding import ThermometerEncoding

logger = logging.getLogger('testLogger')


class TestThermometerEncoding(unittest.TestCase):
    def test_all(self):
        # Test data
        x = np.array([[[[0.2, 0.6, 0.8], [0.9, 0.4, 0.3], [0.2, 0.8, 0.5]],
                      [[0.2, 0.6, 0.8], [0.9, 0.4, 0.3], [0.2, 0.8, 0.5]]],
                      [[[0.2, 0.6, 0.8], [0.9, 0.4, 0.3], [0.2, 0.8, 0.5]],
                      [[0.2, 0.6, 0.8], [0.9, 0.4, 0.3], [0.2, 0.8, 0.5]]]])

        # Create an instance of ThermometerEncoding
        thencoder = ThermometerEncoding(num_space=4)

        # Preprocess
        pp_x = thencoder(x)

        # Test
        self.assertTrue(pp_x.shape == (2, 2, 3, 12))

        true_value = np.array([[[[1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1],
                                [1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1]], [[1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1],
                                [0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1]]],
                                [[[1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1],
                                [1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1]], [[1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1],
                                [0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1]]]])
        self.assertTrue((pp_x == true_value).all())


if __name__ == '__main__':
    unittest.main()



