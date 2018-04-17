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
from __future__ import absolute_import, division, print_function

import unittest

from keras.models import Sequential
import numpy as np

from art.layers import activations


class TestBoundedReLU(unittest.TestCase):
    def test_output(self):
        x = np.random.uniform(-2, 2, size=(1, 100))

        min_value, max_value = 0, 1
        model = Sequential()
        layer = activations.BoundedReLU(input_shape=(100, ))
        model.add(layer)
        y = model.predict(x)

        self.assertTrue(np.all(y >= min_value))
        self.assertTrue(np.all(y <= max_value))

        alpha, max_value = .1, .5
        model = Sequential()
        layer = activations.BoundedReLU(alpha=alpha, max_value=max_value, input_shape=(100, ))
        model.add(layer)
        y = model.predict(x)
        min_y = -2. * alpha

        self.assertTrue(np.all(y <= max_value))
        self.assertTrue(np.all(y >= min_y))


if __name__ == '__main__':
    unittest.main()
