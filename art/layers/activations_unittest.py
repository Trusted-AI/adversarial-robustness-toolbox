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
