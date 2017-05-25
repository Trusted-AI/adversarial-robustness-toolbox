import unittest

import numpy as np

from keras import backend as K
from keras.models import Sequential

from src.layers import activations

class TestBoundedReLU(unittest.TestCase):

    def test_output(self):

        x = np.random.uniform(-2, 2, size=(1, 100))

        model = Sequential()
        layer = activations.BoundedReLU(input_shape=(100, ))
        model.add(layer)

        # model.outputs = [layer.output]

        y = model.predict(x)

        self.assertTrue(np.all(y >= 0))
        self.assertTrue(np.all(y <= 1))

        model = Sequential()
        layer = activations.BoundedReLU(alpha=0.1, max_value=0.5, input_shape=(100, ))
        model.add(layer)

        y = model.predict(x)

        min_y = np.min(x)*0.1

        self.assertTrue(np.all(y >= min_y))
        self.assertTrue(np.all(y <= 0.5))


if __name__ == '__main__':
    unittest.main()