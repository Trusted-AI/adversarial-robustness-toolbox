from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

from tensorflow.examples.tutorials.mnist import input_data
from keras.datasets import cifar10
import numpy as np

from art.defences.variance_minimization import TotalVarMin
from art.utils import master_seed

logger = logging.getLogger('testLogger')


class TestTotalVarMin(unittest.TestCase):
    def setUp(self):
        # Set master seed
        master_seed(1234)

    def test_one_channel(self):
        mnist = input_data.read_data_sets("tmp/MNIST_data/")
        x = np.reshape(mnist.test.images[0:2], (-1, 28, 28, 1))
        preprocess = TotalVarMin()
        preprocessed_x = preprocess(x)
        self.assertTrue((preprocessed_x.shape == x.shape))
        self.assertTrue((preprocessed_x <= 1.0).all())
        self.assertTrue((preprocessed_x >= 0.0).all())
        self.assertFalse((preprocessed_x == x).all())

    def test_three_channels(self):
        (train_features, _), (_, _) = cifar10.load_data()
        x = train_features[:2] / 255.0
        preprocess = TotalVarMin()
        preprocessed_x = preprocess(x)
        self.assertTrue((preprocessed_x.shape == x.shape))
        self.assertTrue((preprocessed_x <= 1.0).all())
        self.assertTrue((preprocessed_x >= 0.0).all())
        self.assertFalse((preprocessed_x == x).all())


if __name__ == '__main__':
    unittest.main()
