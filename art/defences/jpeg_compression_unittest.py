from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

from tensorflow.examples.tutorials.mnist import input_data
from keras.datasets import cifar10
import numpy as np

from art.defences.jpeg_compression import JpegCompression
from art.utils import load_mnist, master_seed

logger = logging.getLogger('testLogger')


class TestJpegCompression(unittest.TestCase):
    def setUp(self):
        # Set master seed
        master_seed(1234)

    def test_one_channel(self):
        (x_train, _), (_, _), _, _ = load_mnist()
        x_train = x_train[:2]
        preprocess = JpegCompression()
        compressed_x, _ = preprocess(x_train, quality=70)
        self.assertTrue((compressed_x.shape == x_train.shape))
        self.assertTrue((compressed_x <= 1.0).all())
        self.assertTrue((compressed_x >= 0.0).all())

    def test_three_channels(self):
        (train_features, _), (_, _) = cifar10.load_data()
        x = train_features[:2] / 255.0
        preprocess = JpegCompression()
        compressed_x, _ = preprocess(x, quality=80)
        self.assertTrue((compressed_x.shape == x.shape))
        self.assertTrue((compressed_x <= 1.0).all())
        self.assertTrue((compressed_x >= 0.0).all())

    def test_channel_index(self):
        (train_features, _), (_, _) = cifar10.load_data()
        x = train_features[:2] / 255.0
        x = np.swapaxes(x, 1, 3)
        preprocess = JpegCompression(channel_index=1)
        compressed_x, _ = preprocess(x, quality=80)
        self.assertTrue((compressed_x.shape == x.shape))
        self.assertTrue((compressed_x <= 1.0).all())
        self.assertTrue((compressed_x >= 0.0).all())


if __name__ == '__main__':
    unittest.main()
