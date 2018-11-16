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

from tensorflow.examples.tutorials.mnist import input_data
from keras.datasets import cifar10
import numpy as np

from art.defences.jpeg_compression import JpegCompression

logger = logging.getLogger('testLogger')


class TestJpegCompression(unittest.TestCase):
    def test_one_channel(self):
        mnist = input_data.read_data_sets("tmp/MNIST_data/")
        x = np.reshape(mnist.test.images[0:2], (-1, 28, 28, 1))
        preprocess = JpegCompression()
        compressed_x = preprocess(x, quality=70)
        self.assertTrue((compressed_x.shape == x.shape))
        self.assertTrue((compressed_x <= 1.0).all())
        self.assertTrue((compressed_x >= 0.0).all())

    def test_three_channels(self):
        (train_features, train_labels), (test_data, test_label) = cifar10.load_data()
        x = train_features[:2] / 255.0
        preprocess = JpegCompression()
        compressed_x = preprocess(x, quality=80)
        self.assertTrue((compressed_x.shape == x.shape))
        self.assertTrue((compressed_x <= 1.0).all())
        self.assertTrue((compressed_x >= 0.0).all())

    def test_channel_index(self):
        (train_features, train_labels), (test_data, test_label) = cifar10.load_data()
        x = train_features[:2] / 255.0
        x = np.swapaxes(x, 1, 3)
        preprocess = JpegCompression(channel_index=1)
        compressed_x = preprocess(x, quality=80)
        self.assertTrue((compressed_x.shape == x.shape))
        self.assertTrue((compressed_x <= 1.0).all())
        self.assertTrue((compressed_x >= 0.0).all())


if __name__ == '__main__':
    unittest.main()



