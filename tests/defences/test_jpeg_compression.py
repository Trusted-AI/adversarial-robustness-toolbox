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
        clip_values = (0, 1)
        (x_train, _), (_, _), _, _ = load_mnist()
        x_train = x_train[:2]
        preprocess = JpegCompression(clip_values=clip_values, quality=70)
        x_compressed, _ = preprocess(x_train)
        self.assertEqual(x_compressed.shape, x_train.shape)
        self.assertTrue((x_compressed >= clip_values[0]).all())
        self.assertTrue((x_compressed <= clip_values[1]).all())

    def test_three_channels_0_1(self):
        clip_values = (0, 1)
        (train_features, _), (_, _) = cifar10.load_data()
        x = train_features[:2] / 255.0
        preprocess = JpegCompression(clip_values=clip_values, quality=80)
        x_compressed, _ = preprocess(x)
        self.assertEqual(x_compressed.shape, x.shape)
        self.assertTrue((x_compressed >= clip_values[0]).all())
        self.assertTrue((x_compressed <= clip_values[1]).all())
        self.assertAlmostEqual(x_compressed[0, 14, 14, 0], 0.92941177)
        self.assertAlmostEqual(x_compressed[0, 14, 14, 1], 0.8039216)
        self.assertAlmostEqual(x_compressed[0, 14, 14, 2], 0.6117647)

    def test_three_channels_0_255(self):
        clip_values = (0, 255)
        (train_features, _), (_, _) = cifar10.load_data()
        x = train_features[:2]
        preprocess = JpegCompression(clip_values=clip_values, quality=80)
        x_compressed, _ = preprocess(x)
        self.assertEqual(x_compressed.shape, x.shape)
        self.assertTrue((x_compressed >= clip_values[0]).all())
        self.assertTrue((x_compressed <= clip_values[1]).all())
        self.assertAlmostEqual(x_compressed[0, 14, 14, 0], 0.92941177 * clip_values[1], places=4)
        self.assertAlmostEqual(x_compressed[0, 14, 14, 1], 0.8039216 * clip_values[1], places=4)
        self.assertAlmostEqual(x_compressed[0, 14, 14, 2], 0.6117647 * clip_values[1], places=4)

    def test_channel_index(self):
        clip_values = (0, 255)
        (train_features, _), (_, _) = cifar10.load_data()
        x = train_features[:2]
        x = np.swapaxes(x, 1, 3)
        preprocess = JpegCompression(clip_values=clip_values, channel_index=1, quality=80)
        x_compressed, _ = preprocess(x)
        self.assertTrue((x_compressed.shape == x.shape))
        self.assertTrue((x_compressed >= clip_values[0]).all())
        self.assertTrue((x_compressed <= clip_values[1]).all())

    def test_failure_feature_vectors(self):
        clip_values = (0, 1)
        x = np.random.rand(10, 3)
        preprocess = JpegCompression(clip_values=clip_values, channel_index=1, quality=80)

        # Assert that value error is raised for feature vectors
        with self.assertRaises(ValueError) as context:
            preprocess(x)

        self.assertTrue('Feature vectors detected.' in str(context.exception))

    def test_failure_clip_values_negative(self):
        clip_values = (-1, 1)

        # Assert that value error is raised
        with self.assertRaises(ValueError) as context:
            _ = JpegCompression(clip_values=clip_values, channel_index=1, quality=80)

        self.assertTrue('min value must be 0.' in str(context.exception))

    def test_failure_clip_values_unexpected_maximum(self):
        clip_values = (0, 2)

        # Assert that value error is raised
        with self.assertRaises(ValueError) as context:
            _ = JpegCompression(clip_values=clip_values, channel_index=1, quality=80)

        self.assertIn('max value must be either 1 or 255.', str(context.exception))


if __name__ == '__main__':
    unittest.main()
