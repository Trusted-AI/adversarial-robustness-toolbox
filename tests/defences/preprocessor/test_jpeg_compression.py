# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
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
import pytest
from keras.datasets import cifar10
from numpy.testing import assert_array_equal
from tests.utils import master_seed

from art.config import ART_NUMPY_DTYPE
from art.defences.preprocessor import JpegCompression
from art.utils import load_mnist

logger = logging.getLogger(__name__)


class DataGenerator:
    """
    Create image or video batch.
    """

    def __init__(self, channels_first, channels, image_data=True, batch_size=2):
        self.channels_first = channels_first
        self.channels = (channels,)
        self.batch_size = batch_size
        self.image_data = image_data

    def get_data(self):
        temporal_index = () if self.image_data else (2,)
        if self.channels_first:
            data_shape = (self.batch_size,) + self.channels + temporal_index + (8, 12)
        else:
            data_shape = (self.batch_size,) + temporal_index + (8, 12) + self.channels
        return (255 * np.ones(data_shape)).astype(ART_NUMPY_DTYPE)


@pytest.fixture(params=[1, 3], ids=["grayscale", "RGB"])
def image_batch(request, channels_first):
    """
    Image fixtures of shape NHWC and NCHW.
    """
    channels = request.param
    image_input = DataGenerator(channels_first, channels)
    test_input = image_input.get_data()
    test_output = test_input.copy()
    return test_input, test_output


@pytest.fixture(params=[1, 3], ids=["grayscale", "RGB"])
def video_batch(request, channels_first):
    """
    Video fixtures of shape NFHWC and NCFHW.
    """
    channels = request.param
    video_input = DataGenerator(channels_first, channels, image_data=False)
    test_input = video_input.get_data()
    test_output = test_input.copy()
    return test_input, test_output


@pytest.fixture
def tabular_batch():
    """Create tabular data fixture of shape (batch_size, features)."""
    return np.zeros((2, 4))


class TestJpegCompression:
    """Test JpegCompression."""

    @pytest.mark.parametrize("channels_first", [True, False])
    def test_jpeg_compression_image_data(self, image_batch, channels_first):
        channel_index = 1 if channels_first else 3
        test_input, test_output = image_batch
        jpeg_compression = JpegCompression(clip_values=(0, 255), channel_index=channel_index)

        assert_array_equal(jpeg_compression(test_input)[0], test_output)

    @pytest.mark.parametrize("channels_first", [True, False])
    def test_jpeg_compression_video_data(self, video_batch, channels_first):
        channel_index = 1 if channels_first else 4
        test_input, test_output = video_batch
        jpeg_compression = JpegCompression(clip_values=(0, 255), channel_index=channel_index)

        assert_array_equal(jpeg_compression(test_input)[0], test_output)

    def test_channel_index_error(self):
        exc_msg = "Data channel must be an integer equal to 1, 3 or 4. The batch dimension is not a valid channel."
        with pytest.raises(ValueError, match=exc_msg):
            JpegCompression(clip_values=(0, 255), channel_index=0)

    def test_non_spatial_data_error(self, tabular_batch):
        test_input = tabular_batch
        jpeg_compression = JpegCompression(clip_values=(0, 255), channel_index=1)

        exc_msg = "Feature vectors detected. JPEG compression can only be applied to data with spatial dimensions."
        with pytest.raises(ValueError, match=exc_msg):
            jpeg_compression(test_input)

    def test_negative_clip_values_error(self):
        exc_msg = "'clip_values' min value must be 0."
        with pytest.raises(ValueError, match=exc_msg):
            JpegCompression(clip_values=(-1, 255), channel_index=1)

    def test_maximum_clip_values_error(self):
        exc_msg = "'clip_values' max value must be either 1 or 255."
        with pytest.raises(ValueError, match=exc_msg):
            JpegCompression(clip_values=(0, 2), channel_index=1)


class TestJpegCompressionLegacy(unittest.TestCase):
    def setUp(self):
        master_seed(seed=1234)

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
        x_original = x.copy()
        preprocess = JpegCompression(clip_values=clip_values, quality=80)
        x_compressed, _ = preprocess(x)
        self.assertEqual(x_compressed.shape, x.shape)
        self.assertTrue((x_compressed >= clip_values[0]).all())
        self.assertTrue((x_compressed <= clip_values[1]).all())
        self.assertAlmostEqual(x_compressed[0, 14, 14, 0], 0.92941177)
        self.assertAlmostEqual(x_compressed[0, 14, 14, 1], 0.8039216)
        self.assertAlmostEqual(x_compressed[0, 14, 14, 2], 0.6117647)
        # Check that x has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_original - x))), 0.0, delta=0.00001)

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

        self.assertTrue("Feature vectors detected." in str(context.exception))

    def test_failure_clip_values_negative(self):
        clip_values = (-1, 1)

        # Assert that value error is raised
        with self.assertRaises(ValueError) as context:
            _ = JpegCompression(clip_values=clip_values, channel_index=1, quality=80)

        self.assertTrue("min value must be 0." in str(context.exception))

    def test_failure_clip_values_unexpected_maximum(self):
        clip_values = (0, 2)

        # Assert that value error is raised
        with self.assertRaises(ValueError) as context:
            _ = JpegCompression(clip_values=clip_values, channel_index=1, quality=80)

        self.assertIn("max value must be either 1 or 255.", str(context.exception))


# if __name__ == "__main__":
#    unittest.main()
