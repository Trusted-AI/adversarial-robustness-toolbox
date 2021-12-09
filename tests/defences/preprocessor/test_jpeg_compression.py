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

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from art.config import ART_NUMPY_DTYPE
from art.defences.preprocessor import JpegCompression
from tests.utils import ARTTestException

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


@pytest.fixture(params=[1, 2, 3, 5], ids=["grayscale", "grayscale-2", "RGB", "grayscale-5"])
def image_batch(request, channels_first):
    """
    Image fixtures of shape NHWC and NCHW.
    """
    channels = request.param
    image_input = DataGenerator(channels_first, channels)
    test_input = image_input.get_data()
    test_output = test_input.copy()
    return test_input, test_output


@pytest.fixture(params=[1, 2, 3, 5], ids=["grayscale", "grayscale-2", "RGB", "grayscale-5"])
def video_batch(request, channels_first):
    """
    Video fixtures of shape NFHWC and NCFHW.
    """
    channels = request.param
    video_input = DataGenerator(channels_first, channels, image_data=False)
    test_input = video_input.get_data()
    test_output = test_input.copy()
    return test_input, test_output


@pytest.mark.framework_agnostic
@pytest.mark.parametrize("channels_first", [True, False])
def test_jpeg_compression_image_data(art_warning, image_batch, channels_first, framework):
    try:
        test_input, test_output = image_batch
        jpeg_compression = JpegCompression(clip_values=(0, 255), channels_first=channels_first)

        assert_array_equal(jpeg_compression(test_input)[0], test_output)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.parametrize("channels_first", [True, False])
@pytest.mark.framework_agnostic
def test_jpeg_compression_video_data(art_warning, video_batch, channels_first):
    try:
        test_input, test_output = video_batch
        jpeg_compression = JpegCompression(clip_values=(0, 255), channels_first=channels_first)

        assert_array_equal(jpeg_compression(test_input)[0], test_output)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.parametrize("channels_first", [False])
@pytest.mark.framework_agnostic
def test_jpeg_compress(art_warning, image_batch, channels_first):
    try:
        test_input, test_output = image_batch
        # Run only for grayscale [1] and RGB [3] data because testing `_compress` which is applied internally only to
        # either grayscale or RGB data.
        if test_input.shape[-1] in [1, 3]:
            jpeg_compression = JpegCompression(clip_values=(0, 255))

            image_mode = "RGB" if test_input.shape[-1] == 3 else "L"
            test_single_input = np.squeeze(test_input[0]).astype(np.uint8)
            test_single_output = np.squeeze(test_output[0]).astype(np.uint8)

            assert_array_equal(jpeg_compression._compress(test_single_input, image_mode), test_single_output)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_non_spatial_data_error(art_warning, tabular_batch):
    try:
        test_input = tabular_batch
        jpeg_compression = JpegCompression(clip_values=(0, 255), channels_first=True)

        exc_msg = "Unrecognized input dimension. JPEG compression can only be applied to image and video data."
        with pytest.raises(ValueError, match=exc_msg):
            jpeg_compression(test_input)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_check_params(art_warning):
    try:
        with pytest.raises(ValueError):
            JpegCompression(clip_values=(-1, 255))

        with pytest.raises(ValueError):
            _ = JpegCompression(clip_values=(0, 2))

        with pytest.raises(ValueError):
            _ = JpegCompression(clip_values=(0, 1), quality=-1)

        with pytest.raises(ValueError):
            _ = JpegCompression(clip_values=(0, 1, 2))

        with pytest.raises(ValueError):
            _ = JpegCompression(clip_values=(0, 1), verbose="False")

    except ARTTestException as e:
        art_warning(e)
