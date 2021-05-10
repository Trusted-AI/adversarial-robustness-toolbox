# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
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

from art.defences.preprocessor import VideoCompression

from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.fixture
def video_batch(channels_first):
    """
    Video fixture of shape NFHWC and NCFHW.
    """
    test_input = np.stack((np.zeros((3, 25, 4, 6)), np.ones((3, 25, 4, 6))))
    if not channels_first:
        test_input = np.transpose(test_input, (0, 2, 3, 4, 1))
    test_output = test_input.copy()
    return test_input, test_output


@pytest.mark.parametrize("channels_first", [True, False])
@pytest.mark.skip_framework("keras", "pytorch", "scikitlearn", "mxnet")
def test_video_compresssion(art_warning, video_batch, channels_first):
    try:
        test_input, test_output = video_batch
        video_compression = VideoCompression(video_format="mp4", constant_rate_factor=0, channels_first=channels_first)

        assert_array_equal(video_compression(test_input)[0], test_output)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("keras", "pytorch", "scikitlearn", "mxnet")
def test_compress_video_call(art_warning):
    try:
        test_input = np.arange(12).reshape((1, 3, 1, 2, 2))
        video_compression = VideoCompression(video_format="mp4", constant_rate_factor=50, channels_first=True)

        assert np.any(np.not_equal(video_compression(test_input)[0], test_input))
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.parametrize("constant_rate_factor", [-1, 52])
def test_constant_rate_factor_error(art_warning, constant_rate_factor):
    try:
        exc_msg = r"Constant rate factor must be an integer in the range \[0, 51\]."
        with pytest.raises(ValueError, match=exc_msg):
            VideoCompression(video_format="", constant_rate_factor=constant_rate_factor)
    except ARTTestException as e:
        art_warning(e)


def test_non_spatio_temporal_data_error(art_warning, image_batch_small):
    try:
        test_input = image_batch_small
        video_compression = VideoCompression(video_format="")

        exc_msg = "Video compression can only be applied to spatio-temporal data."
        with pytest.raises(ValueError, match=exc_msg):
            video_compression(test_input)
    except ARTTestException as e:
        art_warning(e)
