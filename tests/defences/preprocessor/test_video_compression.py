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


@pytest.fixture
def image_batch():
    """Create image fixture of shape (batch_size, channels, width, height)."""
    return np.zeros((2, 1, 4, 4))


class TestVideoCompression:
    """Test VideoCompression."""

    @pytest.mark.parametrize("channels_first", [True, False])
    @pytest.mark.skipMlFramework("keras", "pytorch", "scikitlearn")
    def test_video_compresssion(self, video_batch, channels_first):
        test_input, test_output = video_batch
        video_compression = VideoCompression(video_format="mp4", constant_rate_factor=0, channels_first=channels_first)

        assert_array_equal(video_compression(test_input)[0], test_output)

    @pytest.mark.skipMlFramework("keras", "pytorch", "scikitlearn")
    def test_compress_video_call(self):
        test_input = np.arange(12).reshape((1, 3, 1, 2, 2))
        video_compression = VideoCompression(video_format="mp4", constant_rate_factor=50, channels_first=True)

        assert np.any(np.not_equal(video_compression(test_input)[0], test_input))

    @pytest.mark.parametrize("constant_rate_factor", [-1, 52])
    def test_constant_rate_factor_error(self, constant_rate_factor):
        exc_msg = r"Constant rate factor must be an integer in the range \[0, 51\]."
        with pytest.raises(ValueError, match=exc_msg):
            VideoCompression(video_format="", constant_rate_factor=constant_rate_factor)

    def test_non_spatio_temporal_data_error(self, image_batch):
        test_input = image_batch
        video_compression = VideoCompression(video_format="")

        exc_msg = "Video compression can only be applied to spatio-temporal data."
        with pytest.raises(ValueError, match=exc_msg):
            video_compression(test_input)


if __name__ == "__main__":
    pytest.cmdline.main("-q -s {} --mlFramework=tensorflow --durations=0".format(__file__).split(" "))
