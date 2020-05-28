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

from art.defences.preprocessor import Mp3Compression

logger = logging.getLogger(__name__)


class AudioInput:
    """
    Create audio batch.
    """

    def __init__(self, channel_index, channels, sample_rate=44100, batch_size=2):
        self.channel_index = channel_index
        self.channels = channels
        self.sample_rate = sample_rate
        self.batch_size = batch_size

    def get_data(self):
        if self.channel_index == 1:
            return np.zeros(
                (self.batch_size, self.channels, self.sample_rate), dtype=np.int16
            )
        else:
            return np.zeros(
                (self.batch_size, self.sample_rate, self.channels), dtype=np.int16
            )


@pytest.fixture(params=[1, 2], ids=["mono", "stereo"])
def audio_batch(request, channel_index):
    """
    Audio fixtures of shape `(batch_size, channels, samples)` or `(batch_size, samples, channels)`.
    """
    channels = request.param
    audio_input = AudioInput(channel_index, channels)
    test_input = audio_input.get_data()
    test_output = test_input.copy()
    return test_input, test_output, audio_input.sample_rate


@pytest.fixture
def image_batch():
    """Create image fixture of shape (batch_size, channels, width, height)."""
    return np.zeros((2, 1, 4, 4))


class TestMp3Compression:
    """Test Mp3Compresssion."""

    def test_channel_index_error(self):
        exc_msg = "Data channel must be an integer equal to 1 or 2. The batch dimension is not a valid channel."
        with pytest.raises(ValueError, match=exc_msg):
            Mp3Compression(sample_rate=16000, channel_index=3)

    def test_sample_rate_error(self):
        exc_msg = "Sample rate be must a positive integer."
        with pytest.raises(ValueError, match=exc_msg):
            Mp3Compression(sample_rate=0, channel_index=1)

    def test_non_temporal_data_error(self, image_batch):
        test_input = image_batch
        mp3compression = Mp3Compression(sample_rate=16000, channel_index=1)

        exc_msg = "Mp3 compression can only be applied to temporal data across at least one channel."
        with pytest.raises(ValueError, match=exc_msg):
            mp3compression(test_input)

    @pytest.mark.parametrize("channel_index", [1, 2])
    def test_mp3_compresssion(self, audio_batch, channel_index):
        test_input, test_output, sample_rate = audio_batch
        mp3compression = Mp3Compression(
            sample_rate=sample_rate, channel_index=channel_index
        )

        assert_array_equal(mp3compression(test_input)[0], test_output)


if __name__ == "__main__":
    pytest.cmdline.main("-q -s {} --mlFramework=tensorflow --durations=0".format(__file__).split(" "))
