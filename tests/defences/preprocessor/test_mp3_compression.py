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

from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


class AudioInput:
    """
    Create audio batch.
    """

    def __init__(self, channels_first, channels, sample_rate=44100, batch_size=2):
        self.channels_first = channels_first
        self.channels = channels
        self.sample_rate = sample_rate
        self.batch_size = batch_size

    def get_data(self):
        if self.channels_first:
            shape = (self.batch_size, self.channels, self.sample_rate)
        else:
            shape = (self.batch_size, self.sample_rate, self.channels)
        return np.zeros(shape, dtype=np.int16)


@pytest.fixture(params=[1, 2], ids=["mono", "stereo"])
def audio_batch(request, channels_first):
    """
    Audio fixtures of shape `(batch_size, channels, samples)` or `(batch_size, samples, channels)`.
    """
    channels = request.param
    audio_input = AudioInput(channels_first, channels)
    test_input = audio_input.get_data()
    test_output = test_input.copy()
    return test_input, test_output, audio_input.sample_rate


def test_sample_rate_error(art_warning):
    try:
        exc_msg = "Sample rate be must a positive integer."
        with pytest.raises(ValueError, match=exc_msg):
            Mp3Compression(sample_rate=0)
    except ARTTestException as e:
        art_warning(e)


def test_non_temporal_data_error(art_warning, image_batch_small):
    try:
        test_input = image_batch_small
        mp3compression = Mp3Compression(sample_rate=16000)

        exc_msg = "Mp3 compression can only be applied to temporal data across at least one channel."
        with pytest.raises(ValueError, match=exc_msg):
            mp3compression(test_input)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.parametrize("channels_first", [True, False])
@pytest.mark.skip_framework("keras", "pytorch", "scikitlearn", "mxnet")
def test_mp3_compresssion(art_warning, audio_batch, channels_first):
    try:
        test_input, test_output, sample_rate = audio_batch
        mp3compression = Mp3Compression(sample_rate=sample_rate, channels_first=channels_first)

        assert_array_equal(mp3compression(test_input)[0], test_output)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.parametrize("channels_first", [True, False])
@pytest.mark.skip_framework("keras", "pytorch", "scikitlearn", "mxnet")
def test_mp3_compresssion_object(art_warning, audio_batch, channels_first):
    try:
        test_input, test_output, sample_rate = audio_batch
        test_input_object = np.array([x for x in test_input], dtype=object)
        mp3compression = Mp3Compression(sample_rate=sample_rate, channels_first=channels_first)

        assert_array_equal(mp3compression(test_input_object)[0], test_output)

        grad = mp3compression.estimate_gradient(x=test_input_object, grad=np.ones_like(test_input_object))

        assert grad.dtype == object
        assert grad.shape == (2, test_input.shape[1], 44100) if channels_first else (2, 44100, test_input.shape[2])

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("keras", "pytorch", "scikitlearn", "mxnet")
def test_check_params(art_warning):
    try:
        with pytest.raises(ValueError):
            _ = Mp3Compression(sample_rate=1000, verbose="False")

    except ARTTestException as e:
        art_warning(e)
