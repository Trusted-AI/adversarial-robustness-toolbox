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
import logging

import numpy as np
import pytest
import resampy

from art.defences.preprocessor import Resample
from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.fixture(params=[1, 2], ids=["one_channel", "two_channel"])
def audio_batch(request):
    """
    Create audio fixtures of shape (batch_size=2, channels={1,2}, samples).
    """
    sample_rate_orig = 16000
    sample_rate_new = 8000
    test_input = np.zeros((2, request.param, sample_rate_orig), dtype=np.int16)
    test_output = np.zeros((2, request.param, sample_rate_new), dtype=np.int16)
    return test_input, test_output, sample_rate_orig, sample_rate_new


@pytest.mark.framework_agnostic
def test_sample_rate_original_error(art_warning):
    try:
        exc_msg = "Original sampling rate be must a positive integer."
        with pytest.raises(ValueError, match=exc_msg):
            Resample(sr_original=0, sr_new=16000)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_sample_rate_new_error(art_warning):
    try:
        exc_msg = "New sampling rate be must a positive integer."
        with pytest.raises(ValueError, match=exc_msg):
            Resample(sr_original=16000, sr_new=0)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_non_temporal_data_error(art_warning, image_batch_small):
    try:
        test_input = image_batch_small
        resample = Resample(16000, 16000)

        exc_msg = "Resampling can only be applied to temporal data across at least one channel."
        with pytest.raises(ValueError, match=exc_msg):
            resample(test_input)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_resample(art_warning, audio_batch, mocker):
    try:
        test_input, test_output, sr_orig, sr_new = audio_batch

        mocker.patch("resampy.resample", autospec=True)
        resampy.resample.return_value = test_input[:, :, :sr_new]

        resampler = Resample(sr_original=sr_orig, sr_new=sr_new, channels_first=True)
        assert resampler(test_input)[0].shape == test_output.shape
    except ARTTestException as e:
        art_warning(e)
