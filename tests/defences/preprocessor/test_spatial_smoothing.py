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

from art.defences.preprocessor import SpatialSmoothing

logger = logging.getLogger(__name__)


@pytest.fixture
def image_batch(channels_first):
    """
    Image fixture of shape NHWC and NCHW.
    """
    test_input = np.repeat(np.array(range(6)).reshape(6, 1), 24, axis=1).reshape(2, 3, 4, 6)
    if not channels_first:
        test_input = np.transpose(test_input, (0, 2, 3, 1))
    test_output = test_input.copy()
    return test_input, test_output


@pytest.fixture
def tabular_batch():
    """
    Create tabular data fixture of shape (batch_size, features).
    """
    return np.zeros((2, 4))


class TestLocalSpatialSmoothing:
    """
    Test SpatialSmoothing.
    """

    def test_spatial_smoothing_median_filter_call(self):
        test_input = np.array([[[[1, 2], [3, 4]]]])
        test_output = np.array([[[[1, 2], [3, 3]]]])
        spatial_smoothing = SpatialSmoothing(channel_index=1, window_size=2)

        assert_array_equal(spatial_smoothing(test_input)[0], test_output)

    @pytest.mark.parametrize("channels_first", [True, False])
    @pytest.mark.parametrize("window_size", [1, 2, 10])
    def test_spatial_smoothing_image_data(self, image_batch, channels_first, window_size):
        channel_index = 1 if channels_first else 3
        test_input, test_output = image_batch
        spatial_smoothing = SpatialSmoothing(channel_index=channel_index, window_size=window_size)

        assert_array_equal(spatial_smoothing(test_input)[0], test_output)

    def test_non_spatial_data_error(self, tabular_batch):
        test_input = tabular_batch
        spatial_smoothing = SpatialSmoothing(channel_index=1)

        exc_msg = "Feature vectors detected. Smoothing can only be applied to data with spatial dimensions."
        with pytest.raises(ValueError, match=exc_msg):
            spatial_smoothing(test_input)

    def test_channel_index_error(self):
        exc_msg = "Data channel must be an integer equal to 1 or 3. The batch dimension is not a valid channel."
        with pytest.raises(ValueError, match=exc_msg):
            SpatialSmoothing(channel_index=0)

    def test_window_size_error(self):
        exc_msg = "Sliding window size must be a positive integer."
        with pytest.raises(ValueError, match=exc_msg):
            SpatialSmoothing(window_size=0)

    def test_triple_clip_values_error(self):
        exc_msg = "'clip_values' should be a tuple of 2 floats or arrays containing the allowed data range."
        with pytest.raises(ValueError, match=exc_msg):
            SpatialSmoothing(clip_values=(0, 1, 2))

    def test_relation_clip_values_error(self):
        exc_msg = "Invalid 'clip_values': min >= max."
        with pytest.raises(ValueError, match=exc_msg):
            SpatialSmoothing(clip_values=(1, 0))


if __name__ == "__main__":
    pytest.cmdline.main("-q -s {} --mlFramework=tensorflow --durations=0".format(__file__).split(" "))
