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
import numpy as np
import pytest


@pytest.fixture
def image_batch(channels_first):
    """
    Image fixture of shape NHWC and NCHW.
    """
    test_input = np.repeat(np.array(range(6)).reshape(6, 1), 24, axis=1).reshape((2, 3, 4, 6))
    if not channels_first:
        test_input = np.transpose(test_input, (0, 2, 3, 1))
    test_output = test_input.copy()
    return test_input, test_output


@pytest.fixture
def image_batch_small():
    """Create image fixture of shape (batch_size, channels, width, height)."""
    return np.zeros((2, 1, 4, 4))


@pytest.fixture
def video_batch(channels_first):
    """
    Video fixture of shape NFHWC and NCFHW.
    """
    test_input = np.repeat(np.array(range(6)).reshape(6, 1), 24, axis=1).reshape((1, 3, 2, 4, 6))
    if not channels_first:
        test_input = np.transpose(test_input, (0, 2, 3, 4, 1))
    test_output = test_input.copy()
    return test_input, test_output


@pytest.fixture
def tabular_batch():
    """
    Create tabular data fixture of shape (batch_size, features).
    """
    return np.zeros((2, 4))
