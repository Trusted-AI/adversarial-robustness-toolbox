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

import numpy as np
import pytest


@pytest.fixture
def audio_data():
    """
    Create audio fixtures of shape (nb_samples=3,) with elements of variable length.
    """
    sample_rate = 16000
    test_input = np.array(
        [
            np.zeros(sample_rate),
            np.ones(sample_rate * 2) * 2e3,
            np.ones(sample_rate * 3) * 3e3,
            np.ones(sample_rate * 3) * 3e3,
        ],
        dtype=object,
    )
    return test_input


@pytest.fixture
def audio_batch_padded():
    """
    Create audio fixtures of shape (batch_size=2,) with elements of variable length.
    """
    sample_rate = 16000
    frequency_length = (sample_rate // 2 + 1) // 240 * 3
    test_input = np.zeros((2, sample_rate))
    test_mask_frequency = np.ones((2, frequency_length, 80))
    return test_input, test_mask_frequency
