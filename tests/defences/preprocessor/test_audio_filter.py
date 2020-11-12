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

from art.defences.preprocessor import AudioFilter
from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.mark.framework_agnostic
@pytest.mark.parametrize("fir_filter", [False, True])
def test_audio_filter(fir_filter, art_warning, expected_values):
    try:
        # Load data for testing
        expected_data = expected_values()

        x1 = expected_data[0]
        x2 = expected_data[1]
        x3 = expected_data[2]
        result_0 = expected_data[3]
        result_1 = expected_data[4]
        result_2 = expected_data[5]

        # Create signal data
        x = np.array([np.array(x1 * 2), np.array(x2 * 2), np.array(x3 * 2)])

        # Filter params
        numerator_coef = np.array([0.1, 0.2, -0.1, -0.2])

        if fir_filter:
            denumerator_coef = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            denumerator_coef = np.array([1.0, 0.1, 0.3, 0.4])

        # Create filter
        audio_filter = AudioFilter(numerator_coef=numerator_coef, denumerator_coef=denumerator_coef)

        # Apply filter
        result = audio_filter(x)

        # Test
        assert result[1] is None
        np.testing.assert_array_almost_equal(result_0, result[0][0], decimal=0)
        np.testing.assert_array_almost_equal(result_1, result[0][1], decimal=0)
        np.testing.assert_array_almost_equal(result_2, result[0][2], decimal=0)

    except ARTTestException as e:
        art_warning(e)
