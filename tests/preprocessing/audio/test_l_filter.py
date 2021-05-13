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

from art.preprocessing.audio import LFilter
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
        x = np.array([np.array(x1 * 2), np.array(x2 * 2), np.array(x3 * 2)], dtype=object)

        # Filter params
        numerator_coef = np.array([0.1, 0.2, -0.1, -0.2])

        if fir_filter:
            denominator_coef = np.array([1.0])
        else:
            denominator_coef = np.array([1.0, 0.1, 0.3, 0.4])

        # Create filter
        audio_filter = LFilter(numerator_coef=numerator_coef, denominator_coef=denominator_coef)

        # Apply filter
        result = audio_filter(x)

        # Test
        assert result[1] is None
        np.testing.assert_array_almost_equal(result_0, result[0][0], decimal=0)
        np.testing.assert_array_almost_equal(result_1, result[0][1], decimal=0)
        np.testing.assert_array_almost_equal(result_2, result[0][2], decimal=0)

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_default(art_warning):
    try:
        # Small data for testing
        x = np.array([[0.37, 0.68, 0.63, 0.48, 0.48, 0.18, 0.19]])

        # Create filter
        audio_filter = LFilter()

        # Apply filter
        result = audio_filter(x)

        # Test
        assert result[1] is None
        np.testing.assert_array_almost_equal(x, result[0], decimal=0)

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_triple_clip_values_error(art_warning):
    try:
        exc_msg = "`clip_values` should be a tuple of 2 floats containing the allowed data range."
        with pytest.raises(ValueError, match=exc_msg):
            LFilter(
                numerator_coef=np.array([0.1, 0.2, 0.3]),
                denominator_coef=np.array([0.1, 0.2, 0.3]),
                clip_values=(0, 1, 2),
            )

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_relation_clip_values_error(art_warning):
    try:
        exc_msg = "Invalid `clip_values`: min >= max."
        with pytest.raises(ValueError, match=exc_msg):
            LFilter(
                numerator_coef=np.array([0.1, 0.2, 0.3]), denominator_coef=np.array([0.1, 0.2, 0.3]), clip_values=(1, 0)
            )

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
@pytest.mark.parametrize("fir_filter", [False, True])
def test_estimate_gradient(fir_filter, art_warning, expected_values):

    try:
        # Load data for testing
        expected_data = expected_values()

        x1 = expected_data[0]
        x2 = expected_data[1]
        x3 = expected_data[2]

        grad0 = expected_data[3]
        grad1 = expected_data[4]
        grad2 = expected_data[5]

        result0 = expected_data[6]
        result1 = expected_data[7]
        result2 = expected_data[8]

        # Create signal data
        x = np.array([np.array(x1 * 2), np.array(x2 * 2), np.array(x3 * 2)], dtype=object)

        # Create input gradient
        grad = np.array([np.array(grad0), np.array(grad1), np.array(grad2)], dtype=object)

        # Filter params
        numerator_coef = np.array([0.1, 0.2, -0.1, -0.2])

        if fir_filter:
            denominator_coef = np.array([1.0])
        else:
            denominator_coef = np.array([1.0, 0.1, 0.3, 0.4])

        # Create filter
        audio_filter = LFilter(numerator_coef=numerator_coef, denominator_coef=denominator_coef)

        # Estimate gradient
        estimated_grad = audio_filter.estimate_gradient(x, grad=grad)

        # Test
        assert estimated_grad.shape == x.shape
        np.testing.assert_array_almost_equal(result0, estimated_grad[0], decimal=0)
        np.testing.assert_array_almost_equal(result1, estimated_grad[1], decimal=0)
        np.testing.assert_array_almost_equal(result2, estimated_grad[2], decimal=0)

    except ARTTestException as e:
        art_warning(e)
