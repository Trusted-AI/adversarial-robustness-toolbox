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

from art.defences.preprocessor import AudioFilter
from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.mark.framework_agnostic
@pytest.mark.parametrize("fir_filter", [False, True])
def test_audio_filter(fir_filter, art_warning):
    try:

        # Small data for testing
        x1 = np.array(
            [
                -1.0376293e-03,
                -1.0681478e-03,
                -1.0986663e-03,
                -1.1291848e-03,
                -1.1291848e-03,
                -1.1291848e-03,
                -1.1902219e-03,
                -1.1597034e-03,
                -1.1902219e-03,
                -1.1291848e-03,
                -1.1291848e-03,
                -1.0681478e-03,
                -9.1555528e-04,
            ]
            * 100
        )

        x2 = np.array(
            [
                -1.8311106e-04,
                -1.2207404e-04,
                -6.1037019e-05,
                0.0000000e00,
                3.0518509e-05,
                0.0000000e00,
                -3.0518509e-05,
                0.0000000e00,
                0.0000000e00,
                9.1555528e-05,
                2.1362957e-04,
                3.3570360e-04,
                4.2725913e-04,
                4.5777764e-04,
                -1.8311106e-04,
            ]
            * 100
        )

        x3 = np.array(
            [
                -8.2399976e-04,
                -7.0192572e-04,
                -5.4933317e-04,
                -4.2725913e-04,
                -3.6622211e-04,
                -2.7466659e-04,
                -2.1362957e-04,
                5.4933317e-04,
                5.7985168e-04,
                6.1037019e-04,
                6.7140721e-04,
                7.0192572e-04,
                6.7140721e-04,
                -1.5259255e-04,
            ]
            * 100
        )

        test_input = np.array([x1, x2, x3])

        numerator_coef = np.array([0.1, 0.2, -0.1, -0.2])

        if fir_filter:
            denumerator_coef = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            denumerator_coef = np.array([1.0, 0.1, 0.3, 0.4])

#        test_output = np.array([[[[1, 2], [3, 3]]]])
        audio_filter = AudioFilter(numerator_coef=numerator_coef, denumerator_coef=denumerator_coef)
        print(fir_filter, audio_filter(test_input))

#        assert_array_equal(spatial_smoothing(test_input)[0], test_output)

    except ARTTestException as e:
        art_warning(e)


