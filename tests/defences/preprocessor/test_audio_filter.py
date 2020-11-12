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
def test_audio_filter(fir_filter, art_warning, store_expected_values):
    try:

        # Small data for testing
        x1 = [
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
        #    * 2


        x2 = [
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
        #    * 2


        x3 = [
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
        #    * 2


        #test_input = np.array([x1, x2, x3])

        numerator_coef = np.array([0.1, 0.2, -0.1, -0.2])

        if fir_filter:
            denumerator_coef = np.array([1.0, 0.0, 0.0, 0.0])
            result_0 = [-1.0376293e-04, -3.1434064e-04, -2.1973327e-04, -1.8311101e-05,
 -1.5259249e-05, -6.1036999e-06, -6.1037099e-06, -1.5259280e-05,
 -6.1037199e-06,  3.0518599e-06,  1.2207430e-05,  1.8311121e-05,
  3.3570352e-05,  4.5777753e-05, -9.1555521e-06, -3.6622205e-05,
 -1.8311101e-05, -1.5259249e-05, -6.1036999e-06, -6.1037099e-06,
 -1.5259280e-05, -6.1037199e-06,  3.0518599e-06,  1.2207430e-05,
  1.8311121e-05,  3.3570352e-05]

            result_1 = [-1.8311106e-05, -4.8829617e-05, -1.2207404e-05,  3.6622212e-05,
  3.3570363e-05,  1.8311106e-05, -6.1037017e-06, -1.2207403e-05,
  3.0518509e-06,  1.5259255e-05,  3.9674062e-05,  6.7140718e-05,
  7.0192567e-05,  5.4933316e-05, -3.6622212e-05, -1.8616291e-04,
 -1.2207404e-04,  2.4414809e-05,  3.6622212e-05,  3.3570363e-05,
  1.8311106e-05, -6.1037017e-06, -1.2207403e-05,  3.0518509e-06,
  1.5259255e-05,  3.9674062e-05,  6.7140718e-05,  7.0192567e-05,
  5.4933316e-05, -3.6622212e-05]

            result_2 = [-8.2399973e-05, -2.3499252e-04, -1.1291848e-04,  8.2399980e-05,
  7.3244424e-05,  5.1881467e-05,  4.5777761e-05,  1.1291848e-04,
  2.4414808e-04,  1.6479995e-04,  2.1362957e-05,  2.7466658e-05,
  1.8311106e-05, -8.5451829e-05, -3.2044435e-04, -3.5401472e-04,
 -8.2399973e-05,  8.2399980e-05,  7.3244424e-05,  5.1881467e-05,
  4.5777761e-05,  1.1291848e-04,  2.4414808e-04,  1.6479995e-04,
  2.1362957e-05,  2.7466658e-05,  1.8311106e-05, -8.5451829e-05]
        else:
            denumerator_coef = np.array([1.0, 0.1, 0.3, 0.4])

            result_0 = [-1.03762934e-04, - 3.03964334e-04, - 1.58207942e-04,
            1.30204164e-04,
            1.40768461e-04,
            4.04138154e-06, - 1.00820056e-04, - 6.26970723e-05,
            2.87954499e-05,
            5.93094592e-05,
            2.27166784e-05, - 1.32715650e-05,
            4.35872198e-06,
            4.02366786e-05, - 9.17821035e-06, - 4.95188760e-05,
            - 2.67004216e-05,
            5.93773893e-06,
            2.11202023e-05,
            6.83116525e-07,
            - 2.40387490e-05, - 1.23528616e-05,
            1.12255238e-05,
            2.44062358e-05,
            1.74439847e-05,
            2.00138729e-05]
            result_1 = [-1.8311106e-05, - 4.6998506e-05, - 2.0142215e-06,
            5.8247628e-05,
            4.7149268e-05, - 3.0724209e-06, - 4.3240292e-05, - 2.5821355e-05,
            1.9835043e-05,
            3.8318274e-05,
            4.0220264e-05,
            4.3689197e-05,
            3.8430262e-05,
            2.1895425e-05, - 6.7816509e-05, - 2.0132199e-04,
            - 9.0355054e-05,
            1.2097351e-04,
            1.3216017e-04,
            2.0204312e-05,
            - 7.1746785e-05, - 5.7854388e-05,
            7.0203455e-06,
            4.8404847e-05,
            3.1454420e-05,
            1.9199029e-05,
            3.6422553e-05,
            4.8208840e-05,
            3.1506053e-05, - 6.8804489e-05]
            result_2 = [-8.23999726e-05, - 2.26752527e-04, - 6.55232361e-05,
            1.89938044e-04,
            1.64608602e-04,
            4.64848699e-06, - 8.00448834e-05,
            5.36849875e-05,
            2.60933652e-04,
            1.54619047e-04, - 9.38530357e-05, - 1.13907212e-04,
            - 3.98988004e-06, - 1.33394606e-05, - 2.72350560e-04, - 3.21181869e-04,
            3.67591638e-05,
            2.84018839e-04,
            1.62287542e-04, - 6.42566083e-05,
            - 1.10090376e-04,
            7.82894858e-05,
            2.95048871e-04,
            1.55844362e-04,
            - 1.14051938e-04, - 1.25901017e-04,
            2.77904201e-06, - 2.33865194e-06]
        store_expected_values((x1, x2, x3, result_0, result_1, result_2))
        #store_expected_values(x1)
        audio_filter = AudioFilter(numerator_coef=numerator_coef, denumerator_coef=denumerator_coef)
        result = audio_filter(test_input)
#        print(fir_filter, result[0][0] )
#        print(fir_filter, result[0][1])
#        print(fir_filter, result[0][2])

#        assert_array_equal(spatial_smoothing(test_input)[0], test_output)
        #print(result_0, result[0][0])
        assert result[1] is None
        np.testing.assert_array_almost_equal(result_0, result[0][0], decimal=0)
        np.testing.assert_array_almost_equal(result_1, result[0][1], decimal=0)
        np.testing.assert_array_almost_equal(result_2, result[0][2], decimal=0)
#        assert False

    except ARTTestException as e:
        art_warning(e)


