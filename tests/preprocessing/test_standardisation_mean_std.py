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

from art.preprocessing.standardisation_mean_std import (
    StandardisationMeanStd,
    StandardisationMeanStdPyTorch,
    StandardisationMeanStdTensorFlowV2,
)

from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.mark.framework_agnostic
def test_standardisation_mean_std(art_warning):
    try:
        x = np.ones(shape=(32, 32, 3)) * 5
        mean = 2.5
        std = 1.5
        standard = StandardisationMeanStd(mean=mean, std=std)
        x_preprocessed, _ = standard(x=x, y=None)
        np.testing.assert_array_equal(x_preprocessed, np.ones_like(x_preprocessed) * (5 - mean) / std)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_standardisation_mean_std_pytorch(art_warning):
    try:
        x = np.ones(shape=(32, 32, 3)) * 5
        mean = 2.5
        std = 1.5
        standard = StandardisationMeanStdPyTorch(mean=mean, std=std)
        x_preprocessed, _ = standard(x=x, y=None)
        np.testing.assert_array_equal(x_preprocessed, np.ones_like(x_preprocessed) * (5 - mean) / std)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("tensorflow2")
def test_standardisation_mean_std_tensorflow_v2(art_warning):
    try:
        x = np.ones(shape=(32, 32, 3)) * 5
        mean = 2.5
        std = 1.5
        standard = StandardisationMeanStdTensorFlowV2(mean=mean, std=std)
        x_preprocessed, _ = standard(x=x, y=None)
        np.testing.assert_array_equal(x_preprocessed, np.ones_like(x_preprocessed) * (5 - mean) / std)
    except ARTTestException as e:
        art_warning(e)
