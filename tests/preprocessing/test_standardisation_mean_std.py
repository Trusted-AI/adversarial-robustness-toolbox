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
    StandardisationMeanStdTensorFlow,
)
from art.preprocessing.standardisation_mean_std.utils import broadcastable_mean_std
from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.fixture(params=[True, False], ids=["channels_first", "channels_last"])
def image_batch(request):
    """
    Create image fixture of shape NFHWC and NCFHW.
    """
    channels_first = request.param
    test_input = np.ones((2, 3, 32, 32))
    if not channels_first:
        test_input = np.transpose(test_input, (0, 2, 3, 1))
    test_mean = [0] * 3
    test_std = [1] * 3
    test_output = test_input.copy()
    return test_input, test_output, test_mean, test_std


@pytest.mark.framework_agnostic
def test_broadcastable_mean_std(art_warning):
    try:
        mean, std = broadcastable_mean_std(np.ones((1, 3, 20, 20)), np.ones(3), np.ones(3))
        assert mean.shape == std.shape == (1, 3, 1, 1)

        mean, std = broadcastable_mean_std(np.ones((1, 3, 20, 20)), np.ones((1, 3, 1, 1)), np.ones((1, 3, 1, 1)))
        assert mean.shape == std.shape == (1, 3, 1, 1)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_standardisation_mean_std(art_warning, image_batch):
    try:
        x, x_expected, mean, std = image_batch
        standard = StandardisationMeanStd(mean=mean, std=std)

        x_preprocessed, _ = standard(x=x, y=None)
        np.testing.assert_array_equal(x_preprocessed, x_expected)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_standardisation_mean_std_pytorch(art_warning, image_batch):
    try:
        x, x_expected, mean, std = image_batch
        standard = StandardisationMeanStdPyTorch(mean=mean, std=std)

        x_preprocessed, _ = standard(x=x, y=None)
        np.testing.assert_array_equal(x_preprocessed, x_expected)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("tensorflow2")
def test_standardisation_mean_std_tensorflow_v2(art_warning, image_batch):
    try:
        x, x_expected, mean, std = image_batch
        standard = StandardisationMeanStdTensorFlow(mean=mean, std=std)

        x_preprocessed, _ = standard(x=x, y=None)
        np.testing.assert_array_equal(x_preprocessed, x_expected)
    except ARTTestException as e:
        art_warning(e)
