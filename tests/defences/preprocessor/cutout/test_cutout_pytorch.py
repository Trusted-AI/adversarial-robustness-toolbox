# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2022
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

from art.config import ART_NUMPY_DTYPE
from art.defences.preprocessor import CutoutPyTorch
from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.fixture(params=[1, 3], ids=["grayscale", "RGB"])
def image_batch(request, channels_first):
    """
    Image fixtures of shape NHWC and NCHW.
    """
    channels = request.param

    if channels_first:
        data_shape = (2, channels, 12, 8)
    else:
        data_shape = (2, 12, 8, channels)
    return (255 * np.ones(data_shape)).astype(ART_NUMPY_DTYPE)


@pytest.fixture(params=[1, 3], ids=["grayscale", "RGB"])
def video_batch(request, channels_first):
    """
    Video fixtures of shape NFHWC and NCFHW.
    """
    channels = request.param

    if channels_first:
        data_shape = (2, 2, channels, 12, 8)
    else:
        data_shape = (2, 2, 12, 8, channels)
    return (255 * np.ones(data_shape)).astype(ART_NUMPY_DTYPE)


@pytest.fixture(params=[1, 3], ids=["grayscale", "RGB"])
def empty_image(request, channels_first):
    """
    Empty image fixtures of shape NHWC and NCHW.
    """
    channels = request.param

    if channels_first:
        data_shape = (2, channels, 12, 8)
    else:
        data_shape = (2, 12, 8, channels)
    return np.zeros(data_shape).astype(ART_NUMPY_DTYPE)


@pytest.mark.only_with_platform("pytorch")
@pytest.mark.parametrize("length", [2, 4])
@pytest.mark.parametrize("channels_first", [True, False])
def test_cutout_image_data(art_warning, image_batch, length, channels_first):
    try:
        cutout = CutoutPyTorch(length=length, channels_first=channels_first)
        count = np.not_equal(cutout(image_batch)[0], image_batch).sum()

        n = image_batch.shape[0]
        if channels_first:
            channels = image_batch.shape[1]
        else:
            channels = image_batch.shape[-1]
        assert count <= n * channels * length * length
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
@pytest.mark.parametrize("length", [4])
@pytest.mark.parametrize("channels_first", [True, False])
def test_cutout_video_data(art_warning, video_batch, length, channels_first):
    try:
        cutout = CutoutPyTorch(length=length, channels_first=channels_first)
        count = np.not_equal(cutout(video_batch)[0], video_batch).sum()

        n = video_batch.shape[0]
        frames = video_batch.shape[1]
        if channels_first:
            channels = video_batch.shape[2]
        else:
            channels = video_batch.shape[-1]
        assert count <= n * frames * channels * length * length
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
@pytest.mark.parametrize("length", [4])
@pytest.mark.parametrize("channels_first", [True])
def test_cutout_empty_data(art_warning, empty_image, length, channels_first):
    try:
        cutout = CutoutPyTorch(length=length, channels_first=channels_first)
        assert_array_equal(cutout(empty_image)[0], empty_image)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_non_image_data_error(art_warning, tabular_batch):
    try:
        test_input = tabular_batch
        cutout = CutoutPyTorch(length=8, channels_first=True)

        exc_msg = "Unrecognized input dimension. Cutout can only be applied to image and video data."
        with pytest.raises(ValueError, match=exc_msg):
            cutout(test_input)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_check_params(art_warning):
    try:
        with pytest.raises(ValueError):
            _ = CutoutPyTorch(length=-1)

        with pytest.raises(ValueError):
            _ = CutoutPyTorch(length=0)

    except ARTTestException as e:
        art_warning(e)
