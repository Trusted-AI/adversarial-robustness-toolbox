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
from art.defences.preprocessor import Cutout
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


@pytest.mark.framework_agnostic
@pytest.mark.parametrize("length", [4, 5])
@pytest.mark.parametrize("channels_first", [True, False])
def test_cutout_image_data(art_warning, image_batch, length, channels_first):
    try:
        cutout = Cutout(length=length, channels_first=channels_first)
        count = np.not_equal(cutout(image_batch)[0], image_batch).sum()

        n = image_batch.shape[0]
        if channels_first:
            channels = image_batch.shape[1]
        else:
            channels = image_batch.shape[-1]
        assert count <= n * channels * length * length
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
@pytest.mark.parametrize("length", [4])
@pytest.mark.parametrize("channels_first", [True, False])
def test_cutout_video_data(art_warning, video_batch, length, channels_first):
    try:
        cutout = Cutout(length=length, channels_first=channels_first)
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


@pytest.mark.framework_agnostic
@pytest.mark.parametrize("length", [4])
@pytest.mark.parametrize("channels_first", [True])
def test_cutout_empty_data(art_warning, empty_image, length, channels_first):
    try:
        cutout = Cutout(length=length, channels_first=channels_first)
        assert_array_equal(cutout(empty_image)[0], empty_image)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_non_image_data_error(art_warning, tabular_batch):
    try:
        test_input = tabular_batch
        cutout = Cutout(length=4, channels_first=True)

        exc_msg = "Unrecognized input dimension. Cutout can only be applied to image and video data."
        with pytest.raises(ValueError, match=exc_msg):
            cutout(test_input)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_check_params(art_warning):
    try:
        with pytest.raises(ValueError):
            _ = Cutout(length=-1)

        with pytest.raises(ValueError):
            _ = Cutout(length=0)

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
@pytest.mark.parametrize("channels_first", [False])
def test_cutout_region_location_non_square(art_warning, channels_first, monkeypatch):
    """
    Regression test: on a non-square image the cutout must zero a `length x length`
    box centered at (center_y, center_x) with the height-bounds applied to the height
    axis and the width-bounds applied to the width axis (i.e. axes must not be swapped).
    """
    try:
        # Non-square NHWC image: height=8 rows, width=4 cols, single channel, batch 1.
        height, width, length = 8, 4, 4
        x = np.ones((1, height, width, 1)).astype(ART_NUMPY_DTYPE)

        # Pin the random center: Cutout draws center_y first, then center_x.
        centers = iter([2, 3])  # center_y=2 (row), center_x=3 (col)
        monkeypatch.setattr(np.random, "randint", lambda *a, **k: next(centers))

        cutout = Cutout(length=length, channels_first=channels_first)
        result = cutout(x.copy())[0]

        # Expected zeroed region: rows [center_y - L//2 : center_y + L//2] clipped to height,
        #                         cols [center_x - L//2 : center_x + L//2] clipped to width.
        expected = np.ones((1, height, width, 1)).astype(ART_NUMPY_DTYPE)
        r0, r1 = max(0, 2 - length // 2), min(height, 2 + length // 2)
        c0, c1 = max(0, 3 - length // 2), min(width, 3 + length // 2)
        expected[0, r0:r1, c0:c1, :] = 0

        assert_array_equal(result, expected)
    except ARTTestException as e:
        art_warning(e)
