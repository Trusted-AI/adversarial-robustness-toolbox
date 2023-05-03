# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2023
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
import logging

import numpy as np
import cv2
import pytest

from art.preprocessing.image import ImageResize
from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.mark.framework_agnostic
@pytest.mark.parametrize("height", [16, 20, 24])
@pytest.mark.parametrize("width", [16, 20, 24])
@pytest.mark.parametrize("channels_first", [True, False])
@pytest.mark.parametrize("label_type", [None, "classification", "object_detection"])
@pytest.mark.parametrize("interpolation", [cv2.INTER_LINEAR, cv2.INTER_AREA])
def test_resize_numpy(height, width, channels_first, label_type, interpolation, image_batch, art_warning):
    x, y = image_batch

    try:
        resize = ImageResize(
            height=height,
            width=width,
            channels_first=channels_first,
            label_type=label_type,
            interpolation=interpolation,
        )

        x_preprocess, y_preprocess = resize(x, y)

        if channels_first:
            assert x_preprocess.shape == (x.shape[0], x.shape[1], height, width)
        else:
            assert x_preprocess.shape == (x.shape[0], height, width, x.shape[3])

        if label_type == "classification":
            np.testing.assert_array_equal(y, y_preprocess)
        elif label_type == "object_detection":
            assert y[0]["boxes"].shape == y_preprocess[0]["boxes"].shape
            np.testing.assert_array_equal(y[0]["labels"], y_preprocess[0]["labels"])
        else:
            assert y is None

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_check_params_numpy(art_warning):
    try:
        with pytest.raises(ValueError):
            _ = ImageResize(height=0, width=32)

        with pytest.raises(ValueError):
            _ = ImageResize(height=32, width=0)

        with pytest.raises(ValueError):
            _ = ImageResize(height=0, width=0)

        with pytest.raises(ValueError):
            _ = ImageResize(height=32, width=32, clip_values=(0,))

        with pytest.raises(ValueError):
            _ = ImageResize(height=32, width=32, clip_values=(1, 0))

    except ARTTestException as e:
        art_warning(e)
