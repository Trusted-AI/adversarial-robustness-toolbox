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
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np
import pytest

from art.config import ART_NUMPY_DTYPE

logger = logging.getLogger(__name__)


@pytest.fixture()
def image_batch(channels_first):
    """
    Image fixtures of shape NHWC and NCHW and a sample object detection label.
    """
    channels = 3
    height = 20
    width = 16

    if channels_first:
        data_shape = (2, channels, height, width)
    else:
        data_shape = (2, height, width, channels)
    x = (0.5 * np.ones((data_shape))).astype(ART_NUMPY_DTYPE)

    y = []
    for _ in range(len(x)):
        y_1, x_1 = np.random.uniform(0, (height / 2, width / 2))
        y_2, x_2 = np.random.uniform((y_1 + 3, x_1 + 3), (height, width))
        target_dict = {
            "boxes": np.array([[x_1, y_1, x_2, y_2]]),
            "labels": np.array([0]),
        }
        y.append(target_dict)

    return x, y


@pytest.fixture
def tabular_batch():
    """
    Create tabular data fixture of shape (batch_size, features).
    """
    return np.zeros((2, 4)), []
