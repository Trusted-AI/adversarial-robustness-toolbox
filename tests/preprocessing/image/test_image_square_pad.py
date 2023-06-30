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
import pytest

from art.config import ART_NUMPY_DTYPE
from art.preprocessing.image import ImageSquarePad, ImageSquarePadPyTorch, ImageSquarePadTensorFlowV2
from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.fixture()
def image_batch(height, width, channels_first, label_type):
    channels = 3

    if channels_first:
        data_shape = (2, channels, height, width)
    else:
        data_shape = (2, height, width, channels)
    x = (0.5 * np.ones(data_shape)).astype(ART_NUMPY_DTYPE)

    if label_type == "classification":
        y = np.arange(len(x))
    elif label_type == "object_detection":
        y = []
        for _ in range(len(x)):
            y_1, x_1 = np.random.uniform(0, (height / 2, width / 2))
            y_2, x_2 = np.random.uniform((y_1 + 3, x_1 + 3), (height, width))
            target_dict = {
                "boxes": np.array([[x_1, y_1, x_2, y_2]]),
                "labels": np.array([0]),
            }
            y.append(target_dict)
    else:
        y = None

    return x, y


@pytest.mark.framework_agnostic
@pytest.mark.parametrize("height", [16, 20, 24])
@pytest.mark.parametrize("width", [16, 20, 24])
@pytest.mark.parametrize("channels_first", [True, False])
@pytest.mark.parametrize("label_type", [None, "classification", "object_detection"])
@pytest.mark.parametrize("pad_mode", ["constant", "reflect"])
def test_square_pad_numpy(height, width, channels_first, label_type, pad_mode, image_batch, art_warning):
    x, y = image_batch
    length = max(height, width)

    try:
        resize = ImageSquarePad(
            channels_first=channels_first,
            label_type=label_type,
            pad_mode=pad_mode,
        )

        x_preprocess, y_preprocess = resize(x, y)

        if channels_first:
            assert x_preprocess.shape == (x.shape[0], x.shape[1], length, length)
        else:
            assert x_preprocess.shape == (x.shape[0], length, length, x.shape[3])

        if label_type == "classification":
            np.testing.assert_array_equal(y, y_preprocess)
        elif label_type == "object_detection":
            assert y[0]["boxes"].shape == y_preprocess[0]["boxes"].shape
            np.testing.assert_array_equal(y[0]["labels"], y_preprocess[0]["labels"])
        else:
            assert y is None

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
@pytest.mark.parametrize("height", [16, 20, 24])
@pytest.mark.parametrize("width", [16, 20, 24])
@pytest.mark.parametrize("channels_first", [True, False])
@pytest.mark.parametrize("label_type", [None, "classification", "object_detection"])
@pytest.mark.parametrize("pad_mode", ["constant", "reflect"])
def test_square_pad_pytorch(height, width, channels_first, label_type, pad_mode, image_batch, art_warning):
    import torch

    x, y = image_batch
    length = max(height, width)

    x = torch.from_numpy(x)
    if label_type == "classification":
        y = torch.from_numpy(y)
    elif label_type == "object_detection":
        y = [{k: torch.from_numpy(v) for k, v in y_i.items()} for y_i in y]

    try:
        resize = ImageSquarePadPyTorch(
            channels_first=channels_first,
            label_type=label_type,
            pad_mode=pad_mode,
        )

        x_preprocess, y_preprocess = resize.forward(x, y)

        if channels_first:
            assert x_preprocess.shape == (x.shape[0], x.shape[1], length, length)
        else:
            assert x_preprocess.shape == (x.shape[0], length, length, x.shape[3])

        if label_type == "classification":
            np.testing.assert_array_equal(y, y_preprocess)
        elif label_type == "object_detection":
            assert y[0]["boxes"].shape == y_preprocess[0]["boxes"].shape
            np.testing.assert_array_equal(y[0]["labels"], y_preprocess[0]["labels"])
        else:
            assert y is None

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("tensorflow2")
@pytest.mark.parametrize("height", [16, 20, 24])
@pytest.mark.parametrize("width", [16, 20, 24])
@pytest.mark.parametrize("channels_first", [True, False])
@pytest.mark.parametrize("label_type", [None, "classification", "object_detection"])
@pytest.mark.parametrize("pad_mode", ["CONSTANT", "REFLECT"])
def test_square_pad_tensorflow(height, width, channels_first, label_type, pad_mode, image_batch, art_warning):
    import tensorflow as tf

    x, y = image_batch
    length = max(height, width)

    x = tf.convert_to_tensor(x)
    if label_type == "classification":
        y = tf.convert_to_tensor(y)
    elif label_type == "object_detection":
        y = [{k: tf.convert_to_tensor(v) for k, v in y_i.items()} for y_i in y]

    try:
        resize = ImageSquarePadTensorFlowV2(
            channels_first=channels_first,
            label_type=label_type,
            pad_mode=pad_mode,
        )

        x_preprocess, y_preprocess = resize.forward(x, y)

        if channels_first:
            assert x_preprocess.shape == (x.shape[0], x.shape[1], length, length)
        else:
            assert x_preprocess.shape == (x.shape[0], length, length, x.shape[3])

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
            _ = ImageSquarePad(pad_mode="constant", clip_values=(0,))

        with pytest.raises(ValueError):
            _ = ImageSquarePad(pad_mode="constant", clip_values=(1, 0))

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_check_params_pytorch(art_warning):
    try:
        with pytest.raises(ValueError):
            _ = ImageSquarePadPyTorch(pad_mode="constant", clip_values=(0,))

        with pytest.raises(ValueError):
            _ = ImageSquarePadPyTorch(pad_mode="constant", clip_values=(1, 0))

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("tensorflow2")
def test_check_params_tensorflow(art_warning):
    try:
        with pytest.raises(ValueError):
            _ = ImageSquarePadTensorFlowV2(pad_mode="constant", clip_values=(0,))

        with pytest.raises(ValueError):
            _ = ImageSquarePadTensorFlowV2(pad_mode="constant", clip_values=(1, 0))

    except ARTTestException as e:
        art_warning(e)
