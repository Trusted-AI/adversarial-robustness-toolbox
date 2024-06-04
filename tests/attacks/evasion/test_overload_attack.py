# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2024
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

from art.attacks.evasion import OverloadPyTorch
from art.estimators.object_detection import PyTorchYolo
from tests.utils import ARTTestException

logger = logging.getLogger(__name__)

@pytest.mark.only_with_platform("pytorch")
def test_generate(art_warning):
    try:
        import torch
        model = torch.hub.load('ultralytics/yolov5:v7.0',  model='yolov5s')
        py_model = PyTorchYolo(model=model,
                               input_shape=(3, 640, 640),
                               channels_first=True)
        x = np.random.uniform(0.0, 1.0, size=(10, 3, 640, 640)).astype(np.float32)

        attack = OverloadPyTorch(py_model,
                                 eps = 16.0 / 255.0,
                                 max_iter = 5,
                                 num_grid = 10,
                                 batch_size = 1)

        x_adv = attack.generate(x)

        assert x.shape == x_adv.shape
        assert np.min(x_adv) >= 0.0
        assert np.max(x_adv) <= 1.0

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_check_params(art_warning):
    try:
        import torch
        model = torch.hub.load('ultralytics/yolov5:v7.0',  model='yolov5s')
        py_model = PyTorchYolo(model=model,
                               input_shape=(3, 640, 640),
                               channels_first=True)

        with pytest.raises(ValueError):
           _ = OverloadPyTorch(py_model, -1.0, 5, 10, 1)
        with pytest.raises(ValueError):
           _ = OverloadPyTorch(py_model, 2.0, 5, 10, 1)
        with pytest.raises(TypeError):
           _ = OverloadPyTorch(py_model, 8 / 255.0, 1.0, 10, 1)
        with pytest.raises(ValueError):
           _ = OverloadPyTorch(py_model, 8 / 255.0, 0, 10, 1)
        with pytest.raises(TypeError):
           _ = OverloadPyTorch(py_model, 8 / 255.0, 5, 1.0, 1)
        with pytest.raises(ValueError):
           _ = OverloadPyTorch(py_model, 8 / 255.0, 5, 0, 1)
        with pytest.raises(TypeError):
           _ = OverloadPyTorch(py_model, 8 / 255.0, 5, 10, 1.0)
        with pytest.raises(ValueError):
           _ = OverloadPyTorch(py_model, 8 / 255.0, 5, 0, 0)

    except ARTTestException as e:
        art_warning(e)
