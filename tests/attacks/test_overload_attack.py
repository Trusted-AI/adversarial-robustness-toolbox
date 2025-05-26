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

from art.attacks.evasion.overload.overload import OverloadPyTorch

from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.mark.only_with_platform("pytorch")
def test_generate(art_warning, get_pytorch_detector_yolo):
    try:
        from io import BytesIO
        from PIL import Image
        import requests
        import torch

        threshold = 0.85

        object_detector = get_pytorch_detector_yolo
        object_detector.set_params(input_shape=(3, 640, 640))

        # Download a sample image
        target = "https://ultralytics.com/images/zidane.jpg"
        response = requests.get(target)
        org_img = np.asarray(Image.open(BytesIO(response.content)).resize((640, 640)))
        x_255 = np.stack([org_img.transpose((2, 0, 1))], axis=0).astype(np.uint8)
        x = x_255.astype(np.float32) / 255.0

        y_pred = object_detector.predict(x=x)

        attack = OverloadPyTorch(
            object_detector, eps=16.0 / 255.0, max_iter=10, num_grid=10, batch_size=1, threshold=threshold
        )

        x_adv = attack.generate(x=x, y=y_pred)

        assert x.shape == x_adv.shape
        assert np.min(x_adv) >= 0.0
        assert np.max(x_adv) <= 1.0

        y_pred_adv = object_detector.predict(x=x_adv)

        scores_list = list(y_pred[0]["scores"])
        scores_list_adv = list(y_pred_adv[0]["scores"])

        scores_list_filtered = [scores_list.index(x) for x in scores_list if x > threshold]
        scores_list_adv_filtered = [scores_list_adv.index(x) for x in scores_list_adv if x > threshold]

        assert len(scores_list_filtered) == 11
        assert len(scores_list_adv_filtered) == 178

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_check_params(art_warning, get_pytorch_detector_yolo):
    try:
        object_detector = get_pytorch_detector_yolo
        object_detector.set_params(input_shape=(3, 640, 640))

        with pytest.raises(ValueError):
            _ = OverloadPyTorch(
                estimator=object_detector, eps=-1.0, max_iter=5, num_grid=10, batch_size=1, threshold=0.5
            )
        with pytest.raises(ValueError):
            _ = OverloadPyTorch(
                estimator=object_detector, eps=2.0, max_iter=5, num_grid=10, batch_size=1, threshold=0.5
            )
        with pytest.raises(TypeError):
            _ = OverloadPyTorch(
                estimator=object_detector, eps=8 / 255.0, max_iter=1.0, num_grid=10, batch_size=1, threshold=0.5
            )
        with pytest.raises(ValueError):
            _ = OverloadPyTorch(
                estimator=object_detector, eps=8 / 255.0, max_iter=0, num_grid=10, batch_size=1, threshold=0.5
            )
        with pytest.raises(TypeError):
            _ = OverloadPyTorch(
                estimator=object_detector, eps=8 / 255.0, max_iter=5, num_grid=1.0, batch_size=1, threshold=0.5
            )
        with pytest.raises(ValueError):
            _ = OverloadPyTorch(
                estimator=object_detector, eps=8 / 255.0, max_iter=5, num_grid=0, batch_size=1, threshold=0.5
            )
        with pytest.raises(TypeError):
            _ = OverloadPyTorch(
                estimator=object_detector, eps=8 / 255.0, max_iter=5, num_grid=10, batch_size=1.0, threshold=0.5
            )
        with pytest.raises(ValueError):
            _ = OverloadPyTorch(
                estimator=object_detector, eps=8 / 255.0, max_iter=5, num_grid=0, batch_size=0, threshold=0.5
            )
        with pytest.raises(ValueError):
            _ = OverloadPyTorch(
                estimator=object_detector, eps=8 / 255.0, max_iter=5, num_grid=0, batch_size=1, threshold=1.5
            )

    except ARTTestException as e:
        art_warning(e)
