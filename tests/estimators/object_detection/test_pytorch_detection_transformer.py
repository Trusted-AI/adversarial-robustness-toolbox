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

logger = logging.getLogger(__name__)


@pytest.fixture()
@pytest.mark.skip_framework("tensorflow", "tensorflow2v1", "keras", "kerastf", "mxnet", "non_dl_frameworks")
def get_pytorch_detr():
    from art.utils import load_dataset
    from art.estimators.object_detection.pytorch_detection_transformer import PyTorchDetectionTransformer

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    INPUT_SHAPE = (3, 32, 32)

    object_detector = PyTorchDetectionTransformer(
        input_shape=INPUT_SHAPE, clip_values=(0, 1), preprocessing=(MEAN, STD)
    )

    n_test = 2
    (_, _), (x_test, y_test), _, _ = load_dataset("cifar10")
    x_test = x_test.transpose(0, 3, 1, 2).astype(np.float32)
    x_test = x_test[:n_test]

    # Create labels

    result = object_detector.predict(x=x_test)

    y_test = [
        {
            "boxes": result[0]["boxes"],
            "labels": result[0]["labels"],
            "scores": np.ones_like(result[0]["labels"]),
        },
        {
            "boxes": result[1]["boxes"],
            "labels": result[1]["labels"],
            "scores": np.ones_like(result[1]["labels"]),
        },
    ]

    yield object_detector, x_test, y_test


@pytest.mark.only_with_platform("pytorch")
def test_predict(get_pytorch_detr):

    object_detector, x_test, _ = get_pytorch_detr

    result = object_detector.predict(x=x_test)

    assert list(result[0].keys()) == ["boxes", "labels", "scores"]

    assert result[0]["boxes"].shape == (100, 4)
    expected_detection_boxes = np.asarray([-5.9490204e-03, 1.1947733e01, 3.1993944e01, 3.1925127e01])
    np.testing.assert_array_almost_equal(result[0]["boxes"][2, :], expected_detection_boxes, decimal=1)

    assert result[0]["scores"].shape == (100,)
    expected_detection_scores = np.asarray(
        [
            0.00679839,
            0.0250559,
            0.07205943,
            0.01115368,
            0.03321039,
            0.10407761,
            0.00113309,
            0.01442852,
            0.00527624,
            0.01240906,
        ]
    )
    np.testing.assert_array_almost_equal(result[0]["scores"][:10], expected_detection_scores, decimal=1)

    assert result[0]["labels"].shape == (100,)
    expected_detection_classes = np.asarray([17, 17, 33, 17, 17, 17, 74, 17, 17, 17])
    np.testing.assert_array_almost_equal(result[0]["labels"][:10], expected_detection_classes, decimal=5)


@pytest.mark.only_with_platform("pytorch")
def test_loss_gradient(get_pytorch_detr):

    object_detector, x_test, y_test = get_pytorch_detr

    grads = object_detector.loss_gradient(x=x_test, y=y_test)

    assert grads.shape == (2, 3, 800, 800)

    expected_gradients1 = np.asarray(
        [
            -0.00061366,
            0.00322502,
            -0.00039866,
            -0.00807413,
            -0.00476555,
            0.00181204,
            0.01007765,
            0.00415828,
            -0.00073114,
            0.00018387,
            -0.00146992,
            -0.00119636,
            -0.00098966,
            -0.00295517,
            -0.0024271,
            -0.00131314,
            -0.00149217,
            -0.00104926,
            -0.00154239,
            -0.00110989,
            0.00092887,
            0.00049146,
            -0.00292508,
            -0.00124526,
            0.00140347,
            0.00019833,
            0.00191074,
            -0.00117537,
            -0.00080604,
            0.00057427,
            -0.00061728,
            -0.00206535,
        ]
    )

    np.testing.assert_array_almost_equal(grads[0, 0, 10, :32], expected_gradients1, decimal=2)

    expected_gradients2 = np.asarray(
        [
            -1.1787530e-03,
            -2.8500680e-03,
            5.0884970e-03,
            6.4504531e-04,
            -6.8841036e-05,
            2.8184296e-03,
            3.0257765e-03,
            2.8565727e-04,
            -1.0701057e-04,
            1.2945699e-03,
            7.3593057e-04,
            1.0177144e-03,
            -2.4692707e-03,
            -1.3801848e-03,
            6.3182280e-04,
            -4.2305476e-04,
            4.4307750e-04,
            8.5821096e-04,
            -7.1204413e-04,
            -3.1404425e-03,
            -1.5964351e-03,
            -1.9222996e-03,
            -5.3157361e-04,
            -9.9202688e-04,
            -1.5815455e-03,
            2.0060266e-04,
            -2.0584739e-03,
            6.6960667e-04,
            9.7393827e-04,
            -1.6040013e-03,
            -6.9741381e-04,
            1.4657658e-04,
        ]
    )
    np.testing.assert_array_almost_equal(grads[1, 0, 10, :32], expected_gradients2, decimal=2)


@pytest.mark.only_with_platform("pytorch")
def test_errors():

    from torch import hub

    from art.estimators.object_detection.pytorch_detection_transformer import PyTorchDetectionTransformer

    model = hub.load("facebookresearch/detr", "detr_resnet50", pretrained=True)

    with pytest.raises(ValueError):
        PyTorchDetectionTransformer(
            model=model,
            clip_values=(1, 2),
            attack_losses=("loss_ce", "loss_bbox", "loss_giou"),
        )

    with pytest.raises(ValueError):
        PyTorchDetectionTransformer(
            model=model,
            clip_values=(-1, 1),
            attack_losses=("loss_ce", "loss_bbox", "loss_giou"),
        )

    from art.defences.postprocessor.rounded import Rounded

    post_def = Rounded()
    with pytest.raises(ValueError):
        PyTorchDetectionTransformer(
            model=model,
            clip_values=(0, 1),
            attack_losses=("loss_ce", "loss_bbox", "loss_giou"),
            postprocessing_defences=post_def,
        )


@pytest.mark.only_with_platform("pytorch")
def test_preprocessing_defences(get_pytorch_detr):

    object_detector, x_test, _ = get_pytorch_detr

    from art.defences.preprocessor.spatial_smoothing_pytorch import SpatialSmoothingPyTorch

    pre_def = SpatialSmoothingPyTorch()

    object_detector.set_params(preprocessing_defences=pre_def)

    # Create labels
    result = object_detector.predict(x=x_test)

    y = [
        {
            "boxes": result[0]["boxes"],
            "labels": result[0]["labels"],
            "scores": np.ones_like(result[0]["labels"]),
        },
        {
            "boxes": result[1]["boxes"],
            "labels": result[1]["labels"],
            "scores": np.ones_like(result[1]["labels"]),
        },
    ]

    # Compute gradients
    grads = object_detector.loss_gradient(x=x_test, y=y)

    assert grads.shape == (2, 3, 800, 800)


@pytest.mark.only_with_platform("pytorch")
def test_compute_losses(get_pytorch_detr):

    object_detector, x_test, y_test = get_pytorch_detr
    object_detector.attack_losses = "loss_ce"
    losses = object_detector.compute_losses(x=x_test, y=y_test)
    assert len(losses) == 1


@pytest.mark.only_with_platform("pytorch")
def test_compute_loss(get_pytorch_detr):

    object_detector, x_test, _ = get_pytorch_detr
    # Create labels
    result = object_detector.predict(x_test)

    y = [
        {
            "boxes": result[0]["boxes"],
            "labels": result[0]["labels"],
            "scores": np.ones_like(result[0]["labels"]),
        },
        {
            "boxes": result[1]["boxes"],
            "labels": result[1]["labels"],
            "scores": np.ones_like(result[1]["labels"]),
        },
    ]

    # Compute loss
    loss = object_detector.compute_loss(x=x_test, y=y)

    assert pytest.approx(3.9634, abs=0.01) == float(loss)


@pytest.mark.only_with_platform("pytorch")
def test_pgd(get_pytorch_detr):

    object_detector, x_test, y_test = get_pytorch_detr

    from art.attacks.evasion import ProjectedGradientDescent
    from PIL import Image

    imgs = []
    for i in x_test:
        img = Image.fromarray((i * 255).astype(np.uint8).transpose(1, 2, 0))
        img = img.resize(size=(800, 800))
        imgs.append(np.array(img))
    x_test = np.array(imgs).transpose(0, 3, 1, 2)

    attack = ProjectedGradientDescent(estimator=object_detector, max_iter=2)
    x_test_adv = attack.generate(x=x_test, y=y_test)
    np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, x_test_adv, x_test)
