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
    expected_detection_boxes = np.asarray([9.0386868e-03, 5.1708374e00, 7.4301929e00, 3.1964935e01])
    np.testing.assert_array_almost_equal(result[0]["boxes"][2, :], expected_detection_boxes, decimal=3)

    assert result[0]["scores"].shape == (100,)
    expected_detection_scores = np.asarray(
        [
            0.00383973,
            0.0167976,
            0.01714019,
            0.00073999,
            0.00467391,
            0.02399586,
            0.00093301,
            0.02143953,
            0.00202136,
            0.00266351,
        ]
    )
    np.testing.assert_array_almost_equal(result[0]["scores"][:10], expected_detection_scores, decimal=6)

    assert result[0]["labels"].shape == (100,)
    expected_detection_classes = np.asarray([17, 17, 17, 3, 88, 17, 17, 17, 88, 17])
    np.testing.assert_array_almost_equal(result[0]["labels"][:10], expected_detection_classes, decimal=6)


@pytest.mark.only_with_platform("pytorch")
def test_loss_gradient(get_pytorch_detr):

    object_detector, x_test, y_test = get_pytorch_detr

    grads = object_detector.loss_gradient(x=x_test, y=y_test)

    assert grads.shape == (2, 3, 32, 32)

    expected_gradients1 = np.asarray(
        [
            0.04711548,
            0.25275955,
            0.3609573,
            -0.02207462,
            0.02886475,
            0.05820496,
            0.04151949,
            -0.07008387,
            0.24270807,
            0.17703517,
            -0.29346713,
            -0.11548031,
            -0.15658003,
            -0.1412788,
            0.02577158,
            -0.00550455,
            0.05846804,
            -0.04419752,
            0.06333683,
            -0.15242189,
            -0.06642783,
            -0.09545745,
            -0.01154867,
            0.07477856,
            0.05444539,
            0.01678686,
            0.01427085,
            0.01382115,
            -0.15745601,
            -0.13278124,
            0.06169066,
            -0.03915803,
        ]
    )

    np.testing.assert_array_almost_equal(grads[0, 0, 10, :], expected_gradients1, decimal=2)

    expected_gradients2 = np.asarray(
        [
            -0.10913675,
            0.00539385,
            0.11588555,
            0.02486979,
            -0.23739402,
            -0.01673118,
            -0.09709811,
            0.00763445,
            0.10815062,
            -0.3278629,
            -0.23222731,
            0.28806347,
            -0.14222082,
            -0.24168995,
            -0.20170388,
            -0.24570045,
            -0.01220985,
            -0.18616645,
            -0.19678666,
            -0.12424485,
            -0.36253023,
            0.08978511,
            -0.02874891,
            -0.09320692,
            -0.26761073,
            -0.34595487,
            -0.34932154,
            -0.21606845,
            -0.07342689,
            -0.0573133,
            -0.04900078,
            0.03462576,
        ]
    )
    np.testing.assert_array_almost_equal(grads[1, 0, 10, :], expected_gradients2, decimal=2)


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

    assert grads.shape == (2, 3, 32, 32)


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

    assert pytest.approx(63.9855, abs=0.01) == float(loss)


@pytest.mark.only_with_platform("pytorch")
def test_pgd(get_pytorch_detr):

    object_detector, x_test, y_test = get_pytorch_detr

    from art.attacks.evasion import ProjectedGradientDescent

    attack = ProjectedGradientDescent(estimator=object_detector, max_iter=2)
    x_test_adv = attack.generate(x=x_test, y=y_test)
    np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, x_test_adv, x_test)
