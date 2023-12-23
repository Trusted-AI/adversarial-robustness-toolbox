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

from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.mark.only_with_platform("pytorch")
def test_predict(art_warning, get_pytorch_detr):
    try:
        object_detector, x_test, _ = get_pytorch_detr

        result = object_detector.predict(x=x_test)

        assert list(result[0].keys()) == ["boxes", "labels", "scores"]

        assert result[0]["boxes"].shape == (100, 4)
        expected_detection_boxes = np.asarray([-0.12423098, 361.80136, 82.385345, 795.50305])
        np.testing.assert_array_almost_equal(result[0]["boxes"][2, :], expected_detection_boxes, decimal=1)

        assert result[0]["scores"].shape == (100,)
        expected_detection_scores = np.asarray(
            [
                0.00105285,
                0.00261505,
                0.00060220,
                0.00121928,
                0.00154554,
                0.00021678,
                0.00077083,
                0.00045684,
                0.00180561,
                0.00067704,
            ]
        )
        np.testing.assert_array_almost_equal(result[0]["scores"][:10], expected_detection_scores, decimal=1)

        assert result[0]["labels"].shape == (100,)
        expected_detection_classes = np.asarray([1, 23, 23, 1, 1, 23, 23, 23, 1, 1])
        np.testing.assert_array_almost_equal(result[0]["labels"][:10], expected_detection_classes, decimal=1)

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_fit(art_warning, get_pytorch_detr):
    try:
        import torch

        object_detector, x_test, y_test = get_pytorch_detr

        # Create optimizer
        params = [p for p in object_detector.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.01)
        object_detector.set_params(optimizer=optimizer)

        # Compute loss before training
        loss1 = object_detector.compute_loss(x=x_test, y=y_test)

        # Train for one epoch
        object_detector.fit(x_test, y_test, nb_epochs=1)

        # Compute loss after training
        loss2 = object_detector.compute_loss(x=x_test, y=y_test)

        assert loss1 != loss2

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_loss_gradient(art_warning, get_pytorch_detr):
    try:
        object_detector, x_test, y_test = get_pytorch_detr

        grads = object_detector.loss_gradient(x=x_test, y=y_test)

        assert grads.shape == (2, 3, 800, 800)

        expected_gradients1 = np.asarray(
            [
                -0.00757495,
                -0.00101332,
                0.00368362,
                0.00283334,
                -0.00096027,
                0.00873749,
                0.00546095,
                -0.00823532,
                -0.00710872,
                0.00389713,
                -0.00966289,
                0.00448294,
                0.00754991,
                -0.00934104,
                -0.00350194,
                -0.00541577,
                -0.00395624,
                0.00147651,
                0.0105616,
                0.01231265,
                -0.00148831,
                -0.0043609,
                0.00093031,
                0.00884939,
                -0.00356749,
                0.00093475,
                -0.00353712,
                -0.0060132,
                -0.00067899,
                -0.00886974,
                0.00108483,
                -0.00052412,
            ]
        )

        print("expected_gradients1")
        print(grads[0, 0, 10, :32])

        np.testing.assert_array_almost_equal(grads[0, 0, 10, :32], expected_gradients1, decimal=1)

        expected_gradients2 = np.asarray(
            [
                -0.00757495,
                -0.00101332,
                0.00368362,
                0.00283334,
                -0.00096027,
                0.00873749,
                0.00546095,
                -0.00823532,
                -0.00710872,
                0.00389713,
                -0.00966289,
                0.00448294,
                0.00754991,
                -0.00934104,
                -0.00350194,
                -0.00541577,
                -0.00395624,
                0.00147651,
                0.0105616,
                0.01231265,
                -0.00148831,
                -0.0043609,
                0.00093031,
                0.00884939,
                -0.00356749,
                0.00093475,
                -0.00353712,
                -0.0060132,
                -0.00067899,
                -0.00886974,
                0.00108483,
                -0.00052412,
            ]
        )

        print("expected_gradients2")
        print(grads[1, 0, 10, :32])

        np.testing.assert_array_almost_equal(grads[1, 0, 10, :32], expected_gradients2, decimal=2)

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_errors(art_warning):
    try:
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

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_preprocessing_defences(art_warning, get_pytorch_detr):
    try:
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

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_compute_losses(art_warning, get_pytorch_detr):
    try:
        object_detector, x_test, y_test = get_pytorch_detr
        losses = object_detector.compute_losses(x=x_test, y=y_test)
        assert len(losses) == 3

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_compute_loss(art_warning, get_pytorch_detr):
    try:
        object_detector, x_test, y_test = get_pytorch_detr

        # Compute loss
        loss = object_detector.compute_loss(x=x_test, y=y_test)

        assert pytest.approx(6.7767677, abs=0.1) == float(loss)

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_pgd(art_warning, get_pytorch_detr):
    try:
        from art.attacks.evasion import ProjectedGradientDescent

        object_detector, x_test, y_test = get_pytorch_detr

        attack = ProjectedGradientDescent(estimator=object_detector, max_iter=2)
        x_test_adv = attack.generate(x=x_test, y=y_test)
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, x_test_adv, x_test)

    except ARTTestException as e:
        art_warning(e)
