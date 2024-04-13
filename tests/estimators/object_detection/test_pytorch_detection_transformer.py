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
        from art.utils import non_maximum_suppression

        object_detector, x_test, _ = get_pytorch_detr

        preds = object_detector.predict(x_test)
        result = non_maximum_suppression(preds[0], iou_threshold=0.4, confidence_threshold=0.3)
        assert list(result.keys()) == ["boxes", "labels", "scores"]

        assert result["boxes"].shape == (3, 4)
        expected_detection_boxes = np.asarray(
            [
                [1.0126123, 25.658852, 412.70746, 379.12537],
                [-0.089400, 272.08664, 415.90994, 416.25930],
                [0.1522941, 75.882440, 99.139565, 335.11273],
            ]
        )
        np.testing.assert_array_almost_equal(result["boxes"], expected_detection_boxes, decimal=3)

        assert result["scores"].shape == (3,)
        expected_detection_scores = np.asarray([0.8424455, 0.7796526, 0.35387915])
        np.testing.assert_array_almost_equal(result["scores"], expected_detection_scores, decimal=3)

        assert result["labels"].shape == (3,)
        expected_detection_classes = np.asarray([17, 65, 17])
        np.testing.assert_array_equal(result["labels"], expected_detection_classes)

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_fit(art_warning, get_pytorch_detr):
    try:
        object_detector, x_test, y_test = get_pytorch_detr

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

        assert grads.shape == (1, 3, 416, 416)

        expected_gradients1 = np.asarray(
            [
                0.02891439,
                0.0055933,
                -0.00687808,
                0.0095074,
                0.00247894,
                0.00122704,
                -0.00482378,
                -0.00924361,
                -0.02870164,
                -0.00683936,
                0.00904205,
                -0.01315971,
                -0.0151937,
                -0.00156442,
                0.00775309,
                0.01946152,
                0.00523211,
                -0.01682214,
                0.00079588,
                0.01627164,
                -0.01347653,
                -0.00512358,
                0.00610363,
                0.02831643,
                0.00742467,
                0.00293561,
                0.01380033,
                0.02112359,
                0.01725711,
                -0.00431877,
                -0.01007722,
                -0.00526983,
            ]
        )
        np.testing.assert_array_almost_equal(grads[0, 0, 208, 192:224], expected_gradients1, decimal=2)

        expected_gradients2 = np.asarray(
            [
                -0.00549417,
                -0.01592844,
                -0.01073932,
                -0.00443333,
                -0.00780143,
                -0.02033146,
                -0.0191503,
                0.01227987,
                0.019971,
                0.01034214,
                -0.00918145,
                -0.02458049,
                -0.00708776,
                -0.00826812,
                -0.01284431,
                -0.00195021,
                0.00523211,
                0.00661678,
                0.00851441,
                0.01157211,
                -0.00324841,
                -0.00395823,
                0.00756641,
                0.00405913,
                -0.00055517,
                0.00221484,
                -0.02415526,
                -0.02096599,
                0.00980014,
                0.00174731,
                -0.01008899,
                0.00305779,
            ]
        )
        np.testing.assert_array_almost_equal(grads[0, 0, 192:224, 208], expected_gradients2, decimal=2)

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
            }
        ]

        # Compute gradients
        grads = object_detector.loss_gradient(x=x_test, y=y)

        assert grads.shape == (1, 3, 416, 416)

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

        assert pytest.approx(2.172381, abs=0.1) == float(loss)

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
