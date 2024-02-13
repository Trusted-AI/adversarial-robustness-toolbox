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

from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.mark.only_with_platform("pytorch")
def test_predict(art_warning, get_pytorch_yolo):
    try:
        from art.utils import non_maximum_suppression

        object_detector, x_test, _ = get_pytorch_yolo

        preds = object_detector.predict(x_test)
        result = non_maximum_suppression(preds[0], iou_threshold=0.4, confidence_threshold=0.3)
        assert list(result.keys()) == ["boxes", "labels", "scores"]

        assert result["boxes"].shape == (1, 4)
        expected_detection_boxes = np.asarray([[19.709427, 39.02864, 402.08032, 383.65576]])
        np.testing.assert_array_almost_equal(result["boxes"], expected_detection_boxes, decimal=3)

        assert result["scores"].shape == (1,)
        expected_detection_scores = np.asarray([0.40862876])
        np.testing.assert_array_almost_equal(result["scores"], expected_detection_scores, decimal=3)

        assert result["labels"].shape == (1,)
        expected_detection_classes = np.asarray([23])
        np.testing.assert_array_equal(result["labels"], expected_detection_classes)

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_fit(art_warning, get_pytorch_yolo):
    try:
        object_detector, x_test, y_test = get_pytorch_yolo

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
def test_loss_gradient(art_warning, get_pytorch_yolo):
    try:
        object_detector, x_test, y_test = get_pytorch_yolo

        grads = object_detector.loss_gradient(x=x_test, y=y_test)

        assert grads.shape == (1, 3, 416, 416)

        expected_gradients1 = np.asarray(
            [
                -0.00033619,
                0.00458546,
                -0.00084969,
                -0.00095304,
                -0.00403843,
                0.00225406,
                -0.00369539,
                -0.0099816,
                -0.01046214,
                -0.00290693,
                0.00075546,
                -0.0002135,
                -0.00659937,
                -0.00380152,
                -0.00593928,
                -0.00179838,
                -0.00213012,
                0.00526429,
                0.00332446,
                0.00543861,
                0.00284291,
                0.00426832,
                -0.00586808,
                -0.0017767,
                -0.00231807,
                -0.01142277,
                -0.00021731,
                0.00076714,
                0.00289533,
                0.00993828,
                0.00472939,
                0.00232432,
            ]
        )
        np.testing.assert_array_almost_equal(grads[0, 0, 208, 192:224], expected_gradients1, decimal=2)

        expected_gradients2 = np.asarray(
            [
                0.00079487,
                0.00426403,
                -0.00151893,
                0.00798506,
                0.00937666,
                0.01206836,
                -0.00319753,
                0.00506421,
                0.00291614,
                -0.00053876,
                0.00281978,
                -0.0027451,
                0.00319698,
                0.00287863,
                0.00370754,
                0.004611,
                -0.00213012,
                0.00440465,
                -0.00077857,
                0.00023536,
                0.0035248,
                -0.00810297,
                0.00698602,
                0.00877033,
                0.01452724,
                0.00161957,
                0.02649526,
                -0.0071549,
                0.02670361,
                -0.00759722,
                -0.02353876,
                0.00860081,
            ]
        )
        np.testing.assert_array_almost_equal(grads[0, 0, 192:224, 208], expected_gradients2, decimal=2)

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_errors(art_warning):
    try:
        from pytorchyolo import models

        from art.estimators.object_detection.pytorch_yolo import PyTorchYolo

        model_path = "/tmp/PyTorch-YOLOv3/config/yolov3.cfg"
        weights_path = "/tmp/PyTorch-YOLOv3/weights/yolov3.weights"
        model = models.load_model(model_path=model_path, weights_path=weights_path)

        with pytest.raises(ValueError):
            PyTorchYolo(
                model=model,
                clip_values=(1, 2),
                attack_losses=("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"),
            )

        with pytest.raises(ValueError):
            PyTorchYolo(
                model=model,
                clip_values=(-1, 1),
                attack_losses=("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"),
            )

        from art.defences.postprocessor.rounded import Rounded

        post_def = Rounded()
        with pytest.raises(ValueError):
            PyTorchYolo(
                model=model,
                clip_values=(0, 1),
                attack_losses=("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"),
                postprocessing_defences=post_def,
            )

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_preprocessing_defences(art_warning, get_pytorch_yolo):
    try:
        from art.defences.preprocessor.spatial_smoothing import SpatialSmoothing

        pre_def = SpatialSmoothing()

        object_detector, x_test, y_test = get_pytorch_yolo

        object_detector.set_params(preprocessing_defences=pre_def)

        # Compute gradients
        grads = object_detector.loss_gradient(x=x_test, y=y_test)

        assert grads.shape == (1, 3, 416, 416)

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_compute_losses(art_warning, get_pytorch_yolo):
    try:
        object_detector, x_test, y_test = get_pytorch_yolo
        losses = object_detector.compute_losses(x=x_test, y=y_test)
        assert len(losses) == 1

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_compute_loss(art_warning, get_pytorch_yolo):
    try:
        object_detector, x_test, y_test = get_pytorch_yolo

        # Compute loss
        loss = object_detector.compute_loss(x=x_test, y=y_test)

        assert pytest.approx(0.0920641, abs=0.05) == float(loss)

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_pgd(art_warning, get_pytorch_yolo):
    try:
        from art.attacks.evasion import ProjectedGradientDescent

        object_detector, x_test, y_test = get_pytorch_yolo

        attack = ProjectedGradientDescent(estimator=object_detector, max_iter=2)
        x_test_adv = attack.generate(x=x_test, y=y_test)
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, x_test_adv, x_test)

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_patch(art_warning, get_pytorch_yolo):
    try:

        from art.attacks.evasion import AdversarialPatchPyTorch

        rotation_max = 0.0
        scale_min = 0.1
        scale_max = 0.3
        distortion_scale_max = 0.0
        learning_rate = 1.99
        max_iter = 2
        batch_size = 16
        patch_shape = (3, 5, 5)
        patch_type = "circle"
        optimizer = "pgd"

        object_detector, x_test, y_test = get_pytorch_yolo

        ap = AdversarialPatchPyTorch(
            estimator=object_detector,
            rotation_max=rotation_max,
            scale_min=scale_min,
            scale_max=scale_max,
            optimizer=optimizer,
            distortion_scale_max=distortion_scale_max,
            learning_rate=learning_rate,
            max_iter=max_iter,
            batch_size=batch_size,
            patch_shape=patch_shape,
            patch_type=patch_type,
            verbose=True,
            targeted=False,
        )

        _, _ = ap.generate(x=x_test, y=y_test)

        patched_images = ap.apply_patch(x_test, scale=0.4)
        result = object_detector.predict(patched_images)

        assert result[0]["scores"].shape == (10647,)
        expected_detection_scores = np.asarray(
            [
                2.0061936e-08,
                8.2958641e-06,
                1.5368976e-05,
                8.5753290e-06,
                1.5901747e-05,
                3.8245958e-05,
                4.6325898e-05,
                7.1730128e-06,
                4.3095843e-06,
                1.0766385e-06,
            ]
        )
        np.testing.assert_raises(
            AssertionError, np.testing.assert_array_almost_equal, result[0]["scores"][:10], expected_detection_scores, 6
        )

    except ARTTestException as e:
        art_warning(e)
