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
        object_detector, x_test, _ = get_pytorch_yolo

        result = object_detector.predict(x=x_test)

        assert list(result[0].keys()) == ["boxes", "labels", "scores"]

        assert result[0]["boxes"].shape == (10647, 4)
        expected_detection_boxes = np.asarray([0.0000000e00, 0.0000000e00, 1.6367816e02, 4.4342079e01])
        np.testing.assert_array_almost_equal(result[0]["boxes"][2, :], expected_detection_boxes, decimal=3)

        assert result[0]["scores"].shape == (10647,)
        expected_detection_scores = np.asarray(
            [
                4.3653536e-08,
                3.3987994e-06,
                2.5681820e-06,
                3.9782722e-06,
                2.1766680e-05,
                2.6138965e-05,
                6.3377396e-05,
                7.6248516e-06,
                4.3447722e-06,
                3.6515078e-06,
            ]
        )
        np.testing.assert_array_almost_equal(result[0]["scores"][:10], expected_detection_scores, decimal=6)

        assert result[0]["labels"].shape == (10647,)
        expected_detection_classes = np.asarray([0, 0, 14, 14, 14, 14, 14, 14, 14, 0])
        np.testing.assert_array_almost_equal(result[0]["labels"][:10], expected_detection_classes, decimal=6)

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

        assert grads.shape == (2, 3, 416, 416)

        expected_gradients1 = np.asarray(
            [
                0.012576922,
                -0.005133151,
                -0.0028872574,
                -0.0029357928,
                -0.008929219,
                0.012767567,
                -0.00715934,
                0.00987368,
                -0.0014089097,
                -0.004765472,
                -0.007845592,
                -0.0065127434,
                -0.00047654763,
                -0.018194549,
                0.00025652442,
                -0.01420591,
                0.03873131,
                0.080963746,
                -0.009225381,
                0.026824722,
                0.005942673,
                -0.025760904,
                0.008754236,
                -0.037260942,
                0.027838552,
                0.0485742,
                0.020763855,
                -0.013568859,
                -0.0071423287,
                0.000802512,
                0.012983642,
                0.006466129,
                0.0025194373,
                -0.012298459,
                -0.01168492,
                -0.0013298508,
                -0.007176587,
                0.01996972,
                -0.004173076,
                0.029163878,
                0.022482246,
                0.008151911,
                0.025543496,
                0.0007374112,
                0.0008220682,
                -0.005740379,
                0.009537468,
                -0.01116704,
                0.0010225883,
                0.00026052812,
            ]
        )

        np.testing.assert_array_almost_equal(grads[0, 0, 208, 175:225], expected_gradients1, decimal=1)

        expected_gradients2 = np.asarray(
            [
                0.0049910736,
                -0.008941505,
                -0.013645802,
                0.0060615,
                0.0021073571,
                -0.0022195925,
                -0.006654369,
                0.010533731,
                0.0013077373,
                -0.010422451,
                -0.00034834983,
                -0.0040517827,
                -0.0001514384,
                -0.031307846,
                -0.008412821,
                -0.044170827,
                0.055609763,
                0.0220191,
                -0.019813634,
                -0.035893522,
                0.023970673,
                -0.08727841,
                0.0411198,
                0.0072751334,
                0.01716753,
                0.0391037,
                0.020182624,
                0.021557821,
                0.011461802,
                0.0046976856,
                -0.00304008,
                -0.010215744,
                -0.0074639097,
                -0.020115864,
                -0.05325762,
                -0.006238129,
                -0.006486116,
                0.09806269,
                0.03115965,
                0.066279344,
                0.05367205,
                -0.042338565,
                0.04456845,
                0.040167376,
                0.03357561,
                0.01510548,
                0.0006220075,
                -0.027102726,
                -0.020182101,
                -0.04347762,
            ]
        )
        np.testing.assert_array_almost_equal(grads[1, 0, 208, 175:225], expected_gradients2, decimal=1)

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

        assert grads.shape == (2, 3, 416, 416)

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

        assert pytest.approx(11.20741, abs=1.5) == float(loss)

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
                4.3653536e-08,
                3.3987994e-06,
                2.5681820e-06,
                3.9782722e-06,
                2.1766680e-05,
                2.6138965e-05,
                6.3377396e-05,
                7.6248516e-06,
                4.3447722e-06,
                3.6515078e-06,
            ]
        )
        np.testing.assert_raises(
            AssertionError, np.testing.assert_array_almost_equal, result[0]["scores"][:10], expected_detection_scores, 6
        )

    except ARTTestException as e:
        art_warning(e)
