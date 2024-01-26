# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
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
def test_predict(art_warning, get_pytorch_faster_rcnn):
    try:
        from art.utils import non_maximum_suppression

        object_detector, x_test, _ = get_pytorch_faster_rcnn

        preds = object_detector.predict(x_test)
        result = non_maximum_suppression(preds[0], iou_threshold=0.4, confidence_threshold=0.3)
        assert list(result.keys()) == ["boxes", "labels", "scores"]

        assert result["boxes"].shape == (2, 4)
        expected_detection_boxes = np.asarray(
            [
                [6.136914, 22.481018, 413.05814, 346.08746],
                [0.000000, 24.181173, 406.47644, 342.62213],
            ]
        )
        np.testing.assert_array_almost_equal(result["boxes"], expected_detection_boxes, decimal=3)

        assert result["scores"].shape == (2,)
        expected_detection_scores = np.asarray([0.4237412, 0.35696018])
        np.testing.assert_array_almost_equal(result["scores"], expected_detection_scores, decimal=3)

        assert result["labels"].shape == (2,)
        expected_detection_classes = np.asarray([21, 18])
        np.testing.assert_array_equal(result["labels"], expected_detection_classes)

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_fit(art_warning, get_pytorch_faster_rcnn):
    try:
        object_detector, x_test, y_test = get_pytorch_faster_rcnn

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
def test_loss_gradient(art_warning, get_pytorch_faster_rcnn):
    try:
        object_detector, x_test, y_test = get_pytorch_faster_rcnn

        # Compute gradients
        grads = object_detector.loss_gradient(x_test, y_test)
        assert grads.shape == (1, 3, 416, 416)

        expected_gradients1 = np.asarray(
            [
                -2.7225273e-05,
                -2.7225284e-05,
                -3.2535860e-05,
                -9.3287526e-06,
                -1.1088990e-05,
                -3.4527478e-05,
                5.7807661e-06,
                1.1616970e-05,
                2.9732121e-06,
                1.1190044e-05,
                -6.4673945e-06,
                -1.6562306e-05,
                -1.5946282e-05,
                -1.8079168e-06,
                -9.7664342e-06,
                6.2075532e-07,
                -8.9023115e-06,
                -1.5546989e-06,
                -7.2730008e-06,
                -7.5181362e-07,
            ]
        )
        np.testing.assert_array_almost_equal(grads[0, 0, 0, :20], expected_gradients1, decimal=2)

        expected_gradients2 = np.asarray(
            [
                -2.7307957e-05,
                -1.9417710e-05,
                -2.0928457e-05,
                -2.1384752e-05,
                -2.5035972e-05,
                -3.6572790e-05,
                -8.2444545e-05,
                -7.3255811e-05,
                -4.5060227e-05,
                -1.9829258e-05,
                -2.2043951e-05,
                -3.6746951e-05,
                -4.2588043e-05,
                -3.1833035e-05,
                -1.5923406e-05,
                -3.5026955e-05,
                -4.4511849e-05,
                -3.3867167e-05,
                -1.8569792e-05,
                -3.5141209e-05,
            ]
        )
        np.testing.assert_array_almost_equal(grads[0, 0, :20, 0], expected_gradients2, decimal=2)

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_errors(art_warning):
    try:
        from art.estimators.object_detection.pytorch_faster_rcnn import PyTorchFasterRCNN

        with pytest.raises(ValueError):
            PyTorchFasterRCNN(
                clip_values=(1, 2),
                attack_losses=("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"),
            )

        with pytest.raises(ValueError):
            PyTorchFasterRCNN(
                clip_values=(-1, 1),
                attack_losses=("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"),
            )

        from art.defences.postprocessor.rounded import Rounded

        post_def = Rounded()
        with pytest.raises(ValueError):
            PyTorchFasterRCNN(
                clip_values=(0, 1),
                attack_losses=("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"),
                postprocessing_defences=post_def,
            )

    except ARTTestException as e:
        art_warning(e)


def test_preprocessing_defences(art_warning, get_pytorch_faster_rcnn):
    try:
        from art.defences.preprocessor.spatial_smoothing import SpatialSmoothing

        pre_def = SpatialSmoothing()

        object_detector, x_test, y_test = get_pytorch_faster_rcnn

        object_detector.set_params(preprocessing_defences=pre_def)

        # Compute gradients
        grads = object_detector.loss_gradient(x=x_test, y=y_test)

        assert grads.shape == (1, 3, 416, 416)

    except ARTTestException as e:
        art_warning(e)


def test_compute_losses(art_warning, get_pytorch_faster_rcnn):
    try:
        object_detector, x_test, y_test = get_pytorch_faster_rcnn
        losses = object_detector.compute_losses(x=x_test, y=y_test)
        assert len(losses) == 4

    except ARTTestException as e:
        art_warning(e)


def test_compute_loss(art_warning, get_pytorch_faster_rcnn):
    try:
        object_detector, x_test, y_test = get_pytorch_faster_rcnn

        # Compute loss
        loss = object_detector.compute_loss(x=x_test, y=y_test)

        assert pytest.approx(0.0995874, abs=0.05) == float(loss)

    except ARTTestException as e:
        art_warning(e)


def test_pgd(art_warning, get_pytorch_faster_rcnn):
    try:
        from art.attacks.evasion import ProjectedGradientDescent

        object_detector, x_test, y_test = get_pytorch_faster_rcnn

        attack = ProjectedGradientDescent(estimator=object_detector, max_iter=2)
        x_test_adv = attack.generate(x=x_test, y=y_test)
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, x_test_adv, x_test)

    except ARTTestException as e:
        art_warning(e)
