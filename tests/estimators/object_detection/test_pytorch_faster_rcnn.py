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
                -7.2145270e-04,
                -3.9774503e-04,
                -5.5271841e-04,
                2.5251633e-04,
                -4.1167819e-05,
                1.2919735e-04,
                1.1148686e-04,
                4.9278833e-04,
                9.6094189e-04,
                1.1812975e-03,
                2.7167992e-04,
                9.7095188e-05,
                1.4456113e-04,
                -8.8345587e-06,
                4.7151549e-05,
                -1.3497710e-04,
                -2.3394797e-04,
                1.3777621e-04,
                3.2994794e-04,
                3.7001527e-04,
                -2.5945838e-04,
                -8.3444244e-04,
                -6.9832127e-04,
                -3.0403296e-04,
                -5.4019055e-04,
                -3.4545487e-04,
                -5.6993403e-04,
                -2.9818740e-04,
                -9.8479632e-04,
                -4.1015903e-04,
                -6.2145875e-04,
                -1.1365353e-03,
            ]
        )
        np.testing.assert_array_almost_equal(grads[0, 0, 208, 192:224], expected_gradients1, decimal=2)

        expected_gradients2 = np.asarray(
            [
                0.00015462,
                0.00028882,
                -0.00018248,
                -0.00114344,
                -0.00160104,
                -0.00190151,
                -0.00183488,
                -0.00191787,
                -0.00018382,
                0.00095297,
                0.00042502,
                0.00024631,
                0.0002915,
                0.00053676,
                0.00028635,
                0.00035274,
                -0.00023395,
                -0.00044685,
                -0.00016795,
                0.00059767,
                0.00060389,
                0.00010305,
                0.0011498,
                0.00135104,
                0.00095133,
                0.00081004,
                0.00061877,
                0.00089056,
                0.00056647,
                0.00070012,
                0.00016926,
                0.00026042,
            ]
        )
        np.testing.assert_array_almost_equal(grads[0, 0, 192:224, 208], expected_gradients2, decimal=2)

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
