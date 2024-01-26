# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2021
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
def test_predict(art_warning, get_pytorch_object_detector):
    try:
        from art.utils import non_maximum_suppression

        object_detector, x_test, _ = get_pytorch_object_detector

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
def test_predict_mask(art_warning, get_pytorch_object_detector_mask):
    try:
        from art.utils import non_maximum_suppression

        object_detector, x_test, _ = get_pytorch_object_detector_mask

        preds = object_detector.predict(x_test)
        result = non_maximum_suppression(preds[0], iou_threshold=0.4, confidence_threshold=0.3)
        assert list(result.keys()) == ["boxes", "labels", "scores"]

        assert result["boxes"].shape == (2, 4)
        expected_detection_boxes = np.asarray(
            [
                [44.097942, 22.865257, 415.32070, 294.20483],
                [25.739365, 33.178577, 416.00000, 338.51460],
            ]
        )
        np.testing.assert_array_almost_equal(result["boxes"], expected_detection_boxes, decimal=3)

        assert result["scores"].shape == (2,)
        expected_detection_scores = np.asarray([0.67316836, 0.5686724])
        np.testing.assert_array_almost_equal(result["scores"], expected_detection_scores, decimal=3)

        assert result["labels"].shape == (2,)
        expected_detection_classes = np.asarray([18, 21])
        np.testing.assert_array_equal(result["labels"], expected_detection_classes)

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_fit(art_warning, get_pytorch_object_detector):
    try:
        object_detector, x_test, y_test = get_pytorch_object_detector

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
def test_fit_mask(art_warning, get_pytorch_object_detector_mask):
    try:
        object_detector, x_test, y_test = get_pytorch_object_detector_mask

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
def test_loss_gradient(art_warning, get_pytorch_object_detector):
    try:
        object_detector, x_test, y_test = get_pytorch_object_detector

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
def test_loss_gradient_mask(art_warning, get_pytorch_object_detector_mask):
    try:
        object_detector, x_test, y_test = get_pytorch_object_detector_mask

        # Compute gradients
        grads = object_detector.loss_gradient(x_test, y_test)
        assert grads.shape == (1, 3, 416, 416)

        expected_gradients1 = np.asarray(
            [
                -5.5341989e-05,
                -5.4428884e-04,
                5.4366910e-04,
                7.6082360e-04,
                -3.4690551e-05,
                -3.8355158e-04,
                9.4802541e-05,
                -1.2973599e-03,
                -8.5583847e-04,
                -1.9041763e-03,
                -2.0476838e-03,
                1.3446594e-04,
                9.6042868e-04,
                8.8853808e-04,
                4.1893515e-04,
                1.2266783e-04,
                6.0996308e-04,
                4.6253894e-04,
                -1.8787223e-03,
                -1.9494371e-03,
                -1.2018540e-03,
                -7.0822285e-04,
                3.9439899e-04,
                -1.9463699e-03,
                -1.9617968e-03,
                -1.8740186e-04,
                -4.7003134e-04,
                -7.1175391e-04,
                -2.6479245e-03,
                -7.6713605e-04,
                -9.1007189e-04,
                -9.5907447e-04,
            ]
        )
        np.testing.assert_array_almost_equal(grads[0, 0, 208, 192:224], expected_gradients1, decimal=2)

        expected_gradients2 = np.asarray(
            [
                -0.00239724,
                -0.00271061,
                -0.0036578,
                -0.00504796,
                -0.0048536,
                -0.00433594,
                -0.00499022,
                -0.00401875,
                -0.00333852,
                -0.00060027,
                0.00098555,
                0.00249704,
                0.00135383,
                0.00277813,
                0.00033104,
                0.00016026,
                0.00060996,
                0.00010528,
                0.00096368,
                0.00230222,
                0.00169831,
                0.00172231,
                0.00270932,
                0.00224663,
                0.00077922,
                0.00174257,
                0.00041644,
                -0.00126136,
                -0.00112533,
                -0.00110854,
                -0.00126751,
                -0.0014297,
            ]
        )
        np.testing.assert_array_almost_equal(grads[0, 0, 192:224, 208], expected_gradients2, decimal=2)

    except ARTTestException as e:
        art_warning(e)
