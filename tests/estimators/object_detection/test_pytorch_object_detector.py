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
def test_loss_gradient_mask(art_warning, get_pytorch_object_detector_mask):
    try:
        object_detector, x_test, y_test = get_pytorch_object_detector_mask

        # Compute gradients
        grads = object_detector.loss_gradient(x_test, y_test)
        assert grads.shape == (1, 3, 416, 416)

        import pprint

        print()
        pprint.pprint(grads[0, 0, 0, :20])
        print()
        pprint.pprint(grads[0, 0, :20, 0])

        expected_gradients1 = np.asarray(
            [
                -4.2168313e-06,
                -4.4972450e-05,
                -3.6137710e-05,
                -1.2499937e-06,
                1.2728384e-05,
                -1.7352231e-05,
                5.6671047e-06,
                1.4085637e-05,
                5.9047998e-06,
                1.0826078e-05,
                2.2078505e-06,
                -1.3319310e-05,
                -2.4521427e-05,
                -1.8251436e-05,
                -1.9938851e-05,
                -3.6778667e-07,
                1.1899039e-05,
                1.9246204e-06,
                -2.7922330e-05,
                -3.2529952e-06,
            ]
        )
        np.testing.assert_array_almost_equal(grads[0, 0, 0, :20], expected_gradients1, decimal=2)

        expected_gradients2 = np.asarray(
            [
                -4.2168313e-06,
                -9.3028730e-06,
                1.5900954e-06,
                -9.7032771e-06,
                -7.9553565e-06,
                -1.9485701e-06,
                -1.3360468e-05,
                -2.7804586e-05,
                -4.2667002e-06,
                -6.1407286e-06,
                -6.6768125e-06,
                -1.6444834e-06,
                4.7967392e-06,
                2.4288647e-06,
                1.0280205e-05,
                4.2001102e-06,
                2.9494076e-05,
                1.4654281e-05,
                2.5580388e-05,
                3.0241908e-05,
            ]
        )
        np.testing.assert_array_almost_equal(grads[0, 0, :20, 0], expected_gradients2, decimal=2)

    except ARTTestException as e:
        art_warning(e)
