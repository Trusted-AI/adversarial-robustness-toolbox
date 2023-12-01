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
        object_detector, x_test, _ = get_pytorch_object_detector

        result = object_detector.predict(x_test)
        assert list(result[0].keys()) == ["boxes", "labels", "scores"]

        assert result[0]["boxes"].shape == (7, 4)
        expected_detection_boxes = np.asarray([4.4017954, 6.3090835, 22.128296, 27.570665])
        np.testing.assert_array_almost_equal(result[0]["boxes"][2, :], expected_detection_boxes, decimal=3)

        assert result[0]["scores"].shape == (7,)
        expected_detection_scores = np.asarray(
            [0.3314798, 0.14125851, 0.13928168, 0.0996184, 0.08550017, 0.06690315, 0.05359321]
        )
        np.testing.assert_array_almost_equal(result[0]["scores"][:10], expected_detection_scores, decimal=6)

        assert result[0]["labels"].shape == (7,)
        expected_detection_classes = np.asarray([72, 79, 1, 72, 78, 72, 82])
        np.testing.assert_array_almost_equal(result[0]["labels"][:10], expected_detection_classes, decimal=6)

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_predict_mask(art_warning, get_pytorch_object_detector_mask):
    try:
        object_detector, x_test, _ = get_pytorch_object_detector_mask

        result = object_detector.predict(x_test)
        assert list(result[0].keys()) == ["boxes", "labels", "scores", "masks"]

        assert result[0]["boxes"].shape == (4, 4)
        expected_detection_boxes = np.asarray([8.62889, 11.735134, 16.353355, 27.565004])
        np.testing.assert_array_almost_equal(result[0]["boxes"][2, :], expected_detection_boxes, decimal=3)

        assert result[0]["scores"].shape == (4,)
        expected_detection_scores = np.asarray([0.45197296, 0.12707493, 0.082677, 0.05386855])
        np.testing.assert_array_almost_equal(result[0]["scores"][:10], expected_detection_scores, decimal=4)

        assert result[0]["labels"].shape == (4,)
        expected_detection_classes = np.asarray([72, 72, 1, 1])
        np.testing.assert_array_almost_equal(result[0]["labels"][:10], expected_detection_classes, decimal=6)

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
        assert grads.shape == (2, 28, 28, 3)

        expected_gradients1 = np.asarray(
            [
                [4.6265591e-04, 1.2323459e-03, 1.3915040e-03],
                [-3.2658060e-04, -3.6941725e-03, -4.5638453e-04],
                [7.8702159e-04, -3.3072452e-03, 3.0583731e-04],
                [1.0381485e-03, -2.0846087e-03, 2.3015277e-04],
                [2.1460971e-03, -1.3157589e-03, 3.5176644e-04],
                [3.3839934e-03, 1.3083456e-03, 1.6155940e-03],
                [3.8621046e-03, 1.6645766e-03, 1.8313043e-03],
                [3.0887076e-03, 1.4632678e-03, 1.1174511e-03],
                [3.3404885e-03, 2.0578136e-03, 9.6874911e-04],
                [3.2202434e-03, 7.2660763e-04, 8.9162006e-04],
                [3.5761783e-03, 2.3615893e-03, 8.8510796e-04],
                [3.4721815e-03, 1.9500104e-03, 9.2907902e-04],
                [3.4767685e-03, 2.1154548e-03, 5.5654044e-04],
                [3.9492580e-03, 3.5505455e-03, 6.5863604e-04],
                [3.9963769e-03, 4.0338552e-03, 3.9539216e-04],
                [2.2312226e-03, 5.1399925e-06, -1.0743635e-03],
                [2.3955442e-03, 6.7116896e-04, -1.2389944e-03],
                [1.9969011e-03, -4.5717746e-04, -1.5225793e-03],
                [1.8131963e-03, -7.7948131e-04, -1.6078206e-03],
                [1.4277012e-03, -7.7973347e-04, -1.3463887e-03],
                [7.3705515e-04, -1.1704378e-03, -9.8979671e-04],
                [1.0899740e-04, -1.2144407e-03, -1.1339665e-03],
                [1.2254890e-04, -4.7438752e-04, -8.8673591e-04],
                [7.0695346e-04, 7.2568876e-04, -2.5591519e-04],
                [5.0835893e-04, 2.6866698e-04, 2.2731400e-04],
                [-5.9932750e-04, -1.1667561e-03, -4.8044650e-04],
                [4.0421321e-04, 3.1692928e-04, -8.3296909e-05],
                [4.0506107e-05, -3.1728629e-04, -4.4132984e-04],
            ]
        )
        np.testing.assert_array_almost_equal(grads[0, 0, :, :], expected_gradients1, decimal=2)

        expected_gradients2 = np.asarray(
            [
                [4.7986404e-04, 7.7701372e-04, 1.1786318e-03],
                [7.3503907e-04, -2.3474507e-03, -3.9008856e-04],
                [4.1874062e-04, -2.5707064e-03, -1.1054531e-03],
                [-1.7942721e-03, -3.3968450e-03, -1.4989552e-03],
                [-2.9697213e-03, -4.6922294e-03, -1.3162185e-03],
                [-3.1759157e-03, -9.8660104e-03, -4.7163852e-03],
                [1.8666144e-03, -2.8793041e-03, -3.1324378e-03],
                [1.0555880e-02, 7.6373261e-03, 5.3013843e-03],
                [8.9815725e-04, -1.0321697e-02, 1.4192325e-03],
                [8.5643278e-03, 3.0152409e-03, 2.0114987e-03],
                [-2.7870361e-03, -1.1686913e-02, -7.0649502e-03],
                [-7.7482774e-03, -1.3334424e-03, -9.1927368e-03],
                [-8.1487820e-03, -3.8133820e-03, -4.3300558e-03],
                [-7.7006700e-03, -1.2594147e-02, -3.9680018e-03],
                [-9.5743872e-03, -2.1007264e-02, -9.1963671e-03],
                [-8.6777220e-03, -1.7278835e-02, -1.3328674e-02],
                [-1.7368209e-02, -2.3461722e-02, -1.1538444e-02],
                [-4.6307812e-03, -5.7058665e-03, 1.3555109e-03],
                [4.8570461e-03, -5.8050654e-03, 8.1082489e-03],
                [6.4304657e-03, 2.8407066e-03, 8.7463465e-03],
                [5.0593228e-03, 1.4102085e-03, 5.2116364e-03],
                [2.5003455e-03, -6.0178695e-04, 2.0183939e-03],
                [2.1247163e-03, 4.7659015e-04, 7.5940741e-04],
                [1.3499497e-03, 6.2203623e-04, 1.2288829e-04],
                [2.8991612e-04, -4.0216290e-04, -7.2287643e-05],
                [6.6898909e-05, -6.3778006e-04, -3.6294860e-04],
                [5.3613615e-04, 9.9137833e-05, -1.6657988e-05],
                [-3.9828232e-05, -3.8453130e-04, -2.3702848e-04],
            ]
        )
        np.testing.assert_array_almost_equal(grads[1, 0, :, :], expected_gradients2, decimal=2)

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_loss_gradient_mask(art_warning, get_pytorch_object_detector_mask):
    try:
        object_detector, x_test, y_test = get_pytorch_object_detector_mask

        # Compute gradients
        grads = object_detector.loss_gradient(x_test, y_test)
        assert grads.shape == (2, 28, 28, 3)

        expected_gradients1 = np.asarray(
            [
                [1.2062087e-03, 6.7400718e-03, 9.5682510e-04],
                [-3.6111937e-03, -5.3175041e-03, -3.2421902e-03],
                [1.4717830e-03, 1.0347518e-03, 1.7675158e-04],
                [2.9278828e-03, 5.0933827e-03, 3.5095078e-04],
                [-3.1896026e-04, 3.6363016e-04, -6.6032895e-04],
                [-3.8130947e-03, -5.5106943e-03, -2.3003859e-03],
                [-4.1348115e-03, -6.5722968e-03, -1.5899740e-03],
                [-2.4562061e-03, -4.1960045e-03, -1.7881666e-03],
                [2.2911791e-04, -6.4335053e-04, -1.6564501e-03],
                [-1.2582233e-03, -1.5607923e-03, -2.2904854e-03],
                [-1.8436739e-03, -2.7200577e-03, -2.9125123e-03],
                [-1.5151387e-03, -4.4148900e-03, -1.7429549e-03],
                [5.4955669e-03, 8.1859864e-03, 1.6560742e-03],
                [3.1721895e-03, 2.4013112e-03, -1.9453048e-04],
                [5.1122587e-03, 7.4281446e-03, 2.4133435e-04],
                [2.7988979e-03, 4.4798232e-03, -1.2488490e-03],
                [3.1651880e-03, 4.5040119e-03, -1.6507130e-03],
                [8.5774017e-04, 9.9022139e-04, -3.1324981e-03],
                [3.8568545e-04, 4.7918499e-04, -2.4925626e-03],
                [-1.8368122e-03, -3.9491002e-03, -3.9275796e-03],
                [1.6731160e-03, 1.5304115e-03, -1.4627117e-03],
                [1.4445755e-03, 1.4263670e-03, -2.0084691e-03],
                [2.0193408e-04, 7.2605687e-04, -1.8740210e-03],
                [-1.3681910e-03, 1.7499415e-05, -2.4952039e-03],
                [1.3475126e-04, 3.0096075e-03, -2.4493274e-04],
                [-6.2653446e-03, -9.5283017e-03, -2.9458744e-03],
                [-2.6554640e-03, -1.4588287e-03, -3.2393888e-03],
                [-6.4712246e-03, -7.2136321e-03, -5.4933843e-03],
            ]
        )
        np.testing.assert_array_almost_equal(grads[0, 0, :, :], expected_gradients1, decimal=2)

        expected_gradients2 = np.asarray(
            [
                [-2.0123991e-04, -9.0955076e-04, -2.2947363e-04],
                [3.0414842e-04, 3.4150464e-04, 2.1101040e-04],
                [6.6070761e-06, -1.8034373e-04, 1.3608378e-05],
                [-1.3393547e-05, -3.2230929e-04, -5.5581659e-05],
                [-1.0353983e-04, -2.7751207e-04, -2.3205159e-04],
                [-5.3371373e-04, -1.1550108e-03, -2.6975147e-04],
                [-2.6593581e-04, -7.3971582e-04, -7.4292002e-05],
                [-9.3046663e-05, -4.0410538e-04, -1.4271366e-04],
                [-1.3833238e-04, -5.6283473e-04, -8.4650565e-05],
                [-8.0315210e-04, -1.4300735e-03, -9.3330207e-05],
                [2.7694018e-04, 6.8307301e-04, 5.5274006e-04],
                [3.1839000e-04, 9.7277382e-04, 4.6252453e-04],
                [2.8279822e-04, 6.2632316e-04, 3.3778447e-04],
                [4.0508871e-04, 1.2438387e-03, 3.6151547e-04],
                [-7.5090391e-04, -2.6640363e-04, -2.6418429e-04],
                [-2.3455340e-03, -4.9932003e-03, -8.0432917e-04],
                [4.1711782e-03, 5.3390805e-03, 2.4412808e-03],
                [5.1162727e-03, 5.2886135e-03, 3.6190096e-03],
                [6.9976337e-03, 9.7018024e-03, 3.8526775e-03],
                [4.5005931e-03, 4.3762275e-03, 1.7228650e-03],
                [6.3695023e-03, 8.4943371e-03, 1.7638379e-03],
                [3.0587378e-03, 3.9485283e-03, 4.9000646e-05],
                [-3.2190280e-04, -6.6311209e-04, -9.8086358e-04],
                [8.3606638e-04, 2.0184387e-03, -3.5464868e-04],
                [-1.8979331e-04, 3.1042210e-04, -4.2471994e-04],
                [-8.8790455e-04, -1.4127755e-03, -4.4270226e-04],
                [4.1172301e-04, 2.9453568e-04, 2.1122720e-04],
                [1.6500468e-04, 3.7142841e-04, -4.5339554e-04],
            ]
        )
        np.testing.assert_array_almost_equal(grads[1, 0, :, :], expected_gradients2, decimal=2)

    except ARTTestException as e:
        art_warning(e)
