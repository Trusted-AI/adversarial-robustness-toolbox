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
        object_detector, x_test, _ = get_pytorch_faster_rcnn

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

        with pytest.raises(ValueError):
            PyTorchFasterRCNN(
                clip_values=(0, 1),
                attack_losses=("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"),
                preprocessing=(0, 1),
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

        assert grads.shape == (2, 28, 28, 3)

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

        assert pytest.approx(0.84883332, abs=0.01) == float(loss)

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
