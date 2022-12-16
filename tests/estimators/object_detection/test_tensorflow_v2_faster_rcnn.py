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

from art.utils import load_dataset

from tests.utils import ARTTestException, master_seed

logger = logging.getLogger(__name__)


@pytest.mark.skip_module("object_detection")
@pytest.mark.only_with_platform("tensorflow2")
def test_tf_faster_rcnn(art_warning):
    try:
        master_seed(seed=1234, set_tensorflow=True)

        # Only import if object detection module is available
        from art.estimators.object_detection.tensorflow_v2_faster_rcnn import TensorFlowV2FasterRCNN

        # Get test data
        (_, _), (x_test, _), _, _ = load_dataset("cifar10")
        x_test = x_test[:1]
        input_shape = tuple(x_test.shape[1:])

        obj_dec = TensorFlowV2FasterRCNN(input_shape=input_shape)

        # First test predict
        result = obj_dec.predict(x_test)

        assert list(result[0].keys()) == ["boxes", "labels", "scores"]

        assert result[0]["boxes"].shape == (300, 4)
        expected_detection_boxes = np.asarray([0.11661448, 0.46441936, 0.868652, 0.99897605])
        np.testing.assert_array_almost_equal(result[0]["boxes"][2, :], expected_detection_boxes, decimal=3)

        assert result[0]["scores"].shape == (300,)
        expected_detection_scores = np.asarray(
            [
                0.09375852,
                0.04189522,
                0.03094958,
                0.01798256,
                0.01198419,
                0.00832632,
                0.0076474,
                0.00762132,
                0.00662233,
                0.00605294,
            ]
        )
        np.testing.assert_array_almost_equal(result[0]["scores"][:10], expected_detection_scores, decimal=3)

        assert result[0]["labels"].shape == (300,)
        expected_detection_classes = np.asarray([84, 84, 84, 84, 54, 84, 0, 84, 84, 46])
        np.testing.assert_array_almost_equal(result[0]["labels"][:10], expected_detection_classes)

        # Then test loss gradient
        # Create labels
        y = [{"boxes": result[0]["boxes"], "labels": result[0]["labels"], "scores": np.ones_like(result[0]["labels"])}]

        # Compute gradients
        grads = obj_dec.loss_gradient(x_test[:1], y)

        assert grads.shape == (1, 32, 32, 3)

        expected_gradients = np.asarray(
            [
                [4.71095882e-05, 8.39344830e-06, 5.82142093e-06],
                [-1.08160377e-04, 2.27618133e-04, -5.05365351e-05],
                [-3.99912205e-05, 4.34129965e-04, 5.15039403e-07],
                [-1.41627097e-04, 5.76516759e-05, 8.09017874e-05],
                [6.20874243e-06, 2.76702020e-04, -2.28589524e-05],
                [4.32882962e-06, -4.03481281e-05, -7.62806958e-05],
                [-8.37679690e-05, -2.18129950e-04, -9.04274712e-05],
                [-3.67438624e-05, -2.10944389e-04, -9.05940469e-05],
                [-1.75537971e-05, -3.48282600e-04, -8.23857263e-05],
                [5.30820471e-05, -2.86138180e-04, -7.48635794e-05],
                [8.23858354e-05, -1.13339869e-04, -3.49664515e-05],
                [1.59092160e-04, 6.47279012e-05, -1.79400049e-05],
                [-1.63149289e-05, 3.31735782e-06, -4.67280151e-05],
                [-1.98027137e-05, 7.51307743e-05, 2.54857914e-05],
                [6.82972241e-05, 5.97215876e-05, 1.28419619e-04],
                [2.36609863e-04, 2.10922881e-04, 3.50106740e-04],
                [1.47960993e-04, -7.82240568e-06, 2.49002245e-04],
                [7.26436410e-05, -2.87787785e-04, 2.79000669e-04],
                [3.83440783e-05, -2.70720484e-04, 2.26368575e-04],
                [9.61775513e-05, -6.31674193e-05, 2.45212781e-04],
                [1.52097986e-04, -1.31431938e-04, 2.59597989e-04],
                [4.55492234e-04, -8.12296203e-05, 2.45009316e-04],
                [8.83234490e-04, 7.50507315e-05, 3.66395630e-04],
                [1.27273344e-03, 5.82617824e-04, 3.64959065e-04],
                [1.28024910e-03, 5.57218504e-04, 5.37421904e-04],
                [1.23637193e-03, 8.37092637e-04, 4.96422348e-04],
                [7.82434945e-04, 6.26019435e-04, 2.94824480e-04],
                [2.37986183e-04, 1.05056606e-04, 6.55481563e-05],
                [-2.00339942e-04, -1.20437835e-04, -1.66682788e-04],
                [-5.84452318e-05, -1.35937284e-04, -1.51533968e-04],
                [-4.77294707e-05, -2.31526021e-04, -1.24137907e-04],
                [-1.15175018e-04, -1.65152203e-04, -7.45802899e-05],
            ]
        )
        np.testing.assert_array_almost_equal(grads[0, 0, :, :], expected_gradients, decimal=2)

        # Test the predictions and gradients constant for multiple calls of same input
        result_dup = obj_dec.predict(x_test)
        np.testing.assert_array_almost_equal(result[0]["boxes"][2, :], result_dup[0]["boxes"][2, :], decimal=3)
        np.testing.assert_array_almost_equal(result[0]["scores"][:10], result_dup[0]["scores"][:10], decimal=3)
        np.testing.assert_array_almost_equal(result[0]["labels"][:10], result_dup[0]["labels"][:10])

        y_dup = [
            {
                "boxes": result_dup[0]["boxes"],
                "labels": result_dup[0]["labels"],
                "scores": np.ones_like(result_dup[0]["labels"]),
            }
        ]
        grads_dup = obj_dec.loss_gradient(x_test[:1], y_dup)
        np.testing.assert_array_almost_equal(grads[0, 0, :, :], grads_dup[0, 0, :, :], decimal=2)

        # Then test loss gradient with standard format
        # Create labels
        result_tf = obj_dec.predict(x_test, standardise_output=False)
        result = obj_dec.predict(x_test, standardise_output=True)

        from art.estimators.object_detection.utils import convert_tf_to_pt

        result_pt = convert_tf_to_pt(y=result_tf, height=x_test.shape[1], width=x_test.shape[2])

        np.testing.assert_array_equal(result[0]["boxes"], result_pt[0]["boxes"])
        np.testing.assert_array_equal(result[0]["labels"], result_pt[0]["labels"])
        np.testing.assert_array_equal(result[0]["scores"], result_pt[0]["scores"])

        y = [{"boxes": result[0]["boxes"], "labels": result[0]["labels"], "scores": np.ones_like(result[0]["labels"])}]

        # Compute gradients
        grads = obj_dec.loss_gradient(x_test[:1], y, standardise_output=True)

        assert grads.shape == (1, 32, 32, 3)

        expected_gradients = np.asarray(
            [
                [3.20258296e-05, 1.18448479e-04, 5.11901126e-05],
                [5.20282541e-04, 4.31609747e-04, -2.20944603e-05],
                [8.58511135e-04, -3.65446263e-04, 1.36029223e-04],
                [3.47276509e-05, -8.23178736e-04, -2.23043782e-04],
                [1.37017007e-04, -5.57144347e-04, -1.98563328e-04],
                [7.77723617e-05, -3.10823030e-04, -4.05157916e-04],
                [-1.72378772e-04, -3.07416456e-04, -4.87639132e-04],
                [-8.86716807e-05, -4.24879254e-04, -7.27561070e-04],
                [-1.00905372e-05, -1.06700277e-03, -6.71751099e-04],
                [-3.24819557e-04, -1.03535608e-03, -8.76112608e-04],
                [-4.44697944e-04, -2.94831785e-04, -1.28169032e-03],
                [-2.53191713e-04, 7.62609925e-05, -1.11494109e-03],
                [-3.77789664e-04, -7.78697427e-07, -9.06005735e-04],
                [-1.37880095e-04, 2.21254071e-04, -7.10642664e-04],
                [-1.49297048e-04, -2.67961295e-04, -5.52246522e-04],
                [-3.48978385e-04, -7.13465444e-04, -7.76897825e-04],
                [-6.15608995e-04, -1.12241239e-03, -1.01537013e-03],
                [-6.28621201e-04, -1.58125453e-03, -9.77508142e-04],
                [-3.45328706e-04, -1.32934854e-03, -8.63111869e-04],
                [-2.70117362e-05, -8.22714996e-04, -4.30601096e-04],
                [1.72943372e-04, -1.61864562e-04, -2.35522675e-04],
                [2.82775087e-04, -1.11206624e-04, -2.37867673e-04],
                [2.06765530e-04, -2.46381125e-04, -2.45586620e-04],
                [1.34378410e-04, -3.13195225e-04, -4.79331560e-04],
                [3.81465390e-04, -4.43120458e-04, -3.82220693e-04],
                [8.27364449e-04, 6.86867425e-05, -1.02168444e-04],
                [4.70337487e-04, 1.94208715e-05, -1.88505583e-04],
                [1.42172328e-04, 1.59842341e-04, -1.30101529e-04],
                [3.61582970e-05, -1.23159880e-05, -2.09606616e-04],
                [2.71195429e-04, 1.06499057e-04, -7.41584881e-05],
                [1.49051149e-04, -5.21843147e-04, -2.40127789e-04],
                [1.46682723e-04, -2.08231620e-04, -2.79343774e-04],
            ]
        )
        np.testing.assert_array_almost_equal(grads[0, 0, :, :], expected_gradients, decimal=2)

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("tensorflow2")
def test_errors(art_warning):
    try:
        from art.estimators.object_detection.tensorflow_v2_faster_rcnn import TensorFlowV2FasterRCNN

        (_, _), (x_test, _), _, _ = load_dataset("cifar10")
        input_shape = tuple(x_test.shape[1:])

        with pytest.raises(ValueError):
            obj_det = TensorFlowV2FasterRCNN(
                input_shape=input_shape,
                clip_values=(1, 2),
                attack_losses=("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"),
            )
            obj_det.predict(x_test[:1])

        with pytest.raises(ValueError):
            TensorFlowV2FasterRCNN(
                input_shape=input_shape,
                clip_values=(-1, 1),
                attack_losses=("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"),
            )
            obj_det.predict(x_test[:1])

        from art.defences.postprocessor.rounded import Rounded

        post_def = Rounded()
        with pytest.raises(ValueError):
            TensorFlowV2FasterRCNN(
                input_shape=input_shape,
                clip_values=(0, 1),
                attack_losses=("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"),
                postprocessing_defences=post_def,
            )
            obj_det.predict(x_test[:1])

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("tensorflow2")
def test_preprocessing_defences(art_warning):
    try:
        from art.estimators.object_detection.tensorflow_v2_faster_rcnn import TensorFlowV2FasterRCNN
        from art.defences.preprocessor.spatial_smoothing import SpatialSmoothing

        (_, _), (x_test, _), _, _ = load_dataset("cifar10")
        input_shape = tuple(x_test.shape[1:])

        pre_def = SpatialSmoothing()

        with pytest.raises(ValueError):
            _ = TensorFlowV2FasterRCNN(
                input_shape=input_shape,
                clip_values=(0, 1),
                attack_losses=("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"),
                preprocessing_defences=pre_def,
            )

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("tensorflow2")
def test_compute_losses(art_warning):
    try:
        from art.estimators.object_detection.tensorflow_v2_faster_rcnn import TensorFlowV2FasterRCNN

        # Get test data
        (_, _), (x_test, _), _, _ = load_dataset("cifar10")
        input_shape = tuple(x_test.shape[1:])

        frcnn = TensorFlowV2FasterRCNN(
            input_shape=input_shape,
            clip_values=(0, 255),
            attack_losses=(
                "Loss/RPNLoss/localization_loss",
                "Loss/RPNLoss/objectness_loss",
                "Loss/BoxClassifierLoss/localization_loss",
                "Loss/BoxClassifierLoss/classification_loss",
            ),
        )

        # Create labels
        result = frcnn.predict(x_test[:2])

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

        # Compute losses
        losses = frcnn.compute_losses(x_test[:2], y)

        assert len(losses) == 4

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("tensorflow2")
def test_compute_loss(art_warning):
    try:
        from art.estimators.object_detection.tensorflow_v2_faster_rcnn import TensorFlowV2FasterRCNN

        # Get test data
        (_, _), (x_test, _), _, _ = load_dataset("cifar10")
        input_shape = tuple(x_test.shape[1:])

        frcnn = TensorFlowV2FasterRCNN(
            input_shape=input_shape,
            clip_values=(0, 255),
            attack_losses=(
                "Loss/RPNLoss/localization_loss",
                "Loss/RPNLoss/objectness_loss",
                "Loss/BoxClassifierLoss/localization_loss",
                "Loss/BoxClassifierLoss/classification_loss",
            ),
        )

        # Create labels
        result = frcnn.predict(x_test[:2])

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

        # Compute loss
        loss = frcnn.compute_loss(x_test[:2], y)[0]

        assert float(loss) == pytest.approx(11.245838, abs=0.5)

    except ARTTestException as e:
        art_warning(e)
