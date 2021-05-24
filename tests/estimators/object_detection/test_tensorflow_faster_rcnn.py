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
import unittest
import importlib

import numpy as np
import tensorflow as tf

from tests.utils import TestBase, master_seed

object_detection_spec = importlib.util.find_spec("object_detection")
object_detection_found = object_detection_spec is not None

logger = logging.getLogger(__name__)


@unittest.skipIf(
    not object_detection_found,
    reason="Skip unittests if object detection module is not found because of pre-trained model.",
)
@unittest.skipIf(
    tf.__version__[0] == "2" or (tf.__version__[0] == "1" and tf.__version__.split(".")[1] != "15"),
    reason="Skip unittests if not TensorFlow v1.15 because of pre-trained model.",
)
class TestTensorFlowFasterRCNN(TestBase):
    """
    This class tests the TensorFlowFasterRCNN object detector.
    """

    @classmethod
    def setUpClass(cls):
        master_seed(seed=1234, set_tensorflow=True)
        super().setUpClass()

        cls.n_test = 10
        cls.x_test_mnist = cls.x_test_mnist[0 : cls.n_test]
        cls.y_test_mnist = cls.y_test_mnist[0 : cls.n_test]

        # Only import if object detection module is available
        from art.estimators.object_detection.tensorflow_faster_rcnn import TensorFlowFasterRCNN

        # Define object detector
        images = tf.placeholder(tf.float32, shape=[2, 28, 28, 1])
        cls.obj_dec = TensorFlowFasterRCNN(images=images)

    def test_predict(self):
        result = self.obj_dec.predict(self.x_test_mnist)

        self.assertTrue(
            list(result[0].keys())
            == [
                "boxes",
                "labels",
                "scores",
            ]
        )

        self.assertTrue(result[0]["boxes"].shape == (300, 4))
        expected_detection_boxes = np.asarray([0.65566427, 0.0, 1.0, 0.9642794])
        np.testing.assert_array_almost_equal(result[0]["boxes"][2, :], expected_detection_boxes, decimal=6)

        self.assertTrue(result[0]["scores"].shape == (300,))
        expected_detection_scores = np.asarray(
            [
                3.356745e-04,
                3.190193e-04,
                2.967696e-04,
                2.128600e-04,
                1.726381e-04,
                1.472894e-04,
                1.198768e-04,
                1.109493e-04,
                1.066341e-04,
                8.560477e-05,
            ]
        )
        np.testing.assert_array_almost_equal(result[0]["scores"][:10], expected_detection_scores, decimal=6)

        self.assertTrue(result[0]["labels"].shape == (300,))
        expected_detection_classes = np.asarray([71.0, 81.0, 66.0, 15.0, 63.0, 66.0, 64.0, 84.0, 37.0, 2.0])
        np.testing.assert_array_almost_equal(result[0]["labels"][:10], expected_detection_classes, decimal=6)

    def test_loss_gradient(self):
        # Create labels
        result = self.obj_dec.predict(self.x_test_mnist[:2])

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

        # Compute gradients
        grads = self.obj_dec.loss_gradient(self.x_test_mnist[:2], y)

        self.assertTrue(grads.shape == (2, 28, 28, 1))

        expected_gradients1 = np.asarray(
            [
                [-6.1982083e-03],
                [9.2188769e-04],
                [2.2715484e-03],
                [3.0439291e-03],
                [3.9350586e-03],
                [1.3214475e-03],
                [-1.9790903e-03],
                [-1.8616641e-03],
                [-1.7762191e-03],
                [-2.4208077e-03],
                [-2.1795963e-03],
                [-1.3475846e-03],
                [-1.7141351e-04],
                [5.3379539e-04],
                [6.1705662e-04],
                [9.1885449e-05],
                [-2.4936342e-04],
                [-7.8056828e-04],
                [-2.4509570e-04],
                [-1.3246380e-04],
                [-6.9344416e-04],
                [-2.8356430e-04],
                [1.1605137e-03],
                [2.7452575e-03],
                [2.9905243e-03],
                [2.2033940e-03],
                [1.7121597e-03],
                [8.4455572e-03],
            ]
        )
        np.testing.assert_array_almost_equal(grads[0, 0, :, :], expected_gradients1, decimal=2)

        expected_gradients2 = np.asarray(
            [
                [-8.14103708e-03],
                [-5.78497676e-03],
                [-1.93702651e-03],
                [-1.10854053e-04],
                [-3.13712610e-03],
                [-2.40660645e-03],
                [-2.33814842e-03],
                [-1.18874465e-04],
                [-8.61960289e-05],
                [-8.44302267e-05],
                [1.16928865e-03],
                [8.52172205e-04],
                [1.50172669e-03],
                [9.76039213e-04],
                [6.99639553e-04],
                [1.55441079e-03],
                [1.99828879e-03],
                [2.53868615e-03],
                [3.47398920e-03],
                [3.55495396e-03],
                [3.40546807e-03],
                [5.23657538e-03],
                [9.50821862e-03],
                [8.31787288e-03],
                [4.75075701e-03],
                [8.02019704e-03],
                [1.00337435e-02],
                [6.10247999e-03],
            ]
        )
        np.testing.assert_array_almost_equal(grads[1, :, 0, :], expected_gradients2, decimal=2)

    def test_loss_gradient_standard_format(self):
        # Create labels
        result_tf = self.obj_dec.predict(self.x_test_mnist[:2], standardise_output=False)
        result = self.obj_dec.predict(self.x_test_mnist[:2], standardise_output=True)

        from art.estimators.object_detection.utils import convert_tf_to_pt

        result_pt = convert_tf_to_pt(y=result_tf, height=self.x_test_mnist.shape[1], width=self.x_test_mnist.shape[2])
        for i in range(2):
            np.testing.assert_array_equal(result[i]["boxes"], result_pt[i]["boxes"])
            np.testing.assert_array_equal(result[i]["labels"], result_pt[i]["labels"])
            np.testing.assert_array_equal(result[i]["scores"], result_pt[i]["scores"])

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

        # Compute gradients
        grads = self.obj_dec.loss_gradient(self.x_test_mnist[:2], y, standardise_output=True)

        self.assertTrue(grads.shape == (2, 28, 28, 1))

        expected_gradients1 = np.asarray(
            [
                [-6.1982083e-03],
                [9.2188769e-04],
                [2.2715484e-03],
                [3.0439291e-03],
                [3.9350586e-03],
                [1.3214475e-03],
                [-1.9790903e-03],
                [-1.8616641e-03],
                [-1.7762191e-03],
                [-2.4208077e-03],
                [-2.1795963e-03],
                [-1.3475846e-03],
                [-1.7141351e-04],
                [5.3379539e-04],
                [6.1705662e-04],
                [9.1885449e-05],
                [-2.4936342e-04],
                [-7.8056828e-04],
                [-2.4509570e-04],
                [-1.3246380e-04],
                [-6.9344416e-04],
                [-2.8356430e-04],
                [1.1605137e-03],
                [2.7452575e-03],
                [2.9905243e-03],
                [2.2033940e-03],
                [1.7121597e-03],
                [8.4455572e-03],
            ]
        )
        np.testing.assert_array_almost_equal(grads[0, 0, :, :], expected_gradients1, decimal=2)

        expected_gradients2 = np.asarray(
            [
                [-8.14103708e-03],
                [-5.78497676e-03],
                [-1.93702651e-03],
                [-1.10854053e-04],
                [-3.13712610e-03],
                [-2.40660645e-03],
                [-2.33814842e-03],
                [-1.18874465e-04],
                [-8.61960289e-05],
                [-8.44302267e-05],
                [1.16928865e-03],
                [8.52172205e-04],
                [1.50172669e-03],
                [9.76039213e-04],
                [6.99639553e-04],
                [1.55441079e-03],
                [1.99828879e-03],
                [2.53868615e-03],
                [3.47398920e-03],
                [3.55495396e-03],
                [3.40546807e-03],
                [5.23657538e-03],
                [9.50821862e-03],
                [8.31787288e-03],
                [4.75075701e-03],
                [8.02019704e-03],
                [1.00337435e-02],
                [6.10247999e-03],
            ]
        )
        np.testing.assert_array_almost_equal(grads[1, :, 0, :], expected_gradients2, decimal=2)


if __name__ == "__main__":
    unittest.main()
