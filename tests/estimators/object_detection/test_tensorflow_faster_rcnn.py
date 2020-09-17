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
            list(result.keys())
            == [
                "detection_boxes",
                "detection_scores",
                "detection_classes",
                "detection_multiclass_scores",
                "detection_anchor_indices",
                "num_detections",
                "raw_detection_boxes",
                "raw_detection_scores",
            ]
        )

        self.assertTrue(result["detection_boxes"].shape == (10, 300, 4))
        expected_detection_boxes = np.asarray([0.65566427, 0.0, 1.0, 0.9642794])
        np.testing.assert_array_almost_equal(result["detection_boxes"][0, 2, :], expected_detection_boxes, decimal=6)

        self.assertTrue(result["detection_scores"].shape == (10, 300))
        expected_detection_scores = np.asarray(
            [
                6.02739106e-04,
                3.72770795e-04,
                2.96768820e-04,
                2.12859799e-04,
                1.72638058e-04,
                1.51401327e-04,
                1.47289087e-04,
                1.25616702e-04,
                1.19876706e-04,
                1.06633954e-04,
            ]
        )
        np.testing.assert_array_almost_equal(result["detection_scores"][0, :10], expected_detection_scores, decimal=6)

        self.assertTrue(result["detection_classes"].shape == (10, 300))
        expected_detection_classes = np.asarray([81.0, 71.0, 66.0, 15.0, 63.0, 71.0, 66.0, 84.0, 64.0, 37.0])
        np.testing.assert_array_almost_equal(result["detection_classes"][0, :10], expected_detection_classes, decimal=6)

        self.assertTrue(result["detection_multiclass_scores"].shape == (10, 300, 91))
        expected_detection_multiclass_scores = np.asarray(
            [
                9.9915493e-01,
                1.5380951e-05,
                3.2381786e-06,
                2.3546692e-05,
                1.0490003e-06,
                2.9198272e-05,
                1.9808563e-06,
                6.0102529e-06,
                8.9344621e-06,
                2.8579292e-05,
            ]
        )
        np.testing.assert_array_almost_equal(
            result["detection_multiclass_scores"][0, 2, :10], expected_detection_multiclass_scores, decimal=6
        )

        self.assertTrue(result["detection_anchor_indices"].shape == (10, 300))
        expected_detection_anchor_indices = np.asarray([22.0, 22.0, 4.0, 35.0, 61.0, 49.0, 16.0, 22.0, 16.0, 61.0])
        np.testing.assert_array_almost_equal(
            result["detection_anchor_indices"][0, :10], expected_detection_anchor_indices, decimal=6
        )

        self.assertTrue(result["num_detections"].shape == (10,))
        expected_num_detections = np.asarray([300.0, 300.0, 300.0, 300.0, 300.0, 300.0, 300.0, 300.0, 300.0, 300.0])
        np.testing.assert_array_almost_equal(result["num_detections"], expected_num_detections, decimal=6)

        self.assertTrue(result["raw_detection_boxes"].shape == (10, 300, 4))
        expected_raw_detection_boxes = np.asarray([0.05784893, 0.05130966, 0.41411403, 0.95867515])
        np.testing.assert_array_almost_equal(
            result["raw_detection_boxes"][0, 2, :], expected_raw_detection_boxes, decimal=6
        )

        self.assertTrue(result["raw_detection_scores"].shape == (10, 300, 91))
        expected_raw_detection_scores = np.asarray(
            [
                9.9981636e-01,
                2.3866653e-06,
                2.2101715e-06,
                1.3920785e-05,
                9.3873712e-07,
                4.0993282e-06,
                3.3591269e-07,
                6.7879691e-06,
                2.8425752e-06,
                9.0685753e-06,
            ]
        )
        np.testing.assert_array_almost_equal(
            result["raw_detection_scores"][0, 2, :10], expected_raw_detection_scores, decimal=6
        )

    def test_loss_gradient(self):
        # Create labels
        result = self.obj_dec.predict(self.x_test_mnist[:2])

        groundtruth_boxes_list = [result["detection_boxes"][i] for i in range(2)]
        groundtruth_classes_list = [result["detection_classes"][i] for i in range(2)]
        groundtruth_weights_list = [np.ones_like(r) for r in groundtruth_classes_list]

        y = {
            "groundtruth_boxes_list": groundtruth_boxes_list,
            "groundtruth_classes_list": groundtruth_classes_list,
            "groundtruth_weights_list": groundtruth_weights_list,
        }

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


if __name__ == "__main__":
    unittest.main()
