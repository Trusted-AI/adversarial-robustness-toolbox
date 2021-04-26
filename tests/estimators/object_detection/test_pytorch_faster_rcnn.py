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

from tests.utils import TestBase, master_seed

object_detection_spec = importlib.util.find_spec("object_detection")
object_detection_found = object_detection_spec is not None

logger = logging.getLogger(__name__)


class TestPyTorchFasterRCNN(TestBase):
    """
    This class tests the PyTorchFasterRCNN object detector.
    """

    @classmethod
    def setUpClass(cls):
        master_seed(seed=1234, set_tensorflow=True)
        super().setUpClass()

        cls.n_test = 10
        cls.x_test_mnist = cls.x_test_mnist[0 : cls.n_test]
        cls.y_test_mnist = cls.y_test_mnist[0 : cls.n_test]

        # Only import if object detection module is available
        from art.estimators.object_detection.pytorch_faster_rcnn import PyTorchFasterRCNN

        # Define object detector
        cls.obj_dec = PyTorchFasterRCNN(
            clip_values=(0, 1), attack_losses=("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg")
        )

    def test_predict(self):
        result = self.obj_dec.predict(self.x_test_mnist.astype(np.float32))

        self.assertTrue(
            list(result[0].keys())
            == [
                "boxes",
                "labels",
                "scores",
            ]
        )

        self.assertTrue(result[0]["boxes"].shape == (7, 4))
        expected_detection_boxes = np.asarray([4.4017954, 6.3090835, 22.128296, 27.570665])
        np.testing.assert_array_almost_equal(result[0]["boxes"][2, :], expected_detection_boxes, decimal=3)

        self.assertTrue(result[0]["scores"].shape == (7,))
        expected_detection_scores = np.asarray(
            [0.3314798, 0.14125851, 0.13928168, 0.0996184, 0.08550017, 0.06690315, 0.05359321]
        )
        np.testing.assert_array_almost_equal(result[0]["scores"][:10], expected_detection_scores, decimal=6)

        self.assertTrue(result[0]["labels"].shape == (7,))
        expected_detection_classes = np.asarray([72, 79, 1, 72, 78, 72, 82])
        np.testing.assert_array_almost_equal(result[0]["labels"][:10], expected_detection_classes, decimal=6)

    def test_loss_gradient(self):
        # Create labels
        result = self.obj_dec.predict(np.repeat(self.x_test_mnist[:2].astype(np.float32), repeats=3, axis=3))

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
        grads = self.obj_dec.loss_gradient(np.repeat(self.x_test_mnist[:2].astype(np.float32), repeats=3, axis=3), y)

        self.assertTrue(grads.shape == (2, 28, 28, 3))

        expected_gradients1 = np.asarray(
            [
                [2.8594528e-04, 1.1215784e-03, 1.3653790e-03],
                [-2.7976825e-04, -3.4371731e-03, -2.6696315e-04],
                [8.8457583e-04, -2.9963013e-03, 6.3259545e-04],
                [1.1151490e-03, -2.0329244e-03, 4.2273162e-04],
                [2.4935475e-03, -1.0062727e-03, 7.1127195e-04],
                [3.3927602e-03, 1.2877580e-03, 1.6526978e-03],
                [3.9102370e-03, 1.9588615e-03, 1.9851099e-03],
                [3.1604825e-03, 1.6455721e-03, 1.2356425e-03],
                [3.3993176e-03, 2.2629590e-03, 1.0956719e-03],
                [3.2812082e-03, 8.9771382e-04, 1.1293223e-03],
                [3.6680547e-03, 2.3926175e-03, 1.1095805e-03],
                [3.5687620e-03, 2.2955453e-03, 1.0373374e-03],
                [3.6297247e-03, 2.2692063e-03, 8.3115656e-04],
                [3.9731734e-03, 3.6677248e-03, 8.7138364e-04],
                [4.2218431e-03, 4.4516954e-03, 7.7594118e-04],
                [2.6073644e-03, 4.4606114e-04, -8.4252883e-04],
                [2.4440261e-03, 8.8261464e-04, -9.0436038e-04],
                [2.1058477e-03, -9.4505522e-05, -1.1648482e-03],
                [2.0631582e-03, -3.6456017e-04, -1.4869515e-03],
                [1.5254768e-03, -7.1617024e-04, -9.9803146e-04],
                [8.3453697e-04, -1.1078444e-03, -8.9848990e-04],
                [9.5016156e-05, -1.2708809e-03, -1.0779313e-03],
                [1.1510249e-04, -5.3439446e-04, -8.3905994e-04],
                [7.0049067e-04, 7.0042303e-04, -1.7771695e-04],
                [5.1793398e-04, 3.0990795e-04, 3.4147498e-04],
                [-4.9308245e-04, -9.5870328e-04, -4.0385226e-04],
                [5.6163035e-04, 6.5357372e-04, -2.2048228e-04],
                [6.4600166e-04, -2.4371590e-03, -4.7532227e-04],
            ]
        )
        np.testing.assert_array_almost_equal(grads[0, 0, :, :], expected_gradients1, decimal=2)

        expected_gradients2 = np.asarray(
            [
                [4.0832075e-04, 7.4385171e-04, 1.1510107e-03],
                [7.8413740e-04, -1.9708287e-03, -1.7503211e-04],
                [5.8207760e-04, -2.3749834e-03, -9.1943046e-04],
                [-1.8109266e-03, -3.5117613e-03, -1.3074722e-03],
                [-2.8258380e-03, -4.6697278e-03, -1.3580204e-03],
                [-3.0620105e-03, -9.5577771e-03, -4.3166061e-03],
                [2.0600453e-03, -2.9108913e-03, -2.7060930e-03],
                [1.0508234e-02, 8.7555489e-03, 5.9620179e-03],
                [1.3998155e-03, -1.0158662e-02, 2.1992160e-03],
                [8.2888827e-03, 2.2636773e-03, 1.9773452e-03],
                [-2.6896396e-03, -1.2026494e-02, -7.5484989e-03],
                [-7.8031109e-03, -3.3438604e-03, -9.7707417e-03],
                [-8.5932901e-03, -4.4458969e-03, -5.1951385e-03],
                [-7.6420107e-03, -1.2982649e-02, -4.3780585e-03],
                [-9.9381953e-03, -2.2575803e-02, -1.0084845e-02],
                [-8.8340146e-03, -2.0025160e-02, -1.3847021e-02],
                [-1.7016266e-02, -2.6385436e-02, -1.2448472e-02],
                [-5.0954064e-03, -6.7400285e-03, 1.4573098e-03],
                [4.6212324e-03, -5.3683561e-03, 8.4736003e-03],
                [6.6042086e-03, 3.5803786e-03, 9.0765953e-03],
                [5.2064704e-03, 1.6609817e-03, 5.4876884e-03],
                [2.5890339e-03, -5.2395504e-04, 2.2445712e-03],
                [2.2006067e-03, 6.4179802e-04, 9.0203912e-04],
                [1.4054405e-03, 7.2372350e-04, 2.6953171e-04],
                [3.1803478e-04, -3.3220908e-04, 5.3212192e-05],
                [1.3612930e-04, -4.5914305e-04, -3.1649336e-04],
                [7.2048831e-04, 4.7292167e-04, -9.7459262e-05],
                [5.2783103e-04, -2.2626675e-03, -2.8743077e-04],
            ]
        )
        np.testing.assert_array_almost_equal(grads[1, 0, :, :], expected_gradients2, decimal=2)


if __name__ == "__main__":
    unittest.main()
