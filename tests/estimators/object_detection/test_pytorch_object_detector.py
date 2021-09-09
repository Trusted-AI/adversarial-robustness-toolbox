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
import unittest

import numpy as np
import torchvision

from tests.utils import TestBase, master_seed


logger = logging.getLogger(__name__)


class TestPyTorchObjectDetector(TestBase):
    """
    This class tests the PyTorchObjectDetector object detector.
    """

    @classmethod
    def setUpClass(cls):
        master_seed(seed=1234, set_tensorflow=True)
        super().setUpClass()

        cls.n_test = 10
        cls.x_test_mnist = cls.x_test_mnist[0 : cls.n_test]
        cls.y_test_mnist = cls.y_test_mnist[0 : cls.n_test]

        # Only import if object detection module is available
        from art.estimators.object_detection.python_object_detector import PyTorchObjectDetector

        # Define object detectors
        model_1 = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True, progress=True, num_classes=91, pretrained_backbone=True
        )
        model_2 = torchvision.models.detection.maskrcnn_resnet50_fpn(
            pretrained=True, progress=True, num_classes=91, pretrained_backbone=True
        )

        cls.obj_detect_1 = PyTorchObjectDetector(
            model=model_1,
            clip_values=(0, 1),
            attack_losses=("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"),
        )

        cls.obj_detect_2 = PyTorchObjectDetector(
            model=model_2,
            clip_values=(0, 1),
            attack_losses=("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"),
        )

    def test_predict_1(self):
        result = self.obj_detect_1.predict(self.x_test_mnist.astype(np.float32))

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

    def test_predict_2(self):
        result = self.obj_detect_2.predict(self.x_test_mnist.astype(np.float32))

        self.assertTrue(
            list(result[0].keys())
            == [
                "boxes",
                "labels",
                "scores",
                "masks",
            ]
        )

        self.assertTrue(result[0]["boxes"].shape == (4, 4))
        expected_detection_boxes = np.asarray([8.62889, 11.735134, 16.353355, 27.565004])
        np.testing.assert_array_almost_equal(result[0]["boxes"][2, :], expected_detection_boxes, decimal=3)

        self.assertTrue(result[0]["scores"].shape == (4,))
        expected_detection_scores = np.asarray([0.45197296, 0.12707493, 0.082677, 0.05386855])
        np.testing.assert_array_almost_equal(result[0]["scores"][:10], expected_detection_scores, decimal=6)

        self.assertTrue(result[0]["labels"].shape == (4,))
        expected_detection_classes = np.asarray([72, 72, 1, 1])
        np.testing.assert_array_almost_equal(result[0]["labels"][:10], expected_detection_classes, decimal=6)

    def test_loss_gradient_1(self):
        # Create labels
        result = self.obj_detect_1.predict(np.repeat(self.x_test_mnist[:2].astype(np.float32), repeats=3, axis=3))

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
        grads = self.obj_detect_1.loss_gradient(
            np.repeat(self.x_test_mnist[:2].astype(np.float32), repeats=3, axis=3), y
        )

        self.assertTrue(grads.shape == (2, 28, 28, 3))

        expected_gradients1 = np.asarray(
            [
                [5.7717459e-04, 2.2427551e-03, 2.7338031e-03],
                [-5.4135895e-04, -6.8901619e-03, -5.3023611e-04],
                [1.7901474e-03, -6.0165934e-03, 1.2608932e-03],
                [2.2302025e-03, -4.1366839e-03, 8.1665488e-04],
                [5.0025941e-03, -2.0607577e-03, 1.3738470e-03],
                [6.7711552e-03, 2.4779334e-03, 3.2517519e-03],
                [7.7946498e-03, 3.8083603e-03, 3.9150072e-03],
                [6.2914360e-03, 3.2317259e-03, 2.4392023e-03],
                [6.8533504e-03, 4.6805567e-03, 2.1657508e-03],
                [6.4596147e-03, 1.6440222e-03, 2.1018654e-03],
                [7.3140049e-03, 4.9051084e-03, 2.1954530e-03],
                [7.3917350e-03, 5.3877393e-03, 2.5017208e-03],
                [7.1420427e-03, 4.5424267e-03, 1.7418499e-03],
                [7.6933270e-03, 7.0741987e-03, 1.3693030e-03],
                [7.9037091e-03, 8.1887292e-03, 1.0207348e-03],
                [4.7930530e-03, 1.2661386e-04, -2.0549579e-03],
                [4.7417181e-03, 1.1090005e-03, -2.1967045e-03],
                [4.0628687e-03, -1.0743369e-03, -2.7016401e-03],
                [4.1211918e-03, -9.3981961e-04, -3.3123612e-03],
                [2.7677750e-03, -2.0360684e-03, -2.4159362e-03],
                [1.5355040e-03, -2.3622375e-03, -2.2277990e-03],
                [-8.2429928e-05, -2.7951330e-03, -2.4791150e-03],
                [8.6106811e-05, -1.1048347e-03, -1.8214922e-03],
                [1.3870616e-03, 1.4906849e-03, -3.1876419e-04],
                [1.1308161e-03, 6.2550785e-04, 7.9436734e-04],
                [-1.0549244e-03, -2.1480548e-03, -8.4300683e-04],
                [7.4692059e-04, 6.3713623e-04, -2.2322751e-04],
                [1.6337358e-04, -1.2138729e-03, -8.6526090e-04],
            ]
        )
        np.testing.assert_array_almost_equal(grads[0, 0, :, :], expected_gradients1, decimal=2)

        expected_gradients2 = np.asarray(
            [
                [8.09008547e-04, 1.46970048e-03, 2.30784086e-03],
                [1.57560175e-03, -3.95192811e-03, -3.42682266e-04],
                [1.17776252e-03, -4.75858618e-03, -1.83509255e-03],
                [-3.62795522e-03, -7.03671249e-03, -2.61869049e-03],
                [-5.65498043e-03, -9.36302636e-03, -2.72479979e-03],
                [-6.13390049e-03, -1.91371012e-02, -8.64498038e-03],
                [4.13261494e-03, -5.83548984e-03, -5.41773997e-03],
                [2.10555550e-02, 1.75252277e-02, 1.19110784e-02],
                [2.86780880e-03, -2.02223212e-02, 4.42323042e-03],
                [1.66129377e-02, 4.57757805e-03, 3.99308838e-03],
                [-5.31449541e-03, -2.39533130e-02, -1.50507865e-02],
                [-1.55420639e-02, -6.57757046e-03, -1.95033997e-02],
                [-1.71425883e-02, -8.82681739e-03, -1.03681823e-02],
                [-1.52608315e-02, -2.59394385e-02, -8.74401908e-03],
                [-1.98556799e-02, -4.51070368e-02, -2.01500412e-02],
                [-1.76412370e-02, -4.00045775e-02, -2.76774243e-02],
                [-3.39970365e-02, -5.27175590e-02, -2.48762686e-02],
                [-1.01934038e-02, -1.34583283e-02, 2.92114611e-03],
                [9.27460939e-03, -1.07238982e-02, 1.69319492e-02],
                [1.32648731e-02, 7.15299882e-03, 1.81243364e-02],
                [1.04831355e-02, 3.29193124e-03, 1.09448479e-02],
                [5.21936268e-03, -1.08520268e-03, 4.44627739e-03],
                [4.43769246e-03, 1.22211361e-03, 1.76453649e-03],
                [2.82945228e-03, 1.39565568e-03, 5.05451404e-04],
                [6.36306650e-04, -7.02011574e-04, 8.36413165e-05],
                [2.80080014e-04, -9.24700813e-04, -6.42473227e-04],
                [1.44194404e-03, 9.39335907e-04, -1.95080182e-04],
                [1.05228636e-03, -4.52511711e-03, -5.74906298e-04],
            ]
        )
        np.testing.assert_array_almost_equal(grads[1, 0, :, :], expected_gradients2, decimal=2)

    def test_loss_gradient_2(self):
        # Create labels
        result = self.obj_detect_2.predict(np.repeat(self.x_test_mnist[:2].astype(np.float32), repeats=3, axis=3))

        y = [
            {
                "boxes": result[0]["boxes"],
                "labels": result[0]["labels"],
                "scores": np.ones_like(result[0]["labels"]),
                "masks": result[0]["masks"],
            },
            {
                "boxes": result[1]["boxes"],
                "labels": result[1]["labels"],
                "scores": np.ones_like(result[1]["labels"]),
                "masks": result[0]["masks"],
            },
        ]

        # Compute gradients
        grads = self.obj_detect_2.loss_gradient(
            np.repeat(self.x_test_mnist[:2].astype(np.float32), repeats=3, axis=3), y
        )

        self.assertTrue(grads.shape == (2, 28, 28, 3))

        expected_gradients1 = np.asarray(
            [
                [2.4262650e-03, 1.3761593e-02, 2.0420302e-03],
                [-7.2503160e-03, -1.1365028e-02, -6.8405606e-03],
                [2.6330010e-03, 1.6619957e-03, 4.5232830e-04],
                [6.1354763e-03, 1.0420770e-02, 7.9455890e-04],
                [-6.1245268e-04, 3.1636789e-04, -1.2903288e-03],
                [-7.6963343e-03, -1.0485162e-02, -4.7575748e-03],
                [-8.2736937e-03, -1.3176411e-02, -3.3130425e-03],
                [-4.9785916e-03, -8.5923560e-03, -3.6474820e-03],
                [2.3811011e-04, -1.6637128e-03, -3.3566195e-03],
                [-2.7071193e-03, -3.9336397e-03, -5.1789372e-03],
                [-3.7725787e-03, -5.5729761e-03, -5.8548329e-03],
                [-3.2943888e-03, -9.7764274e-03, -3.6585492e-03],
                [1.1066039e-02, 1.5334149e-02, 3.8445129e-03],
                [6.3294196e-03, 5.0334912e-03, -8.2077931e-05],
                [9.6800774e-03, 1.4249773e-02, -2.6656120e-05],
                [5.5468981e-03, 8.3901342e-03, -2.7621232e-03],
                [5.9738639e-03, 9.2917113e-03, -3.7981002e-03],
                [2.0094446e-03, 1.7389947e-03, -6.9772275e-03],
                [2.6797794e-04, -3.0170629e-06, -5.1656305e-03],
                [-3.4391661e-03, -7.3434487e-03, -8.2398113e-03],
                [2.8307869e-03, 2.0932304e-03, -3.4084707e-03],
                [2.7525092e-03, 2.2778441e-03, -4.7691381e-03],
                [1.4954894e-04, 1.0904951e-03, -4.2127748e-03],
                [-3.0063807e-03, -8.4828836e-04, -5.9513715e-03],
                [1.3739272e-04, 5.4094465e-03, -1.5331247e-03],
                [-1.2525261e-02, -1.9251918e-02, -5.8580134e-03],
                [-5.2041751e-03, -2.4381487e-03, -7.2076335e-03],
                [-1.3689174e-02, -1.6345499e-02, -1.1745064e-02],
            ]
        )
        np.testing.assert_array_almost_equal(grads[0, 0, :, :], expected_gradients1, decimal=2)

        expected_gradients2 = np.asarray(
            [
                [-3.7956156e-04, -1.7838631e-03, -4.5084467e-04],
                [5.8878929e-04, 7.1982906e-04, 4.4436965e-04],
                [2.7766402e-05, -3.0779847e-04, 4.0400824e-05],
                [-1.5373411e-05, -6.2050577e-04, -8.7568245e-05],
                [-1.3192007e-04, -3.8452109e-04, -4.3875689e-04],
                [-9.3099382e-04, -2.0350779e-03, -5.0653273e-04],
                [-5.3110311e-04, -1.3327714e-03, -1.6413847e-04],
                [-1.0081082e-04, -7.5704377e-04, -2.4673308e-04],
                [-2.9729211e-04, -9.6929166e-04, -2.0765550e-04],
                [-1.6119743e-03, -2.8538222e-03, -3.6886099e-04],
                [3.5074059e-04, 1.0773649e-03, 1.0213704e-03],
                [6.1042252e-04, 1.7703733e-03, 7.8561704e-04],
                [4.1100761e-04, 9.0549269e-04, 5.9309450e-04],
                [7.3774799e-04, 2.3988779e-03, 6.7986397e-04],
                [-1.5267699e-03, -5.2355917e-04, -5.2752614e-04],
                [-4.4806483e-03, -9.9256616e-03, -1.2656704e-03],
                [8.0325771e-03, 1.0095691e-02, 5.2176970e-03],
                [1.0284576e-02, 1.0698810e-02, 7.4157645e-03],
                [1.4244532e-02, 2.0294065e-02, 8.1336638e-03],
                [9.1136564e-03, 8.7427013e-03, 3.5043361e-03],
                [1.2690859e-02, 1.6872523e-02, 3.6197752e-03],
                [6.3505820e-03, 8.4563270e-03, -5.4673419e-05],
                [-6.8734086e-04, -9.6730480e-04, -2.0344146e-03],
                [1.6517416e-03, 4.0146364e-03, -9.3345798e-04],
                [-3.8086440e-04, 6.1084545e-04, -9.7791851e-04],
                [-1.7071007e-03, -2.5405220e-03, -7.7641435e-04],
                [8.1310543e-04, 6.2752562e-04, 4.1344954e-04],
                [3.4051610e-04, 7.4358581e-04, -8.9037686e-04],
            ]
        )
        np.testing.assert_array_almost_equal(grads[1, 0, :, :], expected_gradients2, decimal=2)


if __name__ == "__main__":
    unittest.main()
