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

import numpy as np

from tests.utils import TestBase, master_seed


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
        cls.obj_detect = PyTorchFasterRCNN(
            clip_values=(0, 1), attack_losses=("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg")
        )

        # Create labels
        cls.x_test = np.repeat(cls.x_test_mnist[:2].astype(np.float32), repeats=3, axis=3)

        result = cls.obj_detect.predict(x=cls.x_test)

        cls.y_test = [
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

    def test_predict(self):
        result = self.obj_detect.predict(self.x_test_mnist.astype(np.float32))

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
        grads = self.obj_detect.loss_gradient(
            np.repeat(self.x_test_mnist[:2].astype(np.float32), repeats=3, axis=3), self.y_test
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

    def test_errors(self):
        from art.estimators.object_detection.pytorch_faster_rcnn import PyTorchFasterRCNN

        with self.assertRaises(ValueError):
            PyTorchFasterRCNN(
                clip_values=(1, 2),
                attack_losses=("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"),
            )

        with self.assertRaises(ValueError):
            PyTorchFasterRCNN(
                clip_values=(-1, 1),
                attack_losses=("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"),
            )

        with self.assertRaises(ValueError):
            PyTorchFasterRCNN(
                clip_values=(0, 1),
                attack_losses=("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"),
                preprocessing=(0, 1),
            )

        from art.defences.postprocessor.rounded import Rounded

        post_def = Rounded()
        with self.assertRaises(ValueError):
            PyTorchFasterRCNN(
                clip_values=(0, 1),
                attack_losses=("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"),
                postprocessing_defences=post_def,
            )

    def test_preprocessing_defences(self):
        from art.estimators.object_detection.pytorch_faster_rcnn import PyTorchFasterRCNN
        from art.defences.preprocessor.spatial_smoothing import SpatialSmoothing

        pre_def = SpatialSmoothing()

        frcnn = PyTorchFasterRCNN(
            clip_values=(0, 1),
            attack_losses=("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"),
            preprocessing_defences=pre_def,
        )

        # Create labels
        result = frcnn.predict(x=self.x_test)

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
        grads = frcnn.loss_gradient(x=self.x_test, y=y)

        self.assertTrue(grads.shape == (2, 28, 28, 3))

    def test_compute_losses(self):
        losses = self.obj_detect.compute_losses(x=self.x_test, y=self.y_test)
        self.assertTrue(len(losses) == 4)

    def test_compute_loss(self):
        from art.estimators.object_detection.pytorch_faster_rcnn import PyTorchFasterRCNN
        from art.defences.preprocessor.spatial_smoothing import SpatialSmoothing

        pre_def = SpatialSmoothing()

        frcnn = PyTorchFasterRCNN(
            clip_values=(0, 1),
            attack_losses=("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"),
            preprocessing_defences=pre_def,
        )

        # Create labels
        result = frcnn.predict(np.repeat(self.x_test_mnist[:2].astype(np.float32), repeats=3, axis=3))

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
        loss = frcnn.compute_loss(x=self.x_test, y=y)

        self.assertAlmostEqual(float(loss), 0.6324392, delta=0.01)

    def test_pgd(self):
        from art.attacks.evasion import ProjectedGradientDescent

        attack = ProjectedGradientDescent(estimator=self.obj_detect, max_iter=2)
        x_test_adv = attack.generate(x=self.x_test, y=self.y_test)
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, x_test_adv, self.x_test)


if __name__ == "__main__":
    unittest.main()
