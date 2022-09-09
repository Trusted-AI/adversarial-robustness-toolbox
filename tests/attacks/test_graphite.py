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
import unittest
import keras.backend as k
import numpy as np

from art.attacks.evasion import GRAPHITEBlackbox, GRAPHITEWhiteboxPyTorch
from art.estimators.estimator import BaseEstimator
from art.estimators.classification import ClassifierMixin

from tests.utils import TestBase
from tests.utils import get_image_classifier_tf, get_image_classifier_kr, get_image_classifier_pt

from tests.utils import master_seed
from tests.attacks.utils import backend_test_classifier_type_check_fail

logger = logging.getLogger(__name__)


class TestGRAPHITE(TestBase):
    """
    A unittest class for testing the GRAPHITE attack.
    """

    @classmethod
    def setUpClass(cls):
        master_seed(seed=1234, set_tensorflow=True, set_torch=True)
        super().setUpClass()

        cls.n_test = 2
        cls.x_test_init_mnist = cls.x_test_mnist[1 : cls.n_test + 1]
        cls.y_test_init_mnist = cls.y_test_mnist[1 : cls.n_test + 1]
        cls.x_test_mnist = cls.x_test_mnist[0 : cls.n_test]
        cls.y_test_mnist = cls.y_test_mnist[0 : cls.n_test]

    def setUp(self):
        master_seed(seed=1234, set_tensorflow=True, set_torch=True)
        super().setUp()

    def test_3_tensorflow_mnist(self):
        """
        First test with the TensorFlowClassifier.
        :return:
        """
        x_test_original = self.x_test_mnist.copy()

        # Build TensorFlowClassifier
        tfc, sess = get_image_classifier_tf()

        # First attack
        graphite = GRAPHITEBlackbox(
            classifier=tfc,
            noise_size=(28, 28),
            net_size=(28, 28),
            heatmap_mode="Target",
            num_xforms_mask=2,
            num_xforms_boost=10,
            rotation_range=(-5, 5),
            gamma_range=(1.0, 1.1),
            crop_percent_range=(-0.001, 0.001),
            off_x_range=(-0.001, 0.001),
            blur_kernels=[0],
        )
        params = {"y": self.y_test_init_mnist, "x_tar": self.x_test_init_mnist}
        x_test_adv = graphite.generate(self.x_test_mnist, **params)

        self.assertFalse((self.x_test_mnist == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())

        target = np.argmax(params["y"], axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        self.assertTrue((target == y_pred_adv).any())

        # Test the masking
        mask = np.random.binomial(n=1, p=0.5, size=np.prod(self.x_test_mnist.shape))
        mask = mask.reshape(self.x_test_mnist.shape)

        params.update(mask=mask)
        x_test_adv = graphite.generate(self.x_test_mnist, **params)
        mask_diff = (1 - mask) * (x_test_adv - self.x_test_mnist)
        self.assertAlmostEqual(float(np.max(np.abs(mask_diff))), 0.0, delta=0.00001)

        unmask_diff = mask * (x_test_adv - self.x_test_mnist)
        self.assertGreater(float(np.sum(np.abs(unmask_diff))), 0.0)

        # Test passing in points
        pts = np.zeros((4, 3, 1))
        pts[0, :, 0] = np.array([0.05, 0.05, 1])
        pts[1, :, 0] = np.array([0.05, 0.95, 1])
        pts[2, :, 0] = np.array([0.95, 0.05, 1])
        pts[3, :, 0] = np.array([0.95, 0.95, 1])

        params.update(pts=pts)
        x_test_adv = graphite.generate(self.x_test_mnist, **params)
        mask_diff = (1 - mask) * (x_test_adv - self.x_test_mnist)
        self.assertAlmostEqual(float(np.max(np.abs(mask_diff))), 0.0, delta=0.00001)

        unmask_diff = mask * (x_test_adv - self.x_test_mnist)
        self.assertGreater(float(np.sum(np.abs(unmask_diff))), 0.0)

        # Check that x_test has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - self.x_test_mnist))), 0.0, delta=0.00001)

        # Clean-up session
        if sess is not None:
            sess.close()

    def test_8_keras_mnist(self):
        """
        Second test with the KerasClassifier.
        :return:
        """
        x_test_original = self.x_test_mnist.copy()

        # Build KerasClassifier
        krc = get_image_classifier_kr()

        # First attack
        graphite = GRAPHITEBlackbox(
            classifier=krc,
            noise_size=(28, 28),
            net_size=(28, 28),
            heatmap_mode="Target",
            num_xforms_mask=2,
            num_xforms_boost=10,
            rotation_range=(-5, 5),
            gamma_range=(1.0, 1.1),
            crop_percent_range=(-0.001, 0.001),
            off_x_range=(-0.001, 0.001),
            blur_kernels=[0],
        )
        params = {"y": self.y_test_init_mnist, "x_tar": self.x_test_init_mnist}
        x_test_adv = graphite.generate(self.x_test_mnist, **params)

        self.assertFalse((self.x_test_mnist == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())

        target = np.argmax(params["y"], axis=1)
        y_pred_adv = np.argmax(krc.predict(x_test_adv), axis=1)
        self.assertTrue((target == y_pred_adv).any())

        # Test the masking
        mask = np.random.binomial(n=1, p=0.5, size=np.prod(self.x_test_mnist.shape))
        mask = mask.reshape(self.x_test_mnist.shape)

        params.update(mask=mask)
        x_test_adv = graphite.generate(self.x_test_mnist, **params)
        mask_diff = (1 - mask) * (x_test_adv - self.x_test_mnist)
        self.assertAlmostEqual(float(np.max(np.abs(mask_diff))), 0.0, delta=0.00001)

        unmask_diff = mask * (x_test_adv - self.x_test_mnist)
        self.assertGreater(float(np.sum(np.abs(unmask_diff))), 0.0)

        # Test passing in points
        pts = np.zeros((4, 3, 1))
        pts[0, :, 0] = np.array([0.05, 0.05, 1])
        pts[1, :, 0] = np.array([0.05, 0.95, 1])
        pts[2, :, 0] = np.array([0.95, 0.05, 1])
        pts[3, :, 0] = np.array([0.95, 0.95, 1])

        params.update(pts=pts)
        x_test_adv = graphite.generate(self.x_test_mnist, **params)
        mask_diff = (1 - mask) * (x_test_adv - self.x_test_mnist)
        self.assertAlmostEqual(float(np.max(np.abs(mask_diff))), 0.0, delta=0.00001)

        unmask_diff = mask * (x_test_adv - self.x_test_mnist)
        self.assertGreater(float(np.sum(np.abs(unmask_diff))), 0.0)

        # Check that x_test has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - self.x_test_mnist))), 0.0, delta=0.00001)

        # Clean-up session
        k.clear_session()

    def test_4_pytorch_classifier(self):
        """
        Third test with the PyTorchClassifier.
        :return:
        """
        x_test = np.transpose(self.x_test_mnist, (0, 3, 1, 2)).astype(np.float32)
        x_test_init = np.transpose(self.x_test_init_mnist, (0, 3, 1, 2)).astype(np.float32)
        x_test_original = x_test.copy()

        # Build PyTorchClassifier
        ptc = get_image_classifier_pt()

        # BLACKBOX
        # First attack
        graphite = GRAPHITEBlackbox(
            classifier=ptc,
            noise_size=(28, 28),
            net_size=(28, 28),
            heatmap_mode="Target",
            num_xforms_mask=2,
            num_xforms_boost=10,
            rotation_range=(-5, 5),
            gamma_range=(1.0, 1.1),
            crop_percent_range=(-0.001, 0.001),
            off_x_range=(-0.001, 0.001),
            blur_kernels=[0],
        )
        params = {"y": self.y_test_init_mnist, "x_tar": x_test_init}
        x_test_adv = graphite.generate(x_test, **params)

        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())

        target = np.argmax(params["y"], axis=1)
        y_pred_adv = np.argmax(ptc.predict(x_test_adv), axis=1)
        self.assertTrue((target == y_pred_adv).any())

        # Test the masking
        mask = np.random.binomial(n=1, p=0.5, size=np.prod(x_test.shape))
        mask = mask.reshape(x_test.shape)

        params.update(mask=mask)
        x_test_adv = graphite.generate(x_test, **params)
        mask_diff = (1 - mask) * (x_test_adv - x_test)
        self.assertAlmostEqual(float(np.max(np.abs(mask_diff))), 0.0, delta=0.00001)

        unmask_diff = mask * (x_test_adv - x_test)
        self.assertGreater(float(np.sum(np.abs(unmask_diff))), 0.0)

        # Test passing in points
        pts = np.zeros((4, 3, 1))
        pts[0, :, 0] = np.array([0.05, 0.05, 1])
        pts[1, :, 0] = np.array([0.05, 0.95, 1])
        pts[2, :, 0] = np.array([0.95, 0.05, 1])
        pts[3, :, 0] = np.array([0.95, 0.95, 1])

        params.update(pts=pts)
        x_test_adv = graphite.generate(x_test, **params)
        mask_diff = (1 - mask) * (x_test_adv - x_test)
        self.assertAlmostEqual(float(np.max(np.abs(mask_diff))), 0.0, delta=0.00001)

        unmask_diff = mask * (x_test_adv - x_test)
        self.assertGreater(float(np.sum(np.abs(unmask_diff))), 0.0)

        # Check that x_test has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test))), 0.0, delta=0.00001)

        # WHITEBOX
        # First attack
        graphite = GRAPHITEWhiteboxPyTorch(classifier=ptc, net_size=(28, 28), num_xforms=10)
        params = {"y": self.y_test_init_mnist}
        x_test_adv = graphite.generate(x_test, **params)

        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())

        target = np.argmax(params["y"], axis=1)
        y_pred_adv = np.argmax(ptc.predict(x_test_adv), axis=1)
        self.assertTrue((target == y_pred_adv).any())

        # Test the masking
        mask = np.random.binomial(n=1, p=0.5, size=np.prod(x_test.shape))
        mask = mask.reshape(x_test.shape)

        params.update(mask=mask)
        x_test_adv = graphite.generate(x_test, **params)
        mask_diff = (1 - mask) * (x_test_adv - x_test)
        self.assertAlmostEqual(float(np.max(np.abs(mask_diff))), 0.0, delta=0.00001)

        unmask_diff = mask * (x_test_adv - x_test)
        self.assertGreater(float(np.sum(np.abs(unmask_diff))), 0.0)

        # Test passing in points
        pts = np.zeros((4, 3, 1))
        pts[0, :, 0] = np.array([0.05, 0.05, 1])
        pts[1, :, 0] = np.array([0.05, 0.95, 1])
        pts[2, :, 0] = np.array([0.95, 0.05, 1])
        pts[3, :, 0] = np.array([0.95, 0.95, 1])

        params.update(pts=pts)
        x_test_adv = graphite.generate(x_test, **params)
        mask_diff = (1 - mask) * (x_test_adv - x_test)
        self.assertAlmostEqual(float(np.max(np.abs(mask_diff))), 0.0, delta=0.00001)

        unmask_diff = mask * (x_test_adv - x_test)
        self.assertGreater(float(np.sum(np.abs(unmask_diff))), 0.0)

        # Check that x_test has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test))), 0.0, delta=0.00001)

    def test_check_params(self):

        ptc = get_image_classifier_pt(from_logits=True)

        with self.assertRaises(ValueError):
            _ = GRAPHITEBlackbox(ptc, noise_size=(1, 1), net_size=(1, 1), heat_patch_size=(2, 2))
        with self.assertRaises(ValueError):
            _ = GRAPHITEBlackbox(ptc, noise_size=(28, 28), net_size=(28, 28), heat_patch_size=(0, 1))
        with self.assertRaises(ValueError):
            _ = GRAPHITEBlackbox(ptc, noise_size=(28, 28), net_size=(28, 28), heat_patch_size=(1.0, 0))
        with self.assertRaises(ValueError):
            _ = GRAPHITEBlackbox(ptc, noise_size=(28, 28), net_size=(28, 28), heat_patch_stride=(0, 1))
        with self.assertRaises(ValueError):
            _ = GRAPHITEBlackbox(ptc, noise_size=(28, 28), net_size=(28, 28), heat_patch_stride=(1.0, 0))
        with self.assertRaises(ValueError):
            _ = GRAPHITEBlackbox(ptc, noise_size=(28, 28), net_size=(28, 28), heatmap_mode="asdf")
        with self.assertRaises(ValueError):
            _ = GRAPHITEBlackbox(ptc, noise_size=(28, 28), net_size=(28, 28), tr_lo=-1)
        with self.assertRaises(ValueError):
            _ = GRAPHITEBlackbox(ptc, noise_size=(28, 28), net_size=(28, 28), tr_lo=1.1)
        with self.assertRaises(ValueError):
            _ = GRAPHITEBlackbox(ptc, noise_size=(28, 28), net_size=(28, 28), tr_hi=-1)
        with self.assertRaises(ValueError):
            _ = GRAPHITEBlackbox(ptc, noise_size=(28, 28), net_size=(28, 28), tr_hi=1.1)
        with self.assertRaises(ValueError):
            _ = GRAPHITEBlackbox(ptc, noise_size=(28, 28), net_size=(28, 28), num_xforms_mask=-1)
        with self.assertRaises(ValueError):
            _ = GRAPHITEBlackbox(ptc, noise_size=(28, 28), net_size=(28, 28), num_xforms_mask=1.0)
        with self.assertRaises(ValueError):
            _ = GRAPHITEBlackbox(ptc, noise_size=(28, 28), net_size=(28, 28), num_xforms_boost=-1)
        with self.assertRaises(ValueError):
            _ = GRAPHITEBlackbox(ptc, noise_size=(28, 28), net_size=(28, 28), num_xforms_boost=1.0)
        with self.assertRaises(ValueError):
            _ = GRAPHITEBlackbox(ptc, noise_size=(28, 28), net_size=(28, 28), num_boost_queries=0)
        with self.assertRaises(ValueError):
            _ = GRAPHITEBlackbox(ptc, noise_size=(28, 28), net_size=(28, 28), num_boost_queries=1.0)
        with self.assertRaises(ValueError):
            _ = GRAPHITEBlackbox(ptc, noise_size=(28, 28), net_size=(28, 28), rotation_range=(-90, 0))
        with self.assertRaises(ValueError):
            _ = GRAPHITEBlackbox(ptc, noise_size=(28, 28), net_size=(28, 28), rotation_range=(90, 0))
        with self.assertRaises(ValueError):
            _ = GRAPHITEBlackbox(ptc, noise_size=(28, 28), net_size=(28, 28), rotation_range=(1, 0))
        with self.assertRaises(ValueError):
            _ = GRAPHITEBlackbox(ptc, noise_size=(28, 28), net_size=(28, 28), dist_range=(-1, 1))
        with self.assertRaises(ValueError):
            _ = GRAPHITEBlackbox(ptc, noise_size=(28, 28), net_size=(28, 28), dist_range=(1, 0))
        with self.assertRaises(ValueError):
            _ = GRAPHITEBlackbox(ptc, noise_size=(28, 28), net_size=(28, 28), gamma_range=(0, 1))
        with self.assertRaises(ValueError):
            _ = GRAPHITEBlackbox(ptc, noise_size=(28, 28), net_size=(28, 28), gamma_range=(1, 0))
        with self.assertRaises(ValueError):
            _ = GRAPHITEBlackbox(ptc, noise_size=(28, 28), net_size=(28, 28), crop_percent_range=(1, 0))
        with self.assertRaises(ValueError):
            _ = GRAPHITEBlackbox(ptc, noise_size=(28, 28), net_size=(28, 28), off_x_range=(1, 0))
        with self.assertRaises(ValueError):
            _ = GRAPHITEBlackbox(ptc, noise_size=(28, 28), net_size=(28, 28), off_y_range=(1, 0))
        with self.assertRaises(ValueError):
            _ = GRAPHITEBlackbox(ptc, noise_size=(28, 28), net_size=(28, 28), blur_kernels=[-1])

        with self.assertRaises(ValueError):
            _ = GRAPHITEWhiteboxPyTorch(ptc, net_size=(28, 28), min_tr=-1)
        with self.assertRaises(ValueError):
            _ = GRAPHITEWhiteboxPyTorch(ptc, net_size=(28, 28), min_tr=1.1)
        with self.assertRaises(ValueError):
            _ = GRAPHITEWhiteboxPyTorch(ptc, net_size=(28, 28), num_xforms=-1)
        with self.assertRaises(ValueError):
            _ = GRAPHITEWhiteboxPyTorch(ptc, net_size=(28, 28), num_xforms=1.0)
        with self.assertRaises(ValueError):
            _ = GRAPHITEWhiteboxPyTorch(ptc, net_size=(28, 28), step_size=0)
        with self.assertRaises(ValueError):
            _ = GRAPHITEWhiteboxPyTorch(ptc, net_size=(28, 28), step_size=-1)
        with self.assertRaises(ValueError):
            _ = GRAPHITEWhiteboxPyTorch(ptc, net_size=(28, 28), first_steps=0)
        with self.assertRaises(ValueError):
            _ = GRAPHITEWhiteboxPyTorch(ptc, net_size=(28, 28), first_steps=1.0)
        with self.assertRaises(ValueError):
            _ = GRAPHITEWhiteboxPyTorch(ptc, net_size=(28, 28), steps=0)
        with self.assertRaises(ValueError):
            _ = GRAPHITEWhiteboxPyTorch(ptc, net_size=(28, 28), steps=1.0)
        with self.assertRaises(ValueError):
            _ = GRAPHITEWhiteboxPyTorch(ptc, net_size=(28, 28), patch_removal_size=0)
        with self.assertRaises(ValueError):
            _ = GRAPHITEWhiteboxPyTorch(ptc, net_size=(28, 28), num_patches_to_remove=0)
        with self.assertRaises(ValueError):
            _ = GRAPHITEWhiteboxPyTorch(ptc, net_size=(28, 28), num_patches_to_remove=1.0)
        with self.assertRaises(ValueError):
            _ = GRAPHITEWhiteboxPyTorch(ptc, net_size=(28, 28), rotation_range=(-90, 0))
        with self.assertRaises(ValueError):
            _ = GRAPHITEWhiteboxPyTorch(ptc, net_size=(28, 28), rotation_range=(90, 0))
        with self.assertRaises(ValueError):
            _ = GRAPHITEWhiteboxPyTorch(ptc, net_size=(28, 28), rotation_range=(1, 0))
        with self.assertRaises(ValueError):
            _ = GRAPHITEWhiteboxPyTorch(ptc, net_size=(28, 28), dist_range=(-1, 1))
        with self.assertRaises(ValueError):
            _ = GRAPHITEWhiteboxPyTorch(ptc, net_size=(28, 28), dist_range=(1, 0))
        with self.assertRaises(ValueError):
            _ = GRAPHITEWhiteboxPyTorch(ptc, net_size=(28, 28), gamma_range=(0, 1))
        with self.assertRaises(ValueError):
            _ = GRAPHITEWhiteboxPyTorch(ptc, net_size=(28, 28), gamma_range=(1, 0))
        with self.assertRaises(ValueError):
            _ = GRAPHITEWhiteboxPyTorch(ptc, net_size=(28, 28), crop_percent_range=(1, 0))
        with self.assertRaises(ValueError):
            _ = GRAPHITEWhiteboxPyTorch(ptc, net_size=(28, 28), off_x_range=(1, 0))
        with self.assertRaises(ValueError):
            _ = GRAPHITEWhiteboxPyTorch(ptc, net_size=(28, 28), off_y_range=(1, 0))
        with self.assertRaises(ValueError):
            _ = GRAPHITEWhiteboxPyTorch(ptc, net_size=(28, 28), blur_kernels=[-1])

    def test_1_classifier_type_check_fail(self):
        backend_test_classifier_type_check_fail(
            GRAPHITEBlackbox, [BaseEstimator, ClassifierMixin], noise_size=(28, 28), net_size=(28, 28)
        )
        backend_test_classifier_type_check_fail(
            GRAPHITEWhiteboxPyTorch, [BaseEstimator, ClassifierMixin], net_size=(28, 28)
        )


if __name__ == "__main__":
    unittest.main()
