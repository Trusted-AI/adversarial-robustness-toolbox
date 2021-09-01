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

from art.attacks.evasion.simba import SimBA
from art.estimators.estimator import BaseEstimator, NeuralNetworkMixin
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import get_labels_np_array

from tests.utils import TestBase
from tests.utils import get_image_classifier_tf, get_image_classifier_kr, get_image_classifier_pt
from tests.attacks.utils import backend_test_classifier_type_check_fail

logger = logging.getLogger(__name__)


class TestSimBA(TestBase):
    """
    A unittest class for testing the Simple Black-box Adversarial Attacks (SimBA).

    This module tests SimBA.
    Note: SimBA runs only in Keras and TensorFlow (not in PyTorch)
    This is due to the channel first format in PyTorch.

    | Paper link: https://arxiv.org/abs/1905.07121
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.n_test = 2
        cls.x_test_mnist = cls.x_test_mnist[0 : cls.n_test]
        cls.y_test_mnist = cls.y_test_mnist[0 : cls.n_test]

    def test_5_keras_mnist(self):
        """
        Test with the KerasClassifier. (Untargeted Attack)
        :return:
        """
        classifier = get_image_classifier_kr()
        self._test_attack(classifier, self.x_test_mnist, self.y_test_mnist, False)

    def test_2_tensorflow_mnist(self):
        """
        Test with the TensorFlowClassifier. (Untargeted Attack)
        :return:
        """
        classifier, sess = get_image_classifier_tf()
        self._test_attack(classifier, self.x_test_mnist, self.y_test_mnist, False)

    def test_3_pytorch_mnist(self):
        """
        Test with the PyTorchClassifier. (Untargeted Attack)
        :return:
        """
        x_test = np.reshape(self.x_test_mnist, (self.x_test_mnist.shape[0], 1, 28, 28)).astype(np.float32)
        classifier = get_image_classifier_pt()
        self._test_attack(classifier, x_test, self.y_test_mnist, False)

    def test_6_keras_mnist_targeted(self):
        """
        Test with the KerasClassifier. (Targeted Attack)
        :return:
        """
        classifier = get_image_classifier_kr()
        self._test_attack(classifier, self.x_test_mnist, self.y_test_mnist, True)

    def test_2_tensorflow_mnist_targeted(self):
        """
        Test with the TensorFlowClassifier. (Targeted Attack)
        :return:
        """
        classifier, sess = get_image_classifier_tf()
        self._test_attack(classifier, self.x_test_mnist, self.y_test_mnist, True)

    # SimBA is not available for PyTorch
    def test_4_pytorch_mnist_targeted(self):
        """
        Test with the PyTorchClassifier. (Targeted Attack)
        :return:
        """
        x_test = np.reshape(self.x_test_mnist, (self.x_test_mnist.shape[0], 1, 28, 28)).astype(np.float32)
        classifier = get_image_classifier_pt()
        self._test_attack(classifier, x_test, self.y_test_mnist, True)

    def _test_attack(self, classifier, x_test, y_test, targeted):
        """
        Test with SimBA
        :return:
        """
        x_test_original = x_test.copy()

        # set the targeted label
        if targeted:
            y_target = np.zeros(10)
            y_target[8] = 1.0

        #######
        # dct #
        #######

        df = SimBA(classifier, attack="dct", targeted=targeted)

        x_i = x_test_original[0][None, ...]
        if targeted:
            x_test_adv = df.generate(x_i, y=y_target.reshape(1, 10))
        else:
            x_test_adv = df.generate(x_i)

        for i in range(1, len(x_test_original)):
            x_i = x_test_original[i][None, ...]
            if targeted:
                tmp_x_test_adv = df.generate(x_i, y=y_target.reshape(1, 10))
                x_test_adv = np.concatenate([x_test_adv, tmp_x_test_adv])
            else:
                tmp_x_test_adv = df.generate(x_i)
                x_test_adv = np.concatenate([x_test_adv, tmp_x_test_adv])

        self.assertFalse((x_test == x_test_adv).all())
        self.assertFalse((0.0 == x_test_adv).all())

        y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        self.assertFalse((y_test == y_pred).all())

        # Check that x_test has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test))), 0.0, delta=0.00001)

        ######
        # px #
        ######
        df_px = SimBA(classifier, attack="px", targeted=targeted)

        x_i = x_test_original[0][None, ...]
        if targeted:
            x_test_adv = df_px.generate(x_i, y=y_target.reshape(1, 10))
        else:
            x_test_adv = df_px.generate(x_i)

        for i in range(1, len(x_test_original)):
            x_i = x_test_original[i][None, ...]
            if targeted:
                tmp_x_test_adv = df_px.generate(x_i, y=y_target.reshape(1, 10))
                x_test_adv = np.concatenate([x_test_adv, tmp_x_test_adv])
            else:
                tmp_x_test_adv = df_px.generate(x_i)
                x_test_adv = np.concatenate([x_test_adv, tmp_x_test_adv])

        self.assertFalse((x_test == x_test_adv).all())
        self.assertFalse((0.0 == x_test_adv).all())

        y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        self.assertFalse((y_test == y_pred).all())

        # Check that x_test has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test))), 0.0, delta=0.00001)

        #############
        # px - diag #
        #############
        df_px = SimBA(classifier, attack="px", targeted=targeted, order="diag")

        x_i = x_test_original[0][None, ...]
        if targeted:
            x_test_adv = df_px.generate(x_i, y=y_target.reshape(1, 10))
        else:
            x_test_adv = df_px.generate(x_i)

        for i in range(1, len(x_test_original)):
            x_i = x_test_original[i][None, ...]
            if targeted:
                tmp_x_test_adv = df_px.generate(x_i, y=y_target.reshape(1, 10))
                x_test_adv = np.concatenate([x_test_adv, tmp_x_test_adv])
            else:
                tmp_x_test_adv = df_px.generate(x_i)
                x_test_adv = np.concatenate([x_test_adv, tmp_x_test_adv])

        self.assertFalse((x_test == x_test_adv).all())
        self.assertFalse((0.0 == x_test_adv).all())

        y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        self.assertFalse((y_test == y_pred).all())

        # Check that x_test has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test))), 0.0, delta=0.00001)

    def test_check_params(self):

        ptc = get_image_classifier_pt(from_logits=True)

        with self.assertRaises(ValueError):
            _ = SimBA(ptc, max_iter=1.0)
        with self.assertRaises(ValueError):
            _ = SimBA(ptc, max_iter=-1)

        with self.assertRaises(ValueError):
            _ = SimBA(ptc, epsilon=-1)

        with self.assertRaises(ValueError):
            _ = SimBA(ptc, batch_size=2)

        with self.assertRaises(ValueError):
            _ = SimBA(ptc, stride=1.0)
        with self.assertRaises(ValueError):
            _ = SimBA(ptc, stride=-1)

        with self.assertRaises(ValueError):
            _ = SimBA(ptc, freq_dim=1.0)
        with self.assertRaises(ValueError):
            _ = SimBA(ptc, freq_dim=-1)

        with self.assertRaises(ValueError):
            _ = SimBA(ptc, order="test")

        with self.assertRaises(ValueError):
            _ = SimBA(ptc, attack="test")

        with self.assertRaises(ValueError):
            _ = SimBA(ptc, targeted="test")

    def test_1_classifier_type_check_fail(self):
        backend_test_classifier_type_check_fail(SimBA, (BaseEstimator, ClassifierMixin, NeuralNetworkMixin))


if __name__ == "__main__":
    unittest.main()
