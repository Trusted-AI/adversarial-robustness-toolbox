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
"""
This module tests the Pixel Attack.
The Pixel Attack is a generalisation of One Pixel Attack.

| One Pixel Attack Paper link:
    https://ieeexplore.ieee.org/abstract/document/8601309/citations#citations
    (arXiv link: https://arxiv.org/pdf/1710.08864.pdf)
| Pixel Attack Paper link:
    https://arxiv.org/abs/1906.06026
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

import numpy as np

from art.attacks.evasion.pixel_threshold import PixelAttack
from art.estimators.estimator import BaseEstimator, NeuralNetworkMixin
from art.estimators.classification.classifier import ClassifierMixin

from tests.utils import TestBase
from tests.utils import get_image_classifier_tf, get_image_classifier_pt  # , get_image_classifier_kr
from tests.attacks.utils import backend_test_classifier_type_check_fail

logger = logging.getLogger(__name__)


class TestPixelAttack(TestBase):
    """
    A unittest class for testing the Pixel Attack.

    This module tests the Pixel Attack.
    The Pixel Attack is a generalisation of One Pixel Attack.

    | One Pixel Attack Paper link:
        https://ieeexplore.ieee.org/abstract/document/8601309/citations#citations
        (arXiv link: https://arxiv.org/pdf/1710.08864.pdf)
    | Pixel Attack Paper link:
        https://arxiv.org/abs/1906.06026
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.n_test = 2
        cls.x_test_mnist = cls.x_test_mnist[0 : cls.n_test]
        cls.y_test_mnist = cls.y_test_mnist[0 : cls.n_test]

    # def test_6_keras_mnist(self):
    #     """
    #     Test with the KerasClassifier. (Untargeted Attack)
    #     :return:
    #     """
    #
    #     classifier = get_image_classifier_kr()
    #     self._test_attack(classifier, self.x_test_mnist, self.y_test_mnist, False)

    # def test_2_tensorflow_mnist(self):
    #     """
    #     Test with the TensorFlowClassifier. (Untargeted Attack)
    #     :return:
    #     """
    #     classifier, sess = get_image_classifier_tf()
    #     self._test_attack(classifier, self.x_test_mnist, self.y_test_mnist, False)

    def test_4_pytorch_mnist(self):
        """
        Test with the PyTorchClassifier. (Untargeted Attack)
        :return:
        """
        x_test = np.reshape(self.x_test_mnist, (self.x_test_mnist.shape[0], 1, 28, 28)).astype(np.float32)
        classifier = get_image_classifier_pt()
        self._test_attack(classifier, x_test, self.y_test_mnist, False)

    def test_8_pytorch_mnist_single_sample(self):
        """
        Test with the PyTorchClassifier on a single sample. (Untargeted Attack)
        :return:
        """
        x_test = np.reshape(self.x_test_mnist[1], (1, 1, 28, 28)).astype(np.float32)
        classifier = get_image_classifier_pt()
        self._test_attack(classifier, x_test, self.y_test_mnist[[1]], False)

    # def test_7_keras_mnist_targeted(self):
    #     """
    #     Test with the KerasClassifier. (Targeted Attack)
    #     :return:
    #     """
    #     classifier = get_image_classifier_kr()
    #     self._test_attack(classifier, self.x_test_mnist, self.y_test_mnist, True)

    def test_3_tensorflow_mnist_targeted(self):
        """
        Test with the TensorFlowClassifier. (Targeted Attack)
        :return:
        """
        classifier, sess = get_image_classifier_tf()
        self._test_attack(classifier, self.x_test_mnist, self.y_test_mnist, True)

    # def test_5_pytorch_mnist_targeted(self):
    #     """
    #     Test with the PyTorchClassifier. (Targeted Attack)
    #     :return:
    #     """
    #     x_test = np.reshape(self.x_test_mnist, (self.x_test_mnist.shape[0], 1, 28, 28)).astype(np.float32)
    #     classifier = get_image_classifier_pt()
    #     self._test_attack(classifier, x_test, self.y_test_mnist, True)

    def _test_attack(self, classifier, x_test, y_test, targeted):
        """
        Test with the Pixel Attack
        :return:
        """
        x_test_original = x_test.copy()

        if targeted:
            # Generate random target classes
            class_y_test = np.argmax(y_test, axis=1)
            nb_classes = np.unique(class_y_test).shape[0]
            targets = np.random.randint(nb_classes, size=self.n_test)
            for i in range(self.n_test):
                if class_y_test[i] == targets[i]:
                    targets[i] -= 1
        else:
            targets = y_test

        for th in [None, 128]:
            for es in [0, 1]:
                df = PixelAttack(classifier, th=th, es=es, max_iter=20, targeted=targeted, verbose=False)
                x_test_adv = df.generate(x_test_original, targets)

                np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, x_test, x_test_adv)
                self.assertFalse((0.0 == x_test_adv).all())

        # Check that x_test has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test))), 0.0, delta=0.00001)

    def test_check_params(self):

        ptc = get_image_classifier_pt(from_logits=True)

        with self.assertRaises(ValueError):
            _ = PixelAttack(ptc, th=-1)

        with self.assertRaises(ValueError):
            _ = PixelAttack(ptc, es=1.0)

        with self.assertRaises(ValueError):
            _ = PixelAttack(ptc, targeted="true")

        with self.assertRaises(ValueError):
            _ = PixelAttack(ptc, verbose="true")

        with self.assertRaises(ValueError):
            ptc._clip_values = None
            _ = PixelAttack(ptc)

    def test_1_classifier_type_check_fail(self):
        backend_test_classifier_type_check_fail(PixelAttack, [BaseEstimator, NeuralNetworkMixin, ClassifierMixin])


if __name__ == "__main__":
    unittest.main()
