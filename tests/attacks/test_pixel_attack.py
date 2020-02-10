# MIT License
#
# Copyright (C) IBM Corporation 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the  rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
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

from tests.utils import TestBase
from tests.utils import get_classifier_tf, get_classifier_kr, get_classifier_pt

from art.attacks import PixelAttack
from art.utils import get_labels_np_array

logger = logging.getLogger(__name__)


class TestPixelAttack(TestBase):
    """
    TODO: Write Comment
    """

    @classmethod
    def setUpClass(cls):
        """
        TODO: Write Comment
        """
        super().setUpClass()

        cls.n_test = 2
        cls.x_test_mnist = cls.x_test_mnist[0:cls.n_test]
        cls.y_test_mnist = cls.y_test_mnist[0:cls.n_test]

    def test_keras_mnist(self):
        """
        TODO: Write Comment
        """
        classifier = get_classifier_kr()
        self._test_attack(
            classifier,
            self.x_test_mnist,
            self.y_test_mnist,
            False)

    def test_tensorflow_mnist(self):
        """
        TODO: Write Comment
        """
        classifier, sess = get_classifier_tf()
        self._test_attack(
            classifier,
            self.x_test_mnist,
            self.y_test_mnist,
            False)

    def test_pytorch_mnist(self):
        """
        TODO: Write Comment
        """
        x_test = np.swapaxes(self.x_test_mnist, 1, 3).astype(np.float32)
        classifier = get_classifier_pt()
        self._test_attack(classifier, x_test, self.y_test_mnist, False)

    def test_keras_mnist_targeted(self):
        """
        TODO: Write Comment
        """
        classifier = get_classifier_kr()
        self._test_attack(
            classifier,
            self.x_test_mnist,
            self.y_test_mnist,
            True)

    def test_tensorflow_mnist_targeted(self):
        """
        TODO: Write Comment
        """
        classifier, sess = get_classifier_tf()
        self._test_attack(
            classifier,
            self.x_test_mnist,
            self.y_test_mnist,
            True)

    def test_pytorch_mnist_targeted(self):
        """
        TODO: Write Comment
        """
        x_test = np.swapaxes(self.x_test_mnist, 1, 3).astype(np.float32)
        classifier = get_classifier_pt()
        self._test_attack(classifier, x_test, self.y_test_mnist, True)

    def _test_attack(self, classifier, x_test, y_test, targeted):
        """
        TODO: Write Comment
        """
        x_test_original = x_test.copy()

        if targeted:
            # Generate random target classes
            class_y_test = np.argmax(y_test, axis=1)
            nb_classes   = np.unique(class_y_test).shape[0]
            targets      = np.random.randint(nb_classes, size=self.n_test)
            for i in range(self.n_test):
                if class_y_test[i] == targets[i]:
                    targets[i] -= 1
        else:
            targets = y_test

        for es in [0, 1]:

            df = PixelAttack(classifier, th=64, es=es, targeted=targeted)
            x_test_adv = df.generate(x_test_original, targets)

            self.assertFalse((x_test == x_test_adv).all())
            self.assertFalse((0.0 == x_test_adv).all())

            y_pred = get_labels_np_array(classifier.predict(x_test_adv))
            self.assertFalse((y_test == y_pred).all())

            accuracy = np.sum(
                np.argmax(
                    y_pred,
                    axis=1) == np.argmax(
                        self.y_test_mnist,
                        axis=1)) / self.n_test
            logger.info(
                'Accuracy on adversarial examples: %.2f%%',
                (accuracy * 100))

        # Check that x_test has not been modified by attack and classifier
        self.assertAlmostEqual(
            float(
                np.max(
                    np.abs(
                        x_test_original -
                        x_test))),
            0.0,
            delta=0.00001)


if __name__ == '__main__':
    unittest.main()
