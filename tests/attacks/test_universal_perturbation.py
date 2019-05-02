# MIT License
#
# Copyright (C) IBM Corporation 2018
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

from art.attacks.universal_perturbation import UniversalPerturbation
from art.utils import load_mnist, master_seed, get_classifier_tf, get_classifier_kr, get_classifier_pt

logger = logging.getLogger('testLogger')

BATCH_SIZE = 100
NB_TRAIN = 500
NB_TEST = 10


class TestUniversalPerturbation(unittest.TestCase):
    """
    A unittest class for testing the UniversalPerturbation attack.
    """

    @classmethod
    def setUpClass(cls):
        # Get MNIST
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        x_train, y_train = x_train[:NB_TRAIN], y_train[:NB_TRAIN]
        x_test, y_test = x_test[:NB_TEST], y_test[:NB_TEST]
        cls.mnist = (x_train, y_train), (x_test, y_test)

    def setUp(self):
        # Set master seed
        master_seed(1234)

    def test_tfclassifier(self):
        """
        First test with the TFClassifier.
        :return:
        """
        # Build TFClassifier
        tfc, sess = get_classifier_tf()

        # Get MNIST
        (x_train, y_train), (x_test, y_test) = self.mnist

        # Attack
        attack_params = {"max_iter": 1, "attacker": "newtonfool", "attacker_params": {"max_iter": 5}}
        up = UniversalPerturbation(tfc)
        x_train_adv = up.generate(x_train, **attack_params)
        self.assertTrue((up.fooling_rate >= 0.2) or not up.converged)

        x_test_adv = x_test + up.v
        self.assertFalse((x_test == x_test_adv).all())

        train_y_pred = np.argmax(tfc.predict(x_train_adv), axis=1)
        test_y_pred = np.argmax(tfc.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(y_test, axis=1) == test_y_pred).all())
        self.assertFalse((np.argmax(y_train, axis=1) == train_y_pred).all())

    def test_krclassifier(self):
        """
        Second test with the KerasClassifier.
        :return:
        """
        # Build KerasClassifier
        krc, sess = get_classifier_kr()

        # Get MNIST
        (x_train, y_train), (x_test, y_test) = self.mnist

        # Attack
        attack_params = {"max_iter": 1, "attacker": "ead", "attacker_params": {"max_iter": 5, "targeted": False}}
        up = UniversalPerturbation(krc)
        x_train_adv = up.generate(x_train, **attack_params)
        self.assertTrue((up.fooling_rate >= 0.2) or not up.converged)

        x_test_adv = x_test + up.v
        self.assertFalse((x_test == x_test_adv).all())

        train_y_pred = np.argmax(krc.predict(x_train_adv), axis=1)
        test_y_pred = np.argmax(krc.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(y_test, axis=1) == test_y_pred).all())
        self.assertFalse((np.argmax(y_train, axis=1) == train_y_pred).all())

    def test_ptclassifier(self):
        """
        Third test with the PyTorchClassifier.
        :return:
        """
        # Build PyTorchClassifier
        ptc = get_classifier_pt()

        # Get MNIST
        (x_train, y_train), (x_test, y_test) = self.mnist
        x_train = np.swapaxes(x_train, 1, 3)
        x_test = np.swapaxes(x_test, 1, 3)

        # Attack
        attack_params = {"max_iter": 1, "attacker": "newtonfool", "attacker_params": {"max_iter": 5}}
        up = UniversalPerturbation(ptc)
        x_train_adv = up.generate(x_train, **attack_params)
        self.assertTrue((up.fooling_rate >= 0.2) or not up.converged)

        x_test_adv = x_test + up.v
        self.assertFalse((x_test == x_test_adv).all())

        train_y_pred = np.argmax(ptc.predict(x_train_adv), axis=1)
        test_y_pred = np.argmax(ptc.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(y_test, axis=1) == test_y_pred).all())
        self.assertFalse((np.argmax(y_train, axis=1) == train_y_pred).all())


if __name__ == '__main__':
    unittest.main()
