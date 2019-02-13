from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

import keras
import keras.backend as k
import numpy as np
import tensorflow as tf
import torch.nn as nn
import torch.optim as optim
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential

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
        attack_params = {"max_iter": 1, "attacker": "margin", "attacker_params": {"max_iter": 5, "target_scan_iters": 5,
                                                                                  "final_restore_iters": 5}}
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
