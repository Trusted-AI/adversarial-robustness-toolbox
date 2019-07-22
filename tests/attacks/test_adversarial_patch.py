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

import keras.backend as k
import numpy as np
import tensorflow as tf

from art.attacks import AdversarialPatch
from art.utils import load_mnist, master_seed
from art.utils_test import get_classifier_tf, get_classifier_kr, get_classifier_pt

logger = logging.getLogger('testLogger')

BATCH_SIZE = 10
NB_TRAIN = 10
NB_TEST = 10


class TestAdversarialPatch(unittest.TestCase):
    """
    A unittest class for testing Adversarial Patch attack.
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
        (x_train, _), (_, _) = self.mnist

        # Attack
        attack_ap = AdversarialPatch(tfc, rotation_max=22.5, scale_min=0.1, scale_max=1.0, learning_rate=5.0,
                                     batch_size=10, max_iter=500)
        patch_adv, _ = attack_ap.generate(x_train)

        self.assertAlmostEqual(patch_adv[8, 8, 0], -3.1106631027725005, delta=.1)
        self.assertAlmostEqual(patch_adv[14, 14, 0], 18.101, delta=.1)
        self.assertAlmostEqual(np.sum(patch_adv), 624.867, delta=.1)

        sess.close()

    @unittest.skipIf(tf.__version__[0] == '2', reason='Skip unittests for Tensorflow v2 until Keras supports Tensorflow'
                                                      ' v2 as backend.')
    def test_krclassifier(self):
        """
        Second test with the KerasClassifier.
        :return:
        """
        # Build KerasClassifier
        krc, _ = get_classifier_kr()

        # Get MNIST
        (x_train, _), (_, _) = self.mnist

        # Attack
        attack_ap = AdversarialPatch(krc, rotation_max=22.5, scale_min=0.1, scale_max=1.0, learning_rate=5.0,
                                     batch_size=10, max_iter=500)
        patch_adv, _ = attack_ap.generate(x_train)

        self.assertAlmostEqual(patch_adv[8, 8, 0], -3.336, delta=.1)
        self.assertAlmostEqual(patch_adv[14, 14, 0], 18.574, delta=.1)
        self.assertAlmostEqual(np.sum(patch_adv), 1054.587, delta=.1)

        k.clear_session()

    def test_ptclassifier(self):
        """
        Third test with the PyTorchClassifier.
        :return:
        """
        # Build PyTorchClassifier
        ptc = get_classifier_pt()

        # Get MNIST
        (x_train, _), (_, _) = self.mnist
        x_train = np.swapaxes(x_train, 1, 3)

        # Attack
        attack_ap = AdversarialPatch(ptc, rotation_max=22.5, scale_min=0.1, scale_max=1.0, learning_rate=5.0,
                                     batch_size=10, max_iter=500)
        patch_adv, _ = attack_ap.generate(x_train)

        self.assertAlmostEqual(patch_adv[0, 8, 8], -3.143605902784875, delta=.1)
        self.assertAlmostEqual(patch_adv[0, 14, 14], 19.790434152473054, delta=.1)
        self.assertAlmostEqual(np.sum(patch_adv), 382.163, delta=.1)

    def test_failure_feature_vectors(self):
        attack_params = {"rotation_max": 22.5, "scale_min": 0.1, "scale_max": 1.0,
                         "learning_rate": 5.0, "number_of_steps": 5, "batch_size": 10}
        attack = AdversarialPatch(classifier=None)
        attack.set_params(**attack_params)
        data = np.random.rand(10, 4)

        # Assert that value error is raised for feature vectors
        with self.assertRaises(ValueError) as context:
            attack.generate(data)

        self.assertIn('Feature vectors detected.', str(context.exception))


if __name__ == '__main__':
    unittest.main()
