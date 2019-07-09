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

from art.attacks.spatial_transformation import SpatialTransformation
from art.utils import load_mnist, master_seed
from art.utils_test import get_classifier_tf, get_classifier_kr, get_classifier_pt

logger = logging.getLogger('testLogger')

BATCH_SIZE = 100
NB_TRAIN = 1000
NB_TEST = 10


class TestSpatialTransformation(unittest.TestCase):
    """
    A unittest class for testing Spatial attack.
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
        (x_train, _), (x_test, _) = self.mnist

        # Attack
        attack_st = SpatialTransformation(tfc, max_translation=10.0, num_translations=3, max_rotation=30.0,
                                          num_rotations=3)
        x_train_adv = attack_st.generate(x_train)

        self.assertLessEqual(abs(x_train_adv[0, 8, 13, 0] - 0.49004024), 0.01)

        self.assertLessEqual(abs(attack_st.fooling_rate - 0.707), 0.01)

        self.assertEqual(attack_st.attack_trans_x, 3)
        self.assertEqual(attack_st.attack_trans_y,  3)
        self.assertEqual(attack_st.attack_rot, 30.0)

        x_test_adv = attack_st.generate(x_test)

        self.assertLessEqual(abs(x_test_adv[0, 14, 14, 0] - 0.013572651), 0.01)

        sess.close()
        tf.reset_default_graph()

    def test_krclassifier(self):
        """
        Second test with the KerasClassifier.
        :return:
        """
        # Build KerasClassifier
        krc, sess = get_classifier_kr()

        # Get MNIST
        (x_train, _), (x_test, _) = self.mnist

        # Attack
        attack_st = SpatialTransformation(krc, max_translation=10.0, num_translations=3, max_rotation=30.0,
                                          num_rotations=3)
        x_train_adv = attack_st.generate(x_train)

        self.assertLessEqual(abs(x_train_adv[0, 8, 13, 0] - 0.49004024), 0.01)
        self.assertLessEqual(abs(attack_st.fooling_rate - 0.707), 0.01)

        self.assertEqual(attack_st.attack_trans_x, 3)
        self.assertEqual(attack_st.attack_trans_y, 3)
        self.assertEqual(attack_st.attack_rot, 30.0)

        x_test_adv = attack_st.generate(x_test)

        self.assertLessEqual(abs(x_test_adv[0, 14, 14, 0] - 0.013572651), 0.01)

        k.clear_session()

    def test_ptclassifier(self):
        """
        Third test with the PyTorchClassifier.
        :return:
        """
        # Build PyTorchClassifier
        ptc = get_classifier_pt()

        # Get MNIST
        (x_train, _), (x_test, _) = self.mnist
        x_train = np.swapaxes(x_train, 1, 3)
        x_test = np.swapaxes(x_test, 1, 3)

        # Attack
        attack_st = SpatialTransformation(ptc, max_translation=10.0, num_translations=3, max_rotation=30.0,
                                          num_rotations=3)
        x_train_adv = attack_st.generate(x_train)

        self.assertLessEqual(abs(x_train_adv[0, 0, 13, 5] - 0.374206543), 0.01)
        self.assertLessEqual(abs(attack_st.fooling_rate - 0.361), 0.01)

        self.assertEqual(attack_st.attack_trans_x, 0)
        self.assertEqual(attack_st.attack_trans_y, -3)
        self.assertEqual(attack_st.attack_rot, 30.0)

        x_test_adv = attack_st.generate(x_test)

        self.assertLessEqual(abs(x_test_adv[0, 0, 14, 14] - 0.008591662), 0.01)

    def test_failure_feature_vectors(self):
        attack_params = {"max_translation": 10.0, "num_translations": 3, "max_rotation": 30.0, "num_rotations": 3}
        attack = SpatialTransformation(classifier=None)
        attack.set_params(**attack_params)
        data = np.random.rand(10, 4)

        # Assert that value error is raised for feature vectors
        with self.assertRaises(ValueError) as context:
            attack.generate(data)

        self.assertIn('Feature vectors detected.', str(context.exception))


if __name__ == '__main__':
    unittest.main()
