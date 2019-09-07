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

from art.attacks import SpatialTransformation
from art.utils import load_dataset, master_seed
from art.utils_test import get_classifier_tf, get_classifier_kr, get_classifier_pt, get_iris_classifier_kr

logger = logging.getLogger('testLogger')

BATCH_SIZE = 10
NB_TRAIN = 100
NB_TEST = 10


class TestSpatialTransformation(unittest.TestCase):
    """
    A unittest class for testing Spatial attack.
    """

    @classmethod
    def setUpClass(cls):
        (x_train, y_train), (x_test, y_test), _, _ = load_dataset('mnist')

        cls.x_train = x_train[:NB_TRAIN]
        cls.y_train = y_train[:NB_TRAIN]
        cls.x_test = x_test[:NB_TEST]
        cls.y_test = y_test[:NB_TEST]

    def setUp(self):
        master_seed(1234)

    def test_tfclassifier(self):
        """
        First test with the TensorFlowClassifier.
        :return:
        """
        # Build TensorFlowClassifier
        tfc, sess = get_classifier_tf()

        # Attack
        attack_st = SpatialTransformation(tfc, max_translation=10.0, num_translations=3, max_rotation=30.0,
                                          num_rotations=3)
        x_train_adv = attack_st.generate(self.x_train)

        self.assertAlmostEqual(x_train_adv[0, 8, 13, 0], 0.49004024, delta=0.01)
        self.assertAlmostEqual(attack_st.fooling_rate, 0.72, delta=0.01)

        self.assertEqual(attack_st.attack_trans_x, 3)
        self.assertEqual(attack_st.attack_trans_y, 3)
        self.assertEqual(attack_st.attack_rot, 30.0)

        x_test_adv = attack_st.generate(self.x_test)

        self.assertAlmostEqual(x_test_adv[0, 14, 14, 0], 0.013572651, delta=0.01)

        sess.close()

    @unittest.skipIf(tf.__version__[0] == '2', reason='Skip unittests for TensorFlow v2 until Keras supports TensorFlow'
                                                      ' v2 as backend.')
    def test_krclassifier(self):
        """
        Second test with the KerasClassifier.
        :return:
        """
        # Build KerasClassifier
        krc = get_classifier_kr()

        # Attack
        attack_st = SpatialTransformation(krc, max_translation=10.0, num_translations=3, max_rotation=30.0,
                                          num_rotations=3)
        x_train_adv = attack_st.generate(self.x_train)

        self.assertAlmostEqual(x_train_adv[0, 8, 13, 0], 0.49004024, delta=0.01)
        self.assertAlmostEqual(attack_st.fooling_rate, 0.72, delta=0.01)

        self.assertEqual(attack_st.attack_trans_x, 3)
        self.assertEqual(attack_st.attack_trans_y, 3)
        self.assertEqual(attack_st.attack_rot, 30.0)

        x_test_adv = attack_st.generate(self.x_test)

        self.assertAlmostEqual(x_test_adv[0, 14, 14, 0], 0.013572651, delta=0.01)

        k.clear_session()

    def test_ptclassifier(self):
        """
        Third test with the PyTorchClassifier.
        :return:
        """
        # Build PyTorchClassifier
        ptc = get_classifier_pt()

        x_train = np.swapaxes(self.x_train, 1, 3).astype(np.float32)
        x_test = np.swapaxes(self.x_test, 1, 3).astype(np.float32)

        # Attack
        attack_st = SpatialTransformation(ptc, max_translation=10.0, num_translations=3, max_rotation=30.0,
                                          num_rotations=3)
        x_train_adv = attack_st.generate(x_train)

        print('abs(x_train_adv[0, 0, 13, :]', abs(x_train[0, 0, 13, :]))
        print('abs(x_train_adv[0, 0, 13, :]', abs(x_train_adv[0, 0, 13, :]))

        self.assertAlmostEqual(x_train_adv[0, 0, 13, 7], 0.287, delta=0.01)
        self.assertAlmostEqual(attack_st.fooling_rate, 0.82, delta=0.01)

        self.assertEqual(attack_st.attack_trans_x, 0)
        self.assertEqual(attack_st.attack_trans_y, 3)
        self.assertEqual(attack_st.attack_rot, -30.0)

        x_test_adv = attack_st.generate(x_test)

        self.assertLessEqual(abs(x_test_adv[0, 0, 14, 14] - 0.008591662), 0.01)

    @unittest.skipIf(tf.__version__[0] == '2', reason='Skip unittests for TensorFlow v2 until Keras supports TensorFlow'
                                                      ' v2 as backend.')
    def test_failure_feature_vectors(self):
        attack_params = {"max_translation": 10.0, "num_translations": 3, "max_rotation": 30.0, "num_rotations": 3}
        classifier, _ = get_iris_classifier_kr()
        attack = SpatialTransformation(classifier=classifier)
        attack.set_params(**attack_params)
        data = np.random.rand(10, 4)

        # Assert that value error is raised for feature vectors
        with self.assertRaises(ValueError) as context:
            attack.generate(data)

        self.assertIn('Feature vectors detected.', str(context.exception))

    def test_classifier_type_check_fail_classifier(self):
        # Use a useless test classifier to test basic classifier properties
        class ClassifierNoAPI:
            pass

        classifier = ClassifierNoAPI
        with self.assertRaises(TypeError) as context:
            _ = SpatialTransformation(classifier=classifier)

        self.assertIn('For `SpatialTransformation` classifier must be an instance of '
                      '`art.classifiers.classifier.Classifier`, the provided classifier is instance of '
                      '(<class \'object\'>,).', str(context.exception))


if __name__ == '__main__':
    unittest.main()
