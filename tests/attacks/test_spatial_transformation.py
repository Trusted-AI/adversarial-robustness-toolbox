# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
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

from art.attacks.evasion.spatial_transformation import SpatialTransformation
from art.estimators.estimator import BaseEstimator, NeuralNetworkMixin

from tests.utils import TestBase
from tests.utils import get_image_classifier_tf, get_image_classifier_kr
from tests.utils import get_image_classifier_pt, get_tabular_classifier_kr
from tests.attacks.utils import backend_test_classifier_type_check_fail

logger = logging.getLogger(__name__)


class TestSpatialTransformation(TestBase):
    """
    A unittest class for testing Spatial attack.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.n_train = 100
        cls.n_test = 10
        cls.x_train_mnist = cls.x_train_mnist[0 : cls.n_train]
        cls.y_train_mnist = cls.y_train_mnist[0 : cls.n_train]
        cls.x_test_mnist = cls.x_test_mnist[0 : cls.n_test]
        cls.y_test_mnist = cls.y_test_mnist[0 : cls.n_test]

    def test_2_tensorflow_classifier(self):
        """
        First test with the TensorFlowClassifier.
        :return:
        """
        x_test_original = self.x_test_mnist.copy()

        # Build TensorFlowClassifier
        tfc, sess = get_image_classifier_tf()

        # Attack
        attack_st = SpatialTransformation(
            tfc, max_translation=10.0, num_translations=3, max_rotation=30.0, num_rotations=3, verbose=False
        )
        x_train_adv = attack_st.generate(self.x_train_mnist)

        self.assertAlmostEqual(x_train_adv[0, 8, 13, 0], 0.49004024, delta=0.01)
        self.assertAlmostEqual(attack_st.fooling_rate, 0.71, delta=0.02)

        self.assertEqual(attack_st.attack_trans_x, 3)
        self.assertEqual(attack_st.attack_trans_y, 3)
        self.assertEqual(attack_st.attack_rot, 30.0)

        x_test_adv = attack_st.generate(self.x_test_mnist)

        self.assertAlmostEqual(x_test_adv[0, 14, 14, 0], 0.013572651, delta=0.01)

        # Check that x_test has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - self.x_test_mnist))), 0.0, delta=0.00001)

        if sess is not None:
            sess.close()

    def test_4_keras_classifier(self):
        """
        Second test with the KerasClassifier.
        :return:
        """
        x_test_original = self.x_test_mnist.copy()

        # Build KerasClassifier
        krc = get_image_classifier_kr()

        # Attack
        attack_st = SpatialTransformation(
            krc, max_translation=10.0, num_translations=3, max_rotation=30.0, num_rotations=3, verbose=False
        )
        x_train_adv = attack_st.generate(self.x_train_mnist)

        self.assertAlmostEqual(x_train_adv[0, 8, 13, 0], 0.49004024, delta=0.01)
        self.assertAlmostEqual(attack_st.fooling_rate, 0.71, delta=0.02)

        self.assertEqual(attack_st.attack_trans_x, 3)
        self.assertEqual(attack_st.attack_trans_y, 3)
        self.assertEqual(attack_st.attack_rot, 30.0)

        x_test_adv = attack_st.generate(self.x_test_mnist)

        self.assertAlmostEqual(x_test_adv[0, 14, 14, 0], 0.013572651, delta=0.01)

        # Check that x_test has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - self.x_test_mnist))), 0.0, delta=0.00001)

        k.clear_session()

    def test_3_pytorch_classifier(self):
        """
        Third test with the PyTorchClassifier.
        :return:
        """
        x_train_mnist = np.reshape(self.x_train_mnist, (self.x_train_mnist.shape[0], 1, 28, 28)).astype(np.float32)
        x_test_mnist = np.reshape(self.x_test_mnist, (self.x_test_mnist.shape[0], 1, 28, 28)).astype(np.float32)
        x_test_original = x_test_mnist.copy()

        # Build PyTorchClassifier
        ptc = get_image_classifier_pt(from_logits=True)

        # Attack
        attack_st = SpatialTransformation(
            ptc, max_translation=10.0, num_translations=3, max_rotation=30.0, num_rotations=3, verbose=False
        )
        x_train__mnistadv = attack_st.generate(x_train_mnist)

        self.assertAlmostEqual(x_train__mnistadv[0, 0, 13, 18], 0.627451, delta=0.01)
        self.assertAlmostEqual(attack_st.fooling_rate, 0.57, delta=0.03)

        self.assertEqual(attack_st.attack_trans_x, 0)
        self.assertEqual(attack_st.attack_trans_y, 3)
        self.assertEqual(attack_st.attack_rot, 0.0)

        x_test_adv = attack_st.generate(x_test_mnist)

        self.assertLessEqual(abs(x_test_adv[0, 0, 14, 14] - 0.008591662), 0.01)

        # Check that x_test has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test_mnist))), 0.0, delta=0.00001)

    def test_5_failure_feature_vectors(self):
        attack_params = {"max_translation": 10.0, "num_translations": 3, "max_rotation": 30.0, "num_rotations": 3}
        classifier = get_tabular_classifier_kr()
        attack = SpatialTransformation(classifier=classifier, verbose=False)
        attack.set_params(**attack_params)
        data = np.random.rand(10, 4)

        # Assert that value error is raised for feature vectors
        with self.assertRaises(ValueError) as context:
            attack.generate(data)

        self.assertIn("Feature vectors detected.", str(context.exception))

    def test_1_classifier_type_check_fail(self):
        backend_test_classifier_type_check_fail(SpatialTransformation, [BaseEstimator, NeuralNetworkMixin])


if __name__ == "__main__":
    unittest.main()
