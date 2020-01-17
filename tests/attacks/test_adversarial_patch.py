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
from sklearn.tree import DecisionTreeClassifier

from art.attacks import AdversarialPatch
from art.utils import master_seed
from art.classifiers.scikitlearn import ScikitlearnDecisionTreeClassifier

from tests.utils_test import TestBase
from tests.utils_test import get_classifier_tf, get_classifier_kr, get_classifier_pt, get_iris_classifier_kr

logger = logging.getLogger(__name__)


class TestAdversarialPatch(TestBase):
    """
    A unittest class for testing Adversarial Patch attack.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.n_train = 10
        cls.n_test = 10
        cls.x_train_mnist = cls.x_train_mnist[0:cls.n_train]
        cls.y_train_mnist = cls.y_train_mnist[0:cls.n_train]
        cls.x_test_mnist = cls.x_test_mnist[0:cls.n_test]
        cls.y_test_mnist = cls.y_test_mnist[0:cls.n_test]

    def test_tensorflow(self):
        """
        First test with the TensorFlowClassifier.
        :return:
        """
        tfc, sess = get_classifier_tf()

        attack_ap = AdversarialPatch(tfc, rotation_max=22.5, scale_min=0.1, scale_max=1.0, learning_rate=5.0,
                                     batch_size=10, max_iter=500)
        patch_adv, _ = attack_ap.generate(self.x_train_mnist)

        self.assertAlmostEqual(patch_adv[8, 8, 0], -3.1106631027725005, delta=0.4)
        self.assertAlmostEqual(patch_adv[14, 14, 0], 18.101, delta=0.2)
        self.assertAlmostEqual(float(np.sum(patch_adv)), 624.867, delta=70.0)

        if sess is not None:
            sess.close()

    def test_keras(self):
        """
        Second test with the KerasClassifier.
        :return:
        """
        krc = get_classifier_kr()

        attack_ap = AdversarialPatch(krc, rotation_max=22.5, scale_min=0.1, scale_max=1.0, learning_rate=5.0,
                                     batch_size=10, max_iter=500)
        master_seed(1234)
        patch_adv, _ = attack_ap.generate(self.x_train_mnist)

        self.assertAlmostEqual(patch_adv[8, 8, 0], -3.494, delta=0.2)
        self.assertAlmostEqual(patch_adv[14, 14, 0], 18.402, delta=0.2)
        self.assertAlmostEqual(float(np.sum(patch_adv)), 1099.293, delta=50)

    def test_pytorch(self):
        """
        Third test with the PyTorchClassifier.
        :return:
        """
        ptc = get_classifier_pt()

        x_train = np.reshape(self.x_train_mnist, (self.n_train, 1, 28, 28)).astype(np.float32)

        attack_ap = AdversarialPatch(ptc, rotation_max=22.5, scale_min=0.1, scale_max=1.0, learning_rate=5.0,
                                     batch_size=10, max_iter=500)

        patch_adv, _ = attack_ap.generate(x_train)

        self.assertAlmostEqual(patch_adv[0, 8, 8], -3.143605902784875, delta=0.1)
        self.assertAlmostEqual(patch_adv[0, 14, 14], 19.790434152473054, delta=0.1)
        self.assertAlmostEqual(float(np.sum(patch_adv)), 383.068, delta=0.1)

    def test_failure_feature_vectors(self):
        attack_params = {"rotation_max": 22.5, "scale_min": 0.1, "scale_max": 1.0, "learning_rate": 5.0,
                         "number_of_steps": 5, "batch_size": 10}
        classifier = get_iris_classifier_kr()
        attack = AdversarialPatch(classifier=classifier)
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
            _ = AdversarialPatch(classifier=classifier)

        self.assertIn('For `AdversarialPatch` classifier must be an instance of '
                      '`art.classifiers.classifier.Classifier`, the provided classifier is instance of '
                      '(<class \'object\'>,).', str(context.exception))

    def test_classifier_type_check_fail_gradients(self):
        # Use a test classifier not providing gradients required by white-box attack
        classifier = ScikitlearnDecisionTreeClassifier(model=DecisionTreeClassifier())
        with self.assertRaises(TypeError) as context:
            _ = AdversarialPatch(classifier=classifier)

        self.assertIn('For `AdversarialPatch` classifier must be an instance of '
                      '`art.classifiers.classifier.ClassifierNeuralNetwork` and '
                      '`art.classifiers.classifier.ClassifierGradients`, the provided classifier is instance of '
                      '(<class \'art.classifiers.scikitlearn.ScikitlearnClassifier\'>,).', str(context.exception))


if __name__ == '__main__':
    unittest.main()
