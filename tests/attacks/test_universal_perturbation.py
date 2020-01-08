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

from art.attacks import UniversalPerturbation
from art.classifiers import KerasClassifier
from art.utils import load_dataset, master_seed
from tests.utils_test import get_classifier_tf, get_classifier_kr, get_classifier_pt
from tests.utils_test import get_iris_classifier_tf, get_iris_classifier_kr, get_iris_classifier_pt

logger = logging.getLogger(__name__)

BATCH_SIZE = 100
NB_TRAIN = 500
NB_TEST = 10


class TestUniversalPerturbation(unittest.TestCase):
    """
    A unittest class for testing the UniversalPerturbation attack.
    """

    @classmethod
    def setUpClass(cls):
        # MNIST
        (x_train, y_train), (x_test, y_test), _, _ = load_dataset('mnist')
        x_train, y_train, x_test, y_test = x_train[:NB_TRAIN], y_train[:NB_TRAIN], x_test[:NB_TEST], y_test[:NB_TEST]
        cls.mnist = (x_train, y_train), (x_test, y_test)

        # Iris
        (x_train, y_train), (x_test, y_test), _, _ = load_dataset('iris')
        cls.iris = (x_train, y_train), (x_test, y_test)

    def setUp(self):
        master_seed(1234)

    def test_tensorflow_mnist(self):
        """
        First test with the TensorFlowClassifier.
        :return:
        """
        (x_train, y_train), (x_test, y_test) = self.mnist
        x_test_original = x_test.copy()

        # Build TensorFlowClassifier
        tfc, sess = get_classifier_tf()

        # Attack
        up = UniversalPerturbation(tfc, max_iter=1, attacker="newtonfool", attacker_params={"max_iter": 5})
        x_train_adv = up.generate(x_train)
        self.assertTrue((up.fooling_rate >= 0.2) or not up.converged)

        x_test_adv = x_test + up.noise
        self.assertFalse((x_test == x_test_adv).all())

        train_y_pred = np.argmax(tfc.predict(x_train_adv), axis=1)
        test_y_pred = np.argmax(tfc.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(y_test, axis=1) == test_y_pred).all())
        self.assertFalse((np.argmax(y_train, axis=1) == train_y_pred).all())

        # Check that x_test has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test))), 0.0, delta=0.00001)

    def test_keras_mnist(self):
        """
        Second test with the KerasClassifier.
        :return:
        """
        (x_train, y_train), (x_test, y_test) = self.mnist
        x_test_original = x_test.copy()

        # Build KerasClassifier
        krc = get_classifier_kr()

        # Attack
        up = UniversalPerturbation(krc, max_iter=1, attacker="ead", attacker_params={"max_iter": 5, "targeted": False})
        x_train_adv = up.generate(x_train)
        self.assertTrue((up.fooling_rate >= 0.2) or not up.converged)

        x_test_adv = x_test + up.noise
        self.assertFalse((x_test == x_test_adv).all())

        train_y_pred = np.argmax(krc.predict(x_train_adv), axis=1)
        test_y_pred = np.argmax(krc.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(y_test, axis=1) == test_y_pred).all())
        self.assertFalse((np.argmax(y_train, axis=1) == train_y_pred).all())

        # Check that x_test has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test))), 0.0, delta=0.00001)

        # sess.close()

    def test_pytorch_mnist(self):
        """
        Third test with the PyTorchClassifier.
        :return:
        """
        (x_train, y_train), (x_test, y_test) = self.mnist
        x_train = np.swapaxes(x_train, 1, 3).astype(np.float32)
        x_test = np.swapaxes(x_test, 1, 3).astype(np.float32)
        x_test_original = x_test.copy()

        # Build PyTorchClassifier
        ptc = get_classifier_pt()

        # Attack
        up = UniversalPerturbation(ptc, max_iter=1, attacker="newtonfool", attacker_params={"max_iter": 5})
        x_train_adv = up.generate(x_train)
        self.assertTrue((up.fooling_rate >= 0.2) or not up.converged)

        x_test_adv = x_test + up.noise
        self.assertFalse((x_test == x_test_adv).all())

        train_y_pred = np.argmax(ptc.predict(x_train_adv), axis=1)
        test_y_pred = np.argmax(ptc.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(y_test, axis=1) == test_y_pred).all())
        self.assertFalse((np.argmax(y_train, axis=1) == train_y_pred).all())

        # Check that x_test has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test))), 0.0, delta=0.00001)

    def test_classifier_type_check_fail_classifier(self):
        # Use a useless test classifier to test basic classifier properties
        class ClassifierNoAPI:
            pass

        classifier = ClassifierNoAPI
        with self.assertRaises(TypeError) as context:
            _ = UniversalPerturbation(classifier=classifier)

        self.assertIn('For `UniversalPerturbation` classifier must be an instance of '
                      '`art.classifiers.classifier.Classifier`, the provided classifier is instance of '
                      '(<class \'object\'>,).', str(context.exception))

    def test_classifier_type_check_fail_gradients(self):
        # Use a test classifier not providing gradients required by white-box attack
        from art.classifiers.scikitlearn import ScikitlearnDecisionTreeClassifier
        from sklearn.tree import DecisionTreeClassifier

        classifier = ScikitlearnDecisionTreeClassifier(model=DecisionTreeClassifier())
        with self.assertRaises(TypeError) as context:
            _ = UniversalPerturbation(classifier=classifier)

        self.assertIn('For `UniversalPerturbation` classifier must be an instance of '
                      '`art.classifiers.classifier.ClassifierNeuralNetwork` and '
                      '`art.classifiers.classifier.ClassifierGradients`, the provided classifier is instance of '
                      '(<class \'art.classifiers.scikitlearn.ScikitlearnClassifier\'>,).', str(context.exception))

    def test_keras_iris_clipped(self):
        (_, _), (x_test, y_test) = self.iris
        classifier = get_iris_classifier_kr()

        # Test untargeted attack
        attack_params = {"max_iter": 1, "attacker": "newtonfool", "attacker_params": {"max_iter": 5}}
        attack = UniversalPerturbation(classifier)
        attack.set_params(**attack_params)
        x_test_adv = attack.generate(x_test)
        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1).all())
        self.assertTrue((x_test_adv >= 0).all())

        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(y_test, axis=1) == preds_adv).all())
        acc = np.sum(preds_adv == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info('Accuracy on Iris with universal adversarial examples: %.2f%%', (acc * 100))

    def test_keras_iris_unbounded(self):
        (_, _), (x_test, y_test) = self.iris
        classifier = get_iris_classifier_kr()

        # Recreate a classifier without clip values
        classifier = KerasClassifier(model=classifier._model, use_logits=False, channel_index=1)
        attack_params = {"max_iter": 1, "attacker": "newtonfool", "attacker_params": {"max_iter": 5}}
        attack = UniversalPerturbation(classifier)
        attack.set_params(**attack_params)
        x_test_adv = attack.generate(x_test)
        self.assertFalse((x_test == x_test_adv).all())

        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(y_test, axis=1) == preds_adv).all())
        acc = np.sum(preds_adv == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info('Accuracy on Iris with universal adversarial examples: %.2f%%', (acc * 100))

    def test_tensorflow_iris(self):
        (_, _), (x_test, y_test) = self.iris
        classifier, _ = get_iris_classifier_tf()

        # Test untargeted attack
        attack_params = {"max_iter": 1, "attacker": "ead", "attacker_params": {"max_iter": 5, "targeted": False}}
        attack = UniversalPerturbation(classifier)
        attack.set_params(**attack_params)
        x_test_adv = attack.generate(x_test)
        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1).all())
        self.assertTrue((x_test_adv >= 0).all())

        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(y_test, axis=1) == preds_adv).all())
        acc = np.sum(preds_adv == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info('Accuracy on Iris with universal adversarial examples: %.2f%%', (acc * 100))

    def test_pytorch_iris(self):
        (_, _), (x_test, y_test) = self.iris
        classifier = get_iris_classifier_pt()

        attack_params = {"max_iter": 1, "attacker": "ead", "attacker_params": {"max_iter": 5, "targeted": False}}
        attack = UniversalPerturbation(classifier)
        attack.set_params(**attack_params)
        x_test_adv = attack.generate(x_test)
        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1).all())
        self.assertTrue((x_test_adv >= 0).all())

        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(y_test, axis=1) == preds_adv).all())
        acc = np.sum(preds_adv == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info('Accuracy on Iris with universal adversarial examples: %.2f%%', (acc * 100))


if __name__ == '__main__':
    unittest.main()
