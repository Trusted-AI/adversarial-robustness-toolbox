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

import keras
import keras.backend as k
import numpy as np

from art.attacks import CarliniL2Method, CarliniLInfMethod
from art.classifiers import KerasClassifier
from art.utils import load_dataset, random_targets, master_seed, to_categorical
from tests.utils_test import get_classifier_tf, get_classifier_kr, get_classifier_pt
from tests.utils_test import get_iris_classifier_tf, get_iris_classifier_kr, get_iris_classifier_pt

logger = logging.getLogger(__name__)

BATCH_SIZE = 100
NB_TRAIN = 10
NB_TEST = 10


class TestCarlini(unittest.TestCase):
    """
    A unittest class for testing the Carlini L2 attack.
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

    def test_tensorflow_failure_attack_L2(self):
        """
        Test the corner case when attack is failed.
        :return:
        """
        (_, _), (x_test, y_test) = self.mnist
        x_test_original = x_test.copy()

        # Build TensorFlowClassifier
        tfc, sess = get_classifier_tf(from_logits=True)

        # Failure attack
        cl2m = CarliniL2Method(classifier=tfc, targeted=True, max_iter=0, binary_search_steps=0, learning_rate=0,
                               initial_const=1)
        params = {'y': random_targets(y_test, tfc.nb_classes())}
        x_test_adv = cl2m.generate(x_test, **params)
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
        np.testing.assert_array_almost_equal(x_test, x_test_adv, decimal=3)

        # Check that x_test has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test))), 0.0, delta=0.00001)

        # Clean-up session
        sess.close()

    def test_tensorflow_mnist_L2(self):
        """
        First test with the TensorFlowClassifier.
        :return:
        """
        (_, _), (x_test, y_test) = self.mnist
        x_test_original = x_test.copy()

        # Build TensorFlowClassifier
        tfc, sess = get_classifier_tf()

        # First attack
        cl2m = CarliniL2Method(classifier=tfc, targeted=True, max_iter=10)
        params = {'y': random_targets(y_test, tfc.nb_classes())}
        x_test_adv = cl2m.generate(x_test, **params)
        self.assertFalse((x_test == x_test_adv).all())
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        logger.debug('CW2 Target: %s', target)
        logger.debug('CW2 Actual: %s', y_pred_adv)
        logger.info('CW2 Success Rate: %.2f', (np.sum(target == y_pred_adv) / float(len(target))))
        self.assertTrue((target == y_pred_adv).any())

        # Second attack, no batching
        cl2m = CarliniL2Method(classifier=tfc, targeted=False, max_iter=10, batch_size=1)
        x_test_adv = cl2m.generate(x_test)
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        logger.debug('CW2 Target: %s', target)
        logger.debug('CW2 Actual: %s', y_pred_adv)
        logger.info('CW2 Success Rate: %.2f', (np.sum(target == y_pred_adv) / float(len(target))))
        self.assertTrue((target != y_pred_adv).any())

        # Check that x_test has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test))), 0.0, delta=0.00001)

        # Clean-up session
        sess.close()

    @unittest.skipIf(not (int(keras.__version__.split('.')[0]) == 2 and int(keras.__version__.split('.')[1]) >= 3),
                     reason='Minimal version of Keras or TensorFlow required.')
    def test_keras_mnist_L2(self):
        """
        Second test with the KerasClassifier.
        :return:
        """
        (_, _), (x_test, y_test) = self.mnist
        x_test_original = x_test.copy()

        # Build KerasClassifier
        krc = get_classifier_kr(from_logits=True)

        # First attack
        cl2m = CarliniL2Method(classifier=krc, targeted=True, max_iter=10)
        y_target = [6, 6, 7, 4, 9, 7, 9, 0, 1, 0]
        x_test_adv = cl2m.generate(x_test, y=to_categorical(y_target, nb_classes=10))
        self.assertFalse((x_test == x_test_adv).all())
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
        y_pred_adv = np.argmax(krc.predict(x_test_adv), axis=1)
        logger.debug('CW2 Target: %s', y_target)
        logger.debug('CW2 Actual: %s', y_pred_adv)
        logger.info('CW2 Success Rate: %.2f', (np.sum(y_target == y_pred_adv) / float(len(y_target))))
        self.assertTrue((y_target == y_pred_adv).any())

        # Second attack
        cl2m = CarliniL2Method(classifier=krc, targeted=False, max_iter=10)
        x_test_adv = cl2m.generate(x_test)
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
        y_pred_adv = np.argmax(krc.predict(x_test_adv), axis=1)
        logger.debug('CW2 Target: %s', y_target)
        logger.debug('CW2 Actual: %s', y_pred_adv)
        logger.info('CW2 Success Rate: %.2f', (np.sum(y_target != y_pred_adv) / float(len(y_target))))
        self.assertTrue((y_target != y_pred_adv).any())

        # Check that x_test has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test))), 0.0, delta=0.00001)

        # Clean-up
        k.clear_session()

    def test_pytorch_mnist_L2(self):
        """
        Third test with the PyTorchClassifier.
        :return:
        """
        (_, _), (x_test, y_test) = self.mnist
        x_test = np.reshape(x_test, (x_test.shape[0], 1, 28, 28)).astype(np.float32)
        x_test_original = x_test.copy()

        # Build PyTorchClassifier
        ptc = get_classifier_pt(from_logits=True)

        # First attack
        cl2m = CarliniL2Method(classifier=ptc, targeted=True, max_iter=10)
        params = {'y': random_targets(y_test, ptc.nb_classes())}
        x_test_adv = cl2m.generate(x_test, **params)
        self.assertFalse((x_test == x_test_adv).all())
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(ptc.predict(x_test_adv), axis=1)
        self.assertTrue((target == y_pred_adv).any())
        logger.info('CW2 Success Rate: %.2f', (sum(target == y_pred_adv) / float(len(target))))

        # Second attack
        cl2m = CarliniL2Method(classifier=ptc, targeted=False, max_iter=10)
        x_test_adv = cl2m.generate(x_test)
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(ptc.predict(x_test_adv), axis=1)
        self.assertTrue((target != y_pred_adv).any())
        logger.info('CW2 Success Rate: %.2f', (sum(target != y_pred_adv) / float(len(target))))

        # Check that x_test has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test))), 0.0, delta=0.00001)

    def test_classifier_type_check_fail_classifier_L2(self):
        # Use a useless test classifier to test basic classifier properties
        class ClassifierNoAPI:
            pass

        classifier = ClassifierNoAPI
        with self.assertRaises(TypeError) as context:
            _ = CarliniL2Method(classifier=classifier)

        self.assertIn('For `CarliniL2Method` classifier must be an instance of `art.classifiers.classifier.Classifier`,'
                      ' the provided classifier is instance of (<class \'object\'>,).', str(context.exception))

    def test_classifier_type_check_fail_gradients_L2(self):
        # Use a test classifier not providing gradients required by white-box attack
        from art.classifiers.scikitlearn import ScikitlearnDecisionTreeClassifier
        from sklearn.tree import DecisionTreeClassifier

        classifier = ScikitlearnDecisionTreeClassifier(model=DecisionTreeClassifier())
        with self.assertRaises(TypeError) as context:
            _ = CarliniL2Method(classifier=classifier)

        self.assertIn('For `CarliniL2Method` classifier must be an instance of '
                      '`art.classifiers.classifier.ClassifierGradients`, the provided classifier is instance of '
                      '(<class \'art.classifiers.scikitlearn.ScikitlearnClassifier\'>,).', str(context.exception))

    def test_keras_iris_clipped_L2(self):
        (_, _), (x_test, y_test) = self.iris
        classifier = get_iris_classifier_kr()
        attack = CarliniL2Method(classifier, targeted=False, max_iter=10)
        x_test_adv = attack.generate(x_test)
        self.assertFalse((x_test == x_test_adv).all())
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)

        predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(y_test, axis=1) == predictions_adv).all())
        accuracy = np.sum(predictions_adv == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info('Accuracy on Iris with C&W adversarial examples: %.2f%%', (accuracy * 100))

    def test_keras_iris_unbounded_L2(self):
        (_, _), (x_test, y_test) = self.iris
        classifier = get_iris_classifier_kr()

        # Recreate a classifier without clip values
        classifier = KerasClassifier(model=classifier._model, use_logits=False, channel_index=1)
        attack = CarliniL2Method(classifier, targeted=False, max_iter=10)
        x_test_adv = attack.generate(x_test)
        self.assertFalse((x_test == x_test_adv).all())

        predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(y_test, axis=1) == predictions_adv).all())
        accuracy = np.sum(predictions_adv == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info('Accuracy on Iris with C&W adversarial examples: %.2f%%', (accuracy * 100))

    def test_tensorflow_iris_L2(self):
        (_, _), (x_test, y_test) = self.iris
        classifier, _ = get_iris_classifier_tf()

        # Test untargeted attack
        attack = CarliniL2Method(classifier, targeted=False, max_iter=10)
        x_test_adv = attack.generate(x_test)
        self.assertFalse((x_test == x_test_adv).all())
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)

        predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(y_test, axis=1) == predictions_adv).all())
        accuracy = np.sum(predictions_adv == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info('Accuracy on Iris with C&W adversarial examples: %.2f%%', (accuracy * 100))

        # Test targeted attack
        targets = random_targets(y_test, nb_classes=3)
        attack = CarliniL2Method(classifier, targeted=True, max_iter=10)
        x_test_adv = attack.generate(x_test, **{'y': targets})
        self.assertFalse((x_test == x_test_adv).all())
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)

        predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertTrue((np.argmax(targets, axis=1) == predictions_adv).any())
        accuracy = np.sum(predictions_adv == np.argmax(targets, axis=1)) / y_test.shape[0]
        logger.info('Success rate of targeted C&W on Iris: %.2f%%', (accuracy * 100))

    def test_pytorch_iris_L2(self):
        (_, _), (x_test, y_test) = self.iris
        classifier = get_iris_classifier_pt()
        attack = CarliniL2Method(classifier, targeted=False, max_iter=10)
        x_test_adv = attack.generate(x_test)
        self.assertFalse((x_test == x_test_adv).all())
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)

        predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(y_test, axis=1) == predictions_adv).all())
        accuracy = np.sum(predictions_adv == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info('Accuracy on Iris with C&W adversarial examples: %.2f%%', (accuracy * 100))

    def test_scikitlearn_L2(self):
        from sklearn.linear_model import LogisticRegression

        from art.classifiers.scikitlearn import ScikitlearnLogisticRegression

        scikitlearn_test_cases = {LogisticRegression: ScikitlearnLogisticRegression}  # ,
        # SVC: ScikitlearnSVC,
        # LinearSVC: ScikitlearnSVC}

        (_, _), (x_test, y_test) = self.iris
        x_test_original = x_test.copy()

        for (model_class, classifier_class) in scikitlearn_test_cases.items():
            model = model_class()
            classifier = classifier_class(model=model, clip_values=(0, 1))
            classifier.fit(x=x_test, y=y_test)

            # Test untargeted attack
            attack = CarliniL2Method(classifier, targeted=False, max_iter=10)
            x_test_adv = attack.generate(x_test)
            self.assertFalse((x_test == x_test_adv).all())
            self.assertLessEqual(np.amax(x_test_adv), 1.0)
            self.assertGreaterEqual(np.amin(x_test_adv), 0.0)

            predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
            self.assertFalse((np.argmax(y_test, axis=1) == predictions_adv).all())
            accuracy = np.sum(predictions_adv == np.argmax(y_test, axis=1)) / y_test.shape[0]
            logger.info('Accuracy of ' + classifier.__class__.__name__ + ' on Iris with C&W adversarial examples: '
                                                                         '%.2f%%', (accuracy * 100))

            # Test targeted attack
            targets = random_targets(y_test, nb_classes=3)
            attack = CarliniL2Method(classifier, targeted=True, max_iter=10)
            x_test_adv = attack.generate(x_test, **{'y': targets})
            self.assertFalse((x_test == x_test_adv).all())
            self.assertLessEqual(np.amax(x_test_adv), 1.0)
            self.assertGreaterEqual(np.amin(x_test_adv), 0.0)

            predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
            self.assertTrue((np.argmax(targets, axis=1) == predictions_adv).any())
            accuracy = np.sum(predictions_adv == np.argmax(targets, axis=1)) / y_test.shape[0]
            logger.info('Success rate of ' + classifier.__class__.__name__ + ' on targeted C&W on Iris: %.2f%%',
                        (accuracy * 100))

            # Check that x_test has not been modified by attack and classifier
            self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test))), 0.0, delta=0.00001)

    """
    A unittest class for testing the Carlini LInf attack.
    """

    def test_tensorflow_failure_attack_LInf(self):
        """
        Test the corner case when attack is failed.
        :return:
        """
        (_, _), (x_test, y_test) = self.mnist

        # Build TensorFlowClassifier
        tfc, sess = get_classifier_tf(from_logits=True)

        # Failure attack
        clinfm = CarliniLInfMethod(classifier=tfc, targeted=True, max_iter=0, learning_rate=0, eps=0.5)
        params = {'y': random_targets(y_test, tfc.nb_classes())}
        x_test_adv = clinfm.generate(x_test, **params)
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
        self.assertTrue(np.allclose(x_test, x_test_adv, atol=1e-3))

        # Clean-up session
        sess.close()

    def test_tensorflow_mnist_LInf(self):
        """
        First test with the TensorFlowClassifier.
        :return:
        """
        (_, _), (x_test, y_test) = self.mnist

        # Build TensorFlowClassifier
        tfc, sess = get_classifier_tf(from_logits=True)

        # First attack
        clinfm = CarliniLInfMethod(classifier=tfc, targeted=True, max_iter=10, eps=0.5)
        params = {'y': random_targets(y_test, tfc.nb_classes())}
        x_test_adv = clinfm.generate(x_test, **params)
        self.assertFalse((x_test == x_test_adv).all())
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        logger.debug('CW0 Target: %s', target)
        logger.debug('CW0 Actual: %s', y_pred_adv)
        logger.info('CW0 Success Rate: %.2f', (np.sum(target == y_pred_adv) / float(len(target))))
        self.assertTrue((target == y_pred_adv).any())

        # Second attack, no batching
        clinfm = CarliniLInfMethod(classifier=tfc, targeted=False, max_iter=10, eps=0.5, batch_size=1)
        x_test_adv = clinfm.generate(x_test)
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), -1e-6)
        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        logger.debug('CW0 Target: %s', target)
        logger.debug('CW0 Actual: %s', y_pred_adv)
        logger.info('CW0 Success Rate: %.2f', (np.sum(target != y_pred_adv) / float(len(target))))
        self.assertTrue((target != y_pred_adv).any())

        # Clean-up session
        sess.close()

    @unittest.skipIf(not (int(keras.__version__.split('.')[0]) == 2 and int(keras.__version__.split('.')[1]) >= 3),
                     reason='Minimal version of Keras or TensorFlow required.')
    def test_keras_mnist_LInf(self):
        """
        Second test with the KerasClassifier.
        :return:
        """
        (_, _), (x_test, y_test) = self.mnist

        # Build KerasClassifier
        krc = get_classifier_kr(from_logits=True)

        # First attack
        clinfm = CarliniLInfMethod(classifier=krc, targeted=True, max_iter=10, eps=0.5)
        params = {'y': random_targets(y_test, krc.nb_classes())}
        x_test_adv = clinfm.generate(x_test, **params)
        self.assertFalse((x_test == x_test_adv).all())
        self.assertLessEqual(np.amax(x_test_adv), 1.000001)
        self.assertGreaterEqual(np.amin(x_test_adv), -1e-6)
        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(krc.predict(x_test_adv), axis=1)
        logger.debug('CW0 Target: %s', target)
        logger.debug('CW0 Actual: %s', y_pred_adv)
        logger.info('CW0 Success Rate: %.2f', (np.sum(target == y_pred_adv) / float(len(target))))
        self.assertTrue((target == y_pred_adv).any())

        # Second attack
        clinfm = CarliniLInfMethod(classifier=krc, targeted=False, max_iter=10, eps=0.5)
        x_test_adv = clinfm.generate(x_test)
        self.assertLessEqual(np.amax(x_test_adv), 1.000001)
        self.assertGreaterEqual(np.amin(x_test_adv), -1e-6)
        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(krc.predict(x_test_adv), axis=1)
        logger.debug('CW0 Target: %s', target)
        logger.debug('CW0 Actual: %s', y_pred_adv)
        logger.info('CW0 Success Rate: %.2f', (np.sum(target != y_pred_adv) / float(len(target))))
        self.assertTrue((target != y_pred_adv).any())

        # Clean-up
        k.clear_session()

    def test_pytorch_mnist_LInf(self):
        """
        Third test with the PyTorchClassifier.
        :return:
        """
        (_, _), (x_test, y_test) = self.mnist
        x_test = np.reshape(x_test, (x_test.shape[0], 1, 28, 28)).astype(np.float32)

        # Build PyTorchClassifier
        ptc = get_classifier_pt(from_logits=True)

        # First attack
        clinfm = CarliniLInfMethod(classifier=ptc, targeted=True, max_iter=10, eps=0.5)
        params = {'y': random_targets(y_test, ptc.nb_classes())}
        x_test_adv = clinfm.generate(x_test, **params)
        self.assertFalse((x_test == x_test_adv).all())
        self.assertLessEqual(np.amax(x_test_adv), 1.0 + 1e-6)
        self.assertGreaterEqual(np.amin(x_test_adv), -1e-6)
        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(ptc.predict(x_test_adv), axis=1)
        self.assertTrue((target == y_pred_adv).any())

        # Second attack
        clinfm = CarliniLInfMethod(classifier=ptc, targeted=False, max_iter=10, eps=0.5)
        x_test_adv = clinfm.generate(x_test)
        self.assertLessEqual(np.amax(x_test_adv), 1.0 + 1e-6)
        self.assertGreaterEqual(np.amin(x_test_adv), -1e-6)

        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(ptc.predict(x_test_adv), axis=1)
        self.assertTrue((target != y_pred_adv).any())

    def test_classifier_type_check_fail_classifier_LInf(self):
        # Use a useless test classifier to test basic classifier properties
        class ClassifierNoAPI:
            pass

        classifier = ClassifierNoAPI
        with self.assertRaises(TypeError) as context:
            _ = CarliniLInfMethod(classifier=classifier)

        self.assertIn('For `CarliniLInfMethod` classifier must be an instance of '
                      '`art.classifiers.classifier.Classifier`, the provided classifier is instance of '
                      '(<class \'object\'>,).', str(context.exception))

    def test_classifier_type_check_fail_gradients_LInf(self):
        # Use a test classifier not providing gradients required by white-box attack
        from art.classifiers.scikitlearn import ScikitlearnDecisionTreeClassifier
        from sklearn.tree import DecisionTreeClassifier

        classifier = ScikitlearnDecisionTreeClassifier(model=DecisionTreeClassifier())
        with self.assertRaises(TypeError) as context:
            _ = CarliniLInfMethod(classifier=classifier)

        self.assertIn('For `CarliniLInfMethod` classifier must be an instance of '
                      '`art.classifiers.classifier.ClassifierGradients`, the provided classifier is instance of '
                      '(<class \'art.classifiers.scikitlearn.ScikitlearnClassifier\'>,).', str(context.exception))

    def test_keras_iris_clipped_LInf(self):
        (_, _), (x_test, y_test) = self.iris
        classifier = get_iris_classifier_kr()
        attack = CarliniLInfMethod(classifier, targeted=False, max_iter=10, eps=0.5)
        x_test_adv = attack.generate(x_test)
        self.assertFalse((x_test == x_test_adv).all())
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)

        predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(y_test, axis=1) == predictions_adv).all())
        accuracy = np.sum(predictions_adv == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info('Accuracy on Iris with C&W adversarial examples: %.2f%%', (accuracy * 100))

    def test_keras_iris_unbounded_LInf(self):
        (_, _), (x_test, y_test) = self.iris
        classifier = get_iris_classifier_kr()

        # Recreate a classifier without clip values
        classifier = KerasClassifier(model=classifier._model, use_logits=False, channel_index=1)
        attack = CarliniLInfMethod(classifier, targeted=False, max_iter=10, eps=1)
        x_test_adv = attack.generate(x_test)
        self.assertFalse((x_test == x_test_adv).all())

        predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(y_test, axis=1) == predictions_adv).all())
        accuracy = np.sum(predictions_adv == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info('Accuracy on Iris with C&W adversarial examples: %.2f%%', (accuracy * 100))

    def test_tensorflow_iris_LInf(self):
        (_, _), (x_test, y_test) = self.iris
        classifier, _ = get_iris_classifier_tf()

        # Test untargeted attack
        attack = CarliniLInfMethod(classifier, targeted=False, max_iter=10, eps=0.5)
        x_test_adv = attack.generate(x_test)
        self.assertFalse((x_test == x_test_adv).all())
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)

        predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(y_test, axis=1) == predictions_adv).all())
        accuracy = np.sum(predictions_adv == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info('Accuracy on Iris with C&W adversarial examples: %.2f%%', (accuracy * 100))

        # Test targeted attack
        targets = random_targets(y_test, nb_classes=3)
        attack = CarliniLInfMethod(classifier, targeted=True, max_iter=10, eps=0.5)
        x_test_adv = attack.generate(x_test, **{'y': targets})
        self.assertFalse((x_test == x_test_adv).all())
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)

        predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertTrue((np.argmax(targets, axis=1) == predictions_adv).any())
        accuracy = np.sum(predictions_adv == np.argmax(targets, axis=1)) / y_test.shape[0]
        logger.info('Success rate of targeted C&W on Iris: %.2f%%', (accuracy * 100))

    def test_pytorch_iris_LInf(self):
        (_, _), (x_test, y_test) = self.iris
        classifier = get_iris_classifier_pt()
        attack = CarliniLInfMethod(classifier, targeted=False, max_iter=10, eps=0.5)
        x_test_adv = attack.generate(x_test.astype(np.float32))
        self.assertFalse((x_test == x_test_adv).all())
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)

        predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(y_test, axis=1) == predictions_adv).all())
        accuracy = np.sum(predictions_adv == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info('Accuracy on Iris with C&W adversarial examples: %.2f%%', (accuracy * 100))

    def test_scikitlearn_LInf(self):
        from sklearn.linear_model import LogisticRegression

        from art.classifiers.scikitlearn import ScikitlearnLogisticRegression

        scikitlearn_test_cases = {LogisticRegression: ScikitlearnLogisticRegression}  # ,
        # SVC: ScikitlearnSVC,
        # LinearSVC: ScikitlearnSVC}

        (_, _), (x_test, y_test) = self.iris

        for (model_class, classifier_class) in scikitlearn_test_cases.items():
            model = model_class()
            classifier = classifier_class(model=model, clip_values=(0, 1))
            classifier.fit(x=x_test, y=y_test)

            # Test untargeted attack
            attack = CarliniLInfMethod(classifier, targeted=False, max_iter=10, eps=0.5)
            x_test_adv = attack.generate(x_test)
            self.assertFalse((x_test == x_test_adv).all())
            self.assertLessEqual(np.amax(x_test_adv), 1.0)
            self.assertGreaterEqual(np.amin(x_test_adv), 0.0)

            predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
            self.assertFalse((np.argmax(y_test, axis=1) == predictions_adv).all())
            accuracy = np.sum(predictions_adv == np.argmax(y_test, axis=1)) / y_test.shape[0]
            logger.info('Accuracy of ' + classifier.__class__.__name__ + ' on Iris with C&W adversarial examples: '
                                                                         '%.2f%%', (accuracy * 100))

            # Test targeted attack
            targets = random_targets(y_test, nb_classes=3)
            attack = CarliniLInfMethod(classifier, targeted=True, max_iter=10, eps=0.5)
            x_test_adv = attack.generate(x_test, **{'y': targets})
            self.assertFalse((x_test == x_test_adv).all())
            self.assertLessEqual(np.amax(x_test_adv), 1.0)
            self.assertGreaterEqual(np.amin(x_test_adv), 0.0)

            predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
            self.assertTrue((np.argmax(targets, axis=1) == predictions_adv).any())
            accuracy = np.sum(predictions_adv == np.argmax(targets, axis=1)) / y_test.shape[0]
            logger.info('Success rate of ' + classifier.__class__.__name__ + ' on targeted C&W on Iris: %.2f%%',
                        (accuracy * 100))


if __name__ == '__main__':
    unittest.main()
