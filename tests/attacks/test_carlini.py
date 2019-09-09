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

from art.attacks import CarliniL2Method, CarliniLInfMethod
from art.classifiers import KerasClassifier
from art.utils import load_dataset, random_targets, master_seed
from art.utils_test import get_classifier_tf, get_classifier_kr, get_classifier_pt
from art.utils_test import get_iris_classifier_tf, get_iris_classifier_kr, get_iris_classifier_pt

logger = logging.getLogger('testLogger')

BATCH_SIZE = 100
NB_TRAIN = 10
NB_TEST = 10


class TestCarliniL2(unittest.TestCase):
    """
    A unittest class for testing the Carlini L2 attack.
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

    @unittest.skipIf(tf.__version__[0] == '2', reason='Skip unittests for Tensorflow v2 until Keras supports Tensorflow'
                                                      ' v2 as backend.')
    def test_failure_attack(self):
        """
        Test the corner case when attack is failed.
        :return:
        """
        # Build TensorFlowClassifier
        tfc, sess = get_classifier_tf()

        # Failure attack
        cl2m = CarliniL2Method(classifier=tfc, targeted=True, max_iter=0, binary_search_steps=0, learning_rate=0,
                               initial_const=1)
        params = {'y': random_targets(self.y_test, tfc.nb_classes())}
        x_test_adv = cl2m.generate(self.x_test, **params)
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
        np.testing.assert_array_almost_equal(self.x_test, x_test_adv, decimal=3)

        # Clean-up session
        sess.close()

    def test_tfclassifier(self):
        """
        First test with the TensorFlowClassifier.
        :return:
        """
        # Build TensorFlowClassifier
        tfc, sess = get_classifier_tf()

        # First attack
        cl2m = CarliniL2Method(classifier=tfc, targeted=True, max_iter=10)
        params = {'y': random_targets(self.y_test, tfc.nb_classes())}
        x_test_adv = cl2m.generate(self.x_test, **params)
        self.assertFalse((self.x_test == x_test_adv).all())
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
        x_test_adv = cl2m.generate(self.x_test)
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        logger.debug('CW2 Target: %s', target)
        logger.debug('CW2 Actual: %s', y_pred_adv)
        logger.info('CW2 Success Rate: %.2f', (np.sum(target == y_pred_adv) / float(len(target))))
        self.assertTrue((target != y_pred_adv).any())

        # Clean-up session
        sess.close()

    @unittest.skipIf(tf.__version__[0] == '2', reason='Skip unittests for Tensorflow v2 until Keras supports Tensorflow'
                                                      ' v2 as backend.')
    def test_krclassifier(self):
        """
        Second test with the KerasClassifier.
        :return:
        """
        # Build KerasClassifier
        krc = get_classifier_kr()

        # First attack
        cl2m = CarliniL2Method(classifier=krc, targeted=True, max_iter=10)
        params = {'y': random_targets(self.y_test, krc.nb_classes())}
        x_test_adv = cl2m.generate(self.x_test, **params)
        self.assertFalse((self.x_test == x_test_adv).all())
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(krc.predict(x_test_adv), axis=1)
        logger.debug('CW2 Target: %s', target)
        logger.debug('CW2 Actual: %s', y_pred_adv)
        logger.info('CW2 Success Rate: %.2f', (np.sum(target == y_pred_adv) / float(len(target))))
        self.assertTrue((target == y_pred_adv).any())

        # Second attack
        cl2m = CarliniL2Method(classifier=krc, targeted=False, max_iter=10)
        x_test_adv = cl2m.generate(self.x_test)
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(krc.predict(x_test_adv), axis=1)
        logger.debug('CW2 Target: %s', target)
        logger.debug('CW2 Actual: %s', y_pred_adv)
        logger.info('CW2 Success Rate: %.2f', (np.sum(target != y_pred_adv) / float(len(target))))
        self.assertTrue((target != y_pred_adv).any())

        # Clean-up
        k.clear_session()

    def test_ptclassifier(self):
        """
        Third test with the PyTorchClassifier.
        :return:
        """
        # Build PyTorchClassifier
        ptc = get_classifier_pt()

        x_test = np.swapaxes(self.x_test, 1, 3).astype(np.float32)

        # First attack
        cl2m = CarliniL2Method(classifier=ptc, targeted=True, max_iter=10)
        params = {'y': random_targets(self.y_test, ptc.nb_classes())}
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

    def test_classifier_type_check_fail_classifier(self):
        # Use a useless test classifier to test basic classifier properties
        class ClassifierNoAPI:
            pass

        classifier = ClassifierNoAPI
        with self.assertRaises(TypeError) as context:
            _ = CarliniL2Method(classifier=classifier)

        self.assertIn('For `CarliniL2Method` classifier must be an instance of `art.classifiers.classifier.Classifier`,'
                      ' the provided classifier is instance of (<class \'object\'>,).', str(context.exception))

    def test_classifier_type_check_fail_gradients(self):
        # Use a test classifier not providing gradients required by white-box attack
        from art.classifiers.scikitlearn import ScikitlearnDecisionTreeClassifier
        from sklearn.tree import DecisionTreeClassifier

        classifier = ScikitlearnDecisionTreeClassifier(model=DecisionTreeClassifier())
        with self.assertRaises(TypeError) as context:
            _ = CarliniL2Method(classifier=classifier)

        self.assertIn('For `CarliniL2Method` classifier must be an instance of '
                      '`art.classifiers.classifier.ClassifierGradients`, the provided classifier is instance of '
                      '(<class \'art.classifiers.scikitlearn.ScikitlearnClassifier\'>,).', str(context.exception))


class TestCarliniL2Vectors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (x_train, y_train), (x_test, y_test), _, _ = load_dataset('iris')

        cls.x_train = x_train
        cls.y_train = y_train
        cls.x_test = x_test
        cls.y_test = y_test

    def setUp(self):
        master_seed(1234)

    @unittest.skipIf(tf.__version__[0] == '2', reason='Skip unittests for Tensorflow v2 until Keras supports Tensorflow'
                                                      ' v2 as backend.')
    def test_iris_k_clipped(self):
        classifier, _ = get_iris_classifier_kr()
        attack = CarliniL2Method(classifier, targeted=False, max_iter=10)
        x_test_adv = attack.generate(self.x_test)
        self.assertFalse((self.x_test == x_test_adv).all())
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)

        predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test, axis=1) == predictions_adv).all())
        accuracy = np.sum(predictions_adv == np.argmax(self.y_test, axis=1)) / self.y_test.shape[0]
        logger.info('Accuracy on Iris with C&W adversarial examples: %.2f%%', (accuracy * 100))

    @unittest.skipIf(tf.__version__[0] == '2', reason='Skip unittests for Tensorflow v2 until Keras supports Tensorflow'
                                                      ' v2 as backend.')
    def test_iris_k_unbounded(self):
        classifier, _ = get_iris_classifier_kr()

        # Recreate a classifier without clip values
        classifier = KerasClassifier(model=classifier._model, use_logits=False, channel_index=1)
        attack = CarliniL2Method(classifier, targeted=False, max_iter=10)
        x_test_adv = attack.generate(self.x_test)
        self.assertFalse((self.x_test == x_test_adv).all())

        predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test, axis=1) == predictions_adv).all())
        accuracy = np.sum(predictions_adv == np.argmax(self.y_test, axis=1)) / self.y_test.shape[0]
        logger.info('Accuracy on Iris with C&W adversarial examples: %.2f%%', (accuracy * 100))

    def test_iris_tf(self):
        classifier, _ = get_iris_classifier_tf()

        # Test untargeted attack
        attack = CarliniL2Method(classifier, targeted=False, max_iter=10)
        x_test_adv = attack.generate(self.x_test)
        self.assertFalse((self.x_test == x_test_adv).all())
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)

        predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test, axis=1) == predictions_adv).all())
        accuracy = np.sum(predictions_adv == np.argmax(self.y_test, axis=1)) / self.y_test.shape[0]
        logger.info('Accuracy on Iris with C&W adversarial examples: %.2f%%', (accuracy * 100))

        # Test targeted attack
        targets = random_targets(self.y_test, nb_classes=3)
        attack = CarliniL2Method(classifier, targeted=True, max_iter=10)
        x_test_adv = attack.generate(self.x_test, **{'y': targets})
        self.assertFalse((self.x_test == x_test_adv).all())
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)

        predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertTrue((np.argmax(targets, axis=1) == predictions_adv).any())
        accuracy = np.sum(predictions_adv == np.argmax(targets, axis=1)) / self.y_test.shape[0]
        logger.info('Success rate of targeted C&W on Iris: %.2f%%', (accuracy * 100))

    def test_iris_pt(self):
        classifier = get_iris_classifier_pt()
        attack = CarliniL2Method(classifier, targeted=False, max_iter=10)
        x_test_adv = attack.generate(self.x_test)
        self.assertFalse((self.x_test == x_test_adv).all())
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)

        predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test, axis=1) == predictions_adv).all())
        accuracy = np.sum(predictions_adv == np.argmax(self.y_test, axis=1)) / self.y_test.shape[0]
        logger.info('Accuracy on Iris with C&W adversarial examples: %.2f%%', (accuracy * 100))

    def test_scikitlearn(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC, LinearSVC

        from art.classifiers.scikitlearn import ScikitlearnLogisticRegression, ScikitlearnSVC

        scikitlearn_test_cases = {LogisticRegression: ScikitlearnLogisticRegression}  # ,
        # SVC: ScikitlearnSVC,
        # LinearSVC: ScikitlearnSVC}

        for (model_class, classifier_class) in scikitlearn_test_cases.items():
            model = model_class()
            classifier = classifier_class(model=model, clip_values=(0, 1))
            classifier.fit(x=self.x_test, y=self.y_test)

            # Test untargeted attack
            attack = CarliniL2Method(classifier, targeted=False, max_iter=10)
            x_test_adv = attack.generate(self.x_test)
            self.assertFalse((self.x_test == x_test_adv).all())
            self.assertLessEqual(np.amax(x_test_adv), 1.0)
            self.assertGreaterEqual(np.amin(x_test_adv), 0.0)

            predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
            self.assertFalse((np.argmax(self.y_test, axis=1) == predictions_adv).all())
            accuracy = np.sum(predictions_adv == np.argmax(self.y_test, axis=1)) / self.y_test.shape[0]
            logger.info('Accuracy of ' + classifier.__class__.__name__ + ' on Iris with C&W adversarial examples: '
                                                                         '%.2f%%', (accuracy * 100))

            # Test targeted attack
            targets = random_targets(self.y_test, nb_classes=3)
            attack = CarliniL2Method(classifier, targeted=True, max_iter=10)
            x_test_adv = attack.generate(self.x_test, **{'y': targets})
            self.assertFalse((self.x_test == x_test_adv).all())
            self.assertLessEqual(np.amax(x_test_adv), 1.0)
            self.assertGreaterEqual(np.amin(x_test_adv), 0.0)

            predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
            self.assertTrue((np.argmax(targets, axis=1) == predictions_adv).any())
            accuracy = np.sum(predictions_adv == np.argmax(targets, axis=1)) / self.y_test.shape[0]
            logger.info('Success rate of ' + classifier.__class__.__name__ + ' on targeted C&W on Iris: %.2f%%',
                        (accuracy * 100))


class TestCarliniLInf(TestCarliniL2):
    """
    A unittest class for testing the Carlini LInf attack.
    """

    def test_failure_attack(self):
        """
        Test the corner case when attack is failed.
        :return:
        """
        # Build TensorFlowClassifier
        tfc, sess = get_classifier_tf()

        # Failure attack
        clinfm = CarliniLInfMethod(classifier=tfc, targeted=True, max_iter=0, learning_rate=0, eps=0.5)
        params = {'y': random_targets(self.y_test, tfc.nb_classes())}
        x_test_adv = clinfm.generate(self.x_test, **params)
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
        self.assertTrue(np.allclose(self.x_test, x_test_adv, atol=1e-3))

        # Clean-up session
        sess.close()

    def test_tfclassifier(self):
        """
        First test with the TensorFlowClassifier.
        :return:
        """
        # Build TensorFlowClassifier
        tfc, sess = get_classifier_tf()

        # First attack
        clinfm = CarliniLInfMethod(classifier=tfc, targeted=True, max_iter=10, eps=0.5)
        params = {'y': random_targets(self.y_test, tfc.nb_classes())}
        x_test_adv = clinfm.generate(self.x_test, **params)
        self.assertFalse((self.x_test == x_test_adv).all())
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
        x_test_adv = clinfm.generate(self.x_test)
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

    @unittest.skipIf(tf.__version__[0] == '2', reason='Skip unittests for Tensorflow v2 until Keras supports Tensorflow'
                                                      ' v2 as backend.')
    def test_krclassifier(self):
        """
        Second test with the KerasClassifier.
        :return:
        """
        # Build KerasClassifier
        krc, sess = get_classifier_tf()

        # First attack
        clinfm = CarliniLInfMethod(classifier=krc, targeted=True, max_iter=10, eps=0.5)
        params = {'y': random_targets(self.y_test, krc.nb_classes())}
        x_test_adv = clinfm.generate(self.x_test, **params)
        self.assertFalse((self.x_test == x_test_adv).all())
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(krc.predict(x_test_adv), axis=1)
        logger.debug('CW0 Target: %s', target)
        logger.debug('CW0 Actual: %s', y_pred_adv)
        logger.info('CW0 Success Rate: %.2f', (np.sum(target == y_pred_adv) / float(len(target))))
        self.assertTrue((target == y_pred_adv).any())

        # Second attack
        clinfm = CarliniLInfMethod(classifier=krc, targeted=False, max_iter=10, eps=0.5)
        x_test_adv = clinfm.generate(self.x_test)
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), -1e-6)
        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(krc.predict(x_test_adv), axis=1)
        logger.debug('CW0 Target: %s', target)
        logger.debug('CW0 Actual: %s', y_pred_adv)
        logger.info('CW0 Success Rate: %.2f', (np.sum(target != y_pred_adv) / float(len(target))))
        self.assertTrue((target != y_pred_adv).any())

        # Clean-up
        k.clear_session()
        sess.close()

    def test_ptclassifier(self):
        """
        Third test with the PyTorchClassifier.
        :return:
        """
        # Build PyTorchClassifier
        ptc = get_classifier_pt()

        x_test = np.swapaxes(self.x_test, 1, 3).astype(np.float32)

        # First attack
        clinfm = CarliniLInfMethod(classifier=ptc, targeted=True, max_iter=10, eps=0.5)
        params = {'y': random_targets(self.y_test, ptc.nb_classes())}
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

    def test_classifier_type_check_fail_classifier(self):
        # Use a useless test classifier to test basic classifier properties
        class ClassifierNoAPI:
            pass

        classifier = ClassifierNoAPI
        with self.assertRaises(TypeError) as context:
            _ = CarliniLInfMethod(classifier=classifier)

        self.assertIn('For `CarliniLInfMethod` classifier must be an instance of '
                      '`art.classifiers.classifier.Classifier`, the provided classifier is instance of '
                      '(<class \'object\'>,).', str(context.exception))

    def test_classifier_type_check_fail_gradients(self):
        # Use a test classifier not providing gradients required by white-box attack
        from art.classifiers.scikitlearn import ScikitlearnDecisionTreeClassifier
        from sklearn.tree import DecisionTreeClassifier

        classifier = ScikitlearnDecisionTreeClassifier(model=DecisionTreeClassifier())
        with self.assertRaises(TypeError) as context:
            _ = CarliniLInfMethod(classifier=classifier)

        self.assertIn('For `CarliniLInfMethod` classifier must be an instance of '
                      '`art.classifiers.classifier.ClassifierGradients`, the provided classifier is instance of '
                      '(<class \'art.classifiers.scikitlearn.ScikitlearnClassifier\'>,).', str(context.exception))


class TestCarliniLInfVectors(TestCarliniL2Vectors):
    @unittest.skipIf(tf.__version__[0] == '2', reason='Skip unittests for Tensorflow v2 until Keras supports Tensorflow'
                                                      ' v2 as backend.')
    def test_iris_k_clipped(self):
        classifier, _ = get_iris_classifier_kr()
        attack = CarliniLInfMethod(classifier, targeted=False, max_iter=10, eps=0.5)
        x_test_adv = attack.generate(self.x_test)
        self.assertFalse((self.x_test == x_test_adv).all())
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)

        predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test, axis=1) == predictions_adv).all())
        accuracy = np.sum(predictions_adv == np.argmax(self.y_test, axis=1)) / self.y_test.shape[0]
        logger.info('Accuracy on Iris with C&W adversarial examples: %.2f%%', (accuracy * 100))

    @unittest.skipIf(tf.__version__[0] == '2', reason='Skip unittests for Tensorflow v2 until Keras supports Tensorflow'
                                                      ' v2 as backend.')
    def test_iris_k_unbounded(self):
        classifier, _ = get_iris_classifier_kr()

        # Recreate a classifier without clip values
        classifier = KerasClassifier(model=classifier._model, use_logits=False, channel_index=1)
        attack = CarliniLInfMethod(classifier, targeted=False, max_iter=10, eps=1)
        x_test_adv = attack.generate(self.x_test)
        self.assertFalse((self.x_test == x_test_adv).all())

        predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test, axis=1) == predictions_adv).all())
        accuracy = np.sum(predictions_adv == np.argmax(self.y_test, axis=1)) / self.y_test.shape[0]
        logger.info('Accuracy on Iris with C&W adversarial examples: %.2f%%', (accuracy * 100))

    def test_iris_tf(self):
        classifier, _ = get_iris_classifier_tf()

        # Test untargeted attack
        attack = CarliniLInfMethod(classifier, targeted=False, max_iter=10, eps=0.5)
        x_test_adv = attack.generate(self.x_test)
        self.assertFalse((self.x_test == x_test_adv).all())
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)

        predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test, axis=1) == predictions_adv).all())
        accuracy = np.sum(predictions_adv == np.argmax(self.y_test, axis=1)) / self.y_test.shape[0]
        logger.info('Accuracy on Iris with C&W adversarial examples: %.2f%%', (accuracy * 100))

        # Test targeted attack
        targets = random_targets(self.y_test, nb_classes=3)
        attack = CarliniLInfMethod(classifier, targeted=True, max_iter=10, eps=0.5)
        x_test_adv = attack.generate(self.x_test, **{'y': targets})
        self.assertFalse((self.x_test == x_test_adv).all())
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)

        predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertTrue((np.argmax(targets, axis=1) == predictions_adv).any())
        accuracy = np.sum(predictions_adv == np.argmax(targets, axis=1)) / self.y_test.shape[0]
        logger.info('Success rate of targeted C&W on Iris: %.2f%%', (accuracy * 100))

    def test_iris_pt(self):
        classifier = get_iris_classifier_pt()
        attack = CarliniLInfMethod(classifier, targeted=False, max_iter=10, eps=0.5)
        x_test_adv = attack.generate(self.x_test.astype(np.float32))
        self.assertFalse((self.x_test == x_test_adv).all())
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)

        predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test, axis=1) == predictions_adv).all())
        accuracy = np.sum(predictions_adv == np.argmax(self.y_test, axis=1)) / self.y_test.shape[0]
        logger.info('Accuracy on Iris with C&W adversarial examples: %.2f%%', (accuracy * 100))

    def test_scikitlearn(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC, LinearSVC

        from art.classifiers.scikitlearn import ScikitlearnLogisticRegression, ScikitlearnSVC

        scikitlearn_test_cases = {LogisticRegression: ScikitlearnLogisticRegression}#,
                                  # SVC: ScikitlearnSVC,
                                  # LinearSVC: ScikitlearnSVC}

        for (model_class, classifier_class) in scikitlearn_test_cases.items():
            model = model_class()
            classifier = classifier_class(model=model, clip_values=(0, 1))
            classifier.fit(x=self.x_test, y=self.y_test)

            # Test untargeted attack
            attack = CarliniLInfMethod(classifier, targeted=False, max_iter=10, eps=0.5)
            x_test_adv = attack.generate(self.x_test)
            self.assertFalse((self.x_test == x_test_adv).all())
            self.assertLessEqual(np.amax(x_test_adv), 1.0)
            self.assertGreaterEqual(np.amin(x_test_adv), 0.0)

            predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
            self.assertFalse((np.argmax(self.y_test, axis=1) == predictions_adv).all())
            accuracy = np.sum(predictions_adv == np.argmax(self.y_test, axis=1)) / self.y_test.shape[0]
            logger.info('Accuracy of ' + classifier.__class__.__name__ + ' on Iris with C&W adversarial examples: '
                                                                         '%.2f%%', (accuracy * 100))

            # Test targeted attack
            targets = random_targets(self.y_test, nb_classes=3)
            attack = CarliniLInfMethod(classifier, targeted=True, max_iter=10, eps=0.5)
            x_test_adv = attack.generate(self.x_test, **{'y': targets})
            self.assertFalse((self.x_test == x_test_adv).all())
            self.assertLessEqual(np.amax(x_test_adv), 1.0)
            self.assertGreaterEqual(np.amin(x_test_adv), 0.0)

            predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
            self.assertTrue((np.argmax(targets, axis=1) == predictions_adv).any())
            accuracy = np.sum(predictions_adv == np.argmax(targets, axis=1)) / self.y_test.shape[0]
            logger.info('Success rate of ' + classifier.__class__.__name__ + ' on targeted C&W on Iris: %.2f%%',
                        (accuracy * 100))


if __name__ == '__main__':
    unittest.main()
