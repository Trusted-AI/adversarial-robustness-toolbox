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

from art.attacks import ElasticNet
from art.classifiers import KerasClassifier
from art.utils import load_dataset, random_targets, master_seed, to_categorical
from tests.utils_test import get_classifier_tf, get_classifier_kr
from tests.utils_test import get_classifier_pt, get_iris_classifier_tf, get_iris_classifier_kr, get_iris_classifier_pt

logger = logging.getLogger(__name__)

BATCH_SIZE = 100
NB_TRAIN = 500
NB_TEST = 10


class TestElasticNet(unittest.TestCase):
    """
    A unittest class for testing the ElasticNet attack.
    """

    @classmethod
    def setUpClass(cls):
        # MNIST
        (x_train, y_train), (x_test, y_test), _, _ = load_dataset('mnist')
        x_train, y_train, x_test, y_test = x_train[:NB_TRAIN], y_train[:NB_TRAIN], x_test[:NB_TEST], y_test[:NB_TEST]
        cls.mnist = (x_train, y_train), (x_test, y_test)

        # Iris
        np.random.seed(1234)
        (x_train, y_train), (x_test, y_test), _, _ = load_dataset('iris')
        cls.iris = (x_train, y_train), (x_test, y_test)

    def setUp(self):
        master_seed(1234)

    def test_tensorflow_failure_attack(self):
        """
        Test the corner case when attack fails.
        :return:
        """
        (_, _), (x_test, y_test) = self.mnist

        # Build TensorFlowClassifier
        tfc, sess = get_classifier_tf()

        # Failure attack
        ead = ElasticNet(classifier=tfc, targeted=True, max_iter=0, binary_search_steps=0, learning_rate=0,
                         initial_const=1)
        params = {'y': random_targets(y_test, tfc.nb_classes())}
        x_test_adv = ead.generate(x_test, **params)
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
        np.testing.assert_almost_equal(x_test, x_test_adv, 3)

        # Clean-up session
        sess.close()

    def test_tensorflow_mnist(self):
        """
        First test with the TensorFlowClassifier.
        :return:
        """
        (_, _), (x_test, y_test) = self.mnist
        x_test_original = x_test.copy()

        # Build TensorFlowClassifier
        tfc, sess = get_classifier_tf(from_logits=True)

        # First attack
        ead = ElasticNet(classifier=tfc, targeted=True, max_iter=2)
        params = {'y': random_targets(y_test, tfc.nb_classes())}
        x_test_adv = ead.generate(x_test, **params)
        expected_x_test_adv = np.asarray([0.45704955, 0.43627003, 0.57238287, 1.0, 0.11541145, 0.12619308,
                                          0.48318917, 0.3457903, 0.17863746, 0.09060935, 0.0, 0.00963121,
                                          0.0, 0.04749763, 0.4058206, 0.17860745, 0.0, 0.9153206,
                                          0.84564775, 0.20603634, 0.10586322, 0.00947509, 0.0, 0.0,
                                          0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(x_test_adv[0, 14, :, 0], expected_x_test_adv, decimal=6)
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        logger.debug('EAD target: %s', target)
        logger.debug('EAD actual: %s', y_pred_adv)
        logger.info('EAD success rate on MNIST: %.2f%%', (100 * sum(target == y_pred_adv) / len(target)))
        self.assertTrue((target == y_pred_adv).any())

        # Second attack
        ead = ElasticNet(classifier=tfc, targeted=False, max_iter=2)
        params = {'y': random_targets(y_test, tfc.nb_classes())}
        x_test_adv = ead.generate(x_test, **params)
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        logger.debug('EAD target: %s', target)
        logger.debug('EAD actual: %s', y_pred_adv)
        logger.info('EAD success rate on MNIST: %.2f%%', (100 * sum(target != y_pred_adv) / float(len(target))))
        np.testing.assert_array_equal(y_pred_adv, np.asarray([7, 1, 1, 4, 4, 1, 4, 4, 4, 4]))

        # Third attack
        ead = ElasticNet(classifier=tfc, targeted=False, max_iter=2)
        params = {}
        x_test_adv = ead.generate(x_test, **params)
        expected_x_test_adv = np.asarray([0.22866514, 0.21826893, 0.22902338, 0.06268515, 0.0, 0.0,
                                          0.04822975, 0.0, 0.0, 0.0, 0.05555382, 0.0,
                                          0.0, 0.0, 0.38986346, 0.10653087, 0.32385707, 0.98043066,
                                          0.75790393, 0.16486718, 0.16069527, 0.0, 0.0, 0.0,
                                          0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(x_test_adv[0, 14, :, 0], expected_x_test_adv, decimal=6)
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
        y_pred = np.argmax(tfc.predict(x_test), axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        logger.debug('EAD target: %s', y_pred)
        logger.debug('EAD actual: %s', y_pred_adv)
        logger.info('EAD success rate: %.2f%%', (100 * sum(y_pred != y_pred_adv) / float(len(y_pred))))
        np.testing.assert_array_equal(y_pred_adv, np.asarray([0, 4, 7, 9, 0, 7, 7, 3, 0, 7]))

        # First attack without batching
        ead_wob = ElasticNet(classifier=tfc, targeted=True, max_iter=2, batch_size=1)
        params = {'y': random_targets(y_test, tfc.nb_classes())}
        x_test_adv = ead_wob.generate(x_test, **params)
        expected_x_test_adv = np.asarray([0.3287169, 0.31374657, 0.42853343, 0.8994576, 0.19850709, 0.11997936,
                                          0.5622535, 0.43854535, 0.19387433, 0.12516324, 0.0, 0.10933565,
                                          0.02162433, 0.07120894, 0.95224255, 0.3072921, 0.48966524, 1.,
                                          0.3814998, 0.15782641, 0.52283823, 0.12852049, 0.0, 0.0,
                                          0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(x_test_adv[0, 14, :, 0], expected_x_test_adv, decimal=6)
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        logger.debug('EAD target: %s', target)
        logger.debug('EAD actual: %s', y_pred_adv)
        logger.info('EAD success rate: %.2f%%', (100 * sum(target == y_pred_adv) / float(len(target))))
        self.assertTrue((target == y_pred_adv).any())

        # Second attack without batching
        ead_wob = ElasticNet(classifier=tfc, targeted=False, max_iter=2, batch_size=1)
        params = {'y': random_targets(y_test, tfc.nb_classes())}
        x_test_adv = ead_wob.generate(x_test, **params)
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        logger.debug('EAD target: %s', target)
        logger.debug('EAD actual: %s', y_pred_adv)
        logger.info('EAD success rate: %.2f%%', (100 * sum(target != y_pred_adv) / float(len(target))))
        np.testing.assert_array_equal(y_pred_adv, np.asarray([7, 1, 1, 4, 4, 1, 4, 4, 4, 4]))

        # Check that x_test has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test))), 0.0, delta=0.00001)

        # Close session
        sess.close()

    def test_keras_mnist(self):
        """
        Second test with the KerasClassifier.
        :return:
        """
        (_, _), (x_test, y_test) = self.mnist
        x_test_original = x_test.copy()

        # Build KerasClassifier
        krc = get_classifier_kr()

        # First attack
        ead = ElasticNet(classifier=krc, targeted=True, max_iter=2)
        y_target = to_categorical(np.asarray([6, 6, 7, 4, 9, 7, 9, 0, 1, 0]), nb_classes=10)
        x_test_adv = ead.generate(x_test, y=y_target)
        expected_x_test_adv = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          0.0, 0.0, 0.0, 0.0, 0.00183569, 0.0,
                                          0.0, 0.49765405, 1., 0.6467149, 0.0033755, 0.0052456,
                                          0.0, 0.01104407, 0.00495547, 0.02747423, 0.0, 0.0,
                                          0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(x_test_adv[2, 14, :, 0], expected_x_test_adv, decimal=6)
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
        target = np.argmax(y_target, axis=1)
        y_pred_adv = np.argmax(krc.predict(x_test_adv), axis=1)
        logger.debug('EAD target: %s', target)
        logger.debug('EAD actual: %s', y_pred_adv)
        logger.info('EAD success rate: %.2f%%', (100 * sum(target == y_pred_adv) / float(len(target))))
        self.assertTrue((target == y_pred_adv).any())

        # Second attack
        ead = ElasticNet(classifier=krc, targeted=False, max_iter=2)
        y_target = to_categorical(np.asarray([9, 5, 6, 7, 1, 6, 1, 5, 8, 5]), nb_classes=10)
        x_test_adv = ead.generate(x_test, y=y_target)
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
        y_pred_adv = np.argmax(krc.predict(x_test_adv), axis=1)
        logger.debug('EAD target: %s', y_target)
        logger.debug('EAD actual: %s', y_pred_adv)
        logger.info('EAD success rate: %.2f', (100 * sum(target != y_pred_adv) / float(len(target))))
        self.assertTrue((target != y_pred_adv).any())
        np.testing.assert_array_equal(y_pred_adv, np.asarray([7, 1, 1, 4, 4, 1, 4, 4, 4, 4]))

        # Check that x_test has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test))), 0.0, delta=0.00001)

        k.clear_session()

    def test_pytorch_mnist(self):
        """
        Third test with the PyTorchClassifier.
        :return:
        """
        (_, _), (x_test, y_test) = self.mnist
        x_test = np.reshape(x_test, (x_test.shape[0], 1, 28, 28)).astype(np.float32)
        x_test_original = x_test.copy()

        # Build PyTorchClassifier
        ptc = get_classifier_pt(from_logits=False)

        # First attack
        ead = ElasticNet(classifier=ptc, targeted=True, max_iter=2)
        params = {'y': random_targets(y_test, ptc.nb_classes())}
        x_test_adv = ead.generate(x_test, **params)
        expected_x_test_adv = np.asarray([0.01678124, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00665895, 0.0, 0.11374763,
                                          0.36250514, 0.5472948, 0.9308808, 1.0, 0.99920374, 0.86274165, 0.6346757,
                                          0.5597227, 0.24191494, 0.25882354, 0.0091916, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(x_test_adv[2, 0, :, 14], expected_x_test_adv, decimal=6)
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(ptc.predict(x_test_adv), axis=1)
        self.assertTrue((target == y_pred_adv).any())

        # Second attack
        ead = ElasticNet(classifier=ptc, targeted=False, max_iter=2)
        params = {'y': random_targets(y_test, ptc.nb_classes())}
        x_test_adv = ead.generate(x_test, **params)
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(ptc.predict(x_test_adv), axis=1)
        self.assertTrue((target != y_pred_adv).any())
        np.testing.assert_array_equal(y_pred_adv, np.asarray([7, 1, 1, 4, 4, 1, 4, 4, 4, 4]))

        # Check that x_test has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test))), 0.0, delta=0.00001)

    def test_classifier_type_check_fail_classifier(self):
        # Use a useless test classifier to test basic classifier properties
        class ClassifierNoAPI:
            pass

        classifier = ClassifierNoAPI
        with self.assertRaises(TypeError) as context:
            _ = ElasticNet(classifier=classifier)

        self.assertIn('For `ElasticNet` classifier must be an instance of '
                      '`art.classifiers.classifier.Classifier`, the provided classifier is instance of '
                      '(<class \'object\'>,).', str(context.exception))

    def test_classifier_type_check_fail_gradients(self):
        # Use a test classifier not providing gradients required by white-box attack
        from art.classifiers.scikitlearn import ScikitlearnDecisionTreeClassifier
        from sklearn.tree import DecisionTreeClassifier

        classifier = ScikitlearnDecisionTreeClassifier(model=DecisionTreeClassifier())
        with self.assertRaises(TypeError) as context:
            _ = ElasticNet(classifier=classifier)

        self.assertIn('For `ElasticNet` classifier must be an instance of '
                      '`art.classifiers.classifier.ClassifierGradients`, the provided classifier is instance of '
                      '(<class \'art.classifiers.scikitlearn.ScikitlearnClassifier\'>,).', str(context.exception))

    def test_keras_iris_clipped(self):
        (_, _), (x_test, y_test) = self.iris
        classifier = get_iris_classifier_kr()
        attack = ElasticNet(classifier, targeted=False, max_iter=10)
        x_test_adv = attack.generate(x_test)
        expected_x_test_adv = np.asarray([0.85931635, 0.44633555, 0.65658355, 0.23840423])
        np.testing.assert_array_almost_equal(x_test_adv[0, :], expected_x_test_adv, decimal=6)
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
        predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        np.testing.assert_array_equal(predictions_adv, np.asarray([1, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 2, 2,
                                                                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 0, 1,
                                                                   1, 1, 2, 0, 2, 2, 1, 1, 2]))
        accuracy = 1.0 - np.sum(predictions_adv == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info('EAD success rate on Iris: %.2f%%', (accuracy * 100))

    def test_keras_iris_unbounded(self):
        (_, _), (x_test, y_test) = self.iris
        classifier = get_iris_classifier_kr()

        # Recreate a classifier without clip values
        classifier = KerasClassifier(model=classifier._model, use_logits=False, channel_index=1)
        attack = ElasticNet(classifier, targeted=False, max_iter=10)
        x_test_adv = attack.generate(x_test)
        expected_x_test_adv = np.asarray([0.85931635, 0.44633555, 0.65658355, 0.23840423])
        np.testing.assert_array_almost_equal(x_test_adv[0, :], expected_x_test_adv, decimal=6)
        predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        np.testing.assert_array_equal(predictions_adv, np.asarray([1, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 2, 2,
                                                                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 0, 1,
                                                                   1, 1, 2, 0, 2, 2, 1, 1, 2]))
        accuracy = 1.0 - np.sum(predictions_adv == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info('EAD success rate on Iris: %.2f%%', (accuracy * 100))

    def test_tensorflow_iris(self):
        (_, _), (x_test, y_test) = self.iris
        classifier, _ = get_iris_classifier_tf()

        # Test untargeted attack
        attack = ElasticNet(classifier, targeted=False, max_iter=10)
        x_test_adv = attack.generate(x_test)
        expected_x_test_adv = np.asarray([0.8479195, 0.42525578, 0.70166135, 0.28664514])
        np.testing.assert_array_almost_equal(x_test_adv[0, :], expected_x_test_adv, decimal=6)
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)

        predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        np.testing.assert_array_equal(predictions_adv, np.asarray(
            [1, 2, 2, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 1, 0, 2,
             2, 1, 2, 0, 2, 2, 1, 1, 2]))
        accuracy = 1.0 - np.sum(predictions_adv == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info('EAD success rate on Iris: %.2f%%', (accuracy * 100))

        # Test targeted attack
        targets = random_targets(y_test, nb_classes=3)
        attack = ElasticNet(classifier, targeted=True, max_iter=10)
        x_test_adv = attack.generate(x_test, **{'y': targets})
        expected_x_test_adv = np.asarray([0.8859426, 0.51877, 0.5014498, 0.05447771])
        np.testing.assert_array_almost_equal(x_test_adv[0, :], expected_x_test_adv, decimal=6)
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)

        predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        np.testing.assert_array_equal(predictions_adv, np.asarray(
            [0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 0, 0, 2, 2, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0, 2, 0, 2, 2, 2, 2, 2, 0,
             0, 0, 2, 2, 2, 2, 2, 0, 2]))

        accuracy = np.sum(predictions_adv == np.argmax(targets, axis=1)) / y_test.shape[0]
        logger.info('Targeted EAD success rate on Iris: %.2f%%', (accuracy * 100))

    def test_pytorch_iris(self):
        (_, _), (x_test, y_test) = self.iris
        classifier = get_iris_classifier_pt()
        attack = ElasticNet(classifier, targeted=False, max_iter=10)
        x_test_adv = attack.generate(x_test.astype(np.float32))
        expected_x_test_adv = np.asarray([0.8479194, 0.42525578, 0.70166135, 0.28664517])
        np.testing.assert_array_almost_equal(x_test_adv[0, :], expected_x_test_adv, decimal=6)
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)

        predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        np.testing.assert_array_equal(predictions_adv, np.asarray(
            [1, 2, 2, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 1, 0, 2,
             2, 1, 2, 0, 2, 2, 1, 1, 2]))

        accuracy = 1.0 - np.sum(predictions_adv == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info('EAD success rate on Iris: %.2f%%', (accuracy * 100))

    def test_scikitlearn(self):
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
            attack = ElasticNet(classifier, targeted=False, max_iter=10)
            x_test_adv = attack.generate(x_test)
            self.assertFalse((x_test == x_test_adv).all())
            self.assertLessEqual(np.amax(x_test_adv), 1.0)
            self.assertGreaterEqual(np.amin(x_test_adv), 0.0)

            predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
            self.assertFalse((np.argmax(y_test, axis=1) == predictions_adv).all())
            accuracy = 1.0 - np.sum(predictions_adv == np.argmax(y_test, axis=1)) / y_test.shape[0]
            logger.info('EAD success rate  of ' + classifier.__class__.__name__ + ' on Iris: %.2f%%', (accuracy * 100))

            # Test targeted attack
            targets = random_targets(y_test, nb_classes=3)
            attack = ElasticNet(classifier, targeted=True, max_iter=10)
            x_test_adv = attack.generate(x_test, **{'y': targets})
            self.assertFalse((x_test == x_test_adv).all())
            self.assertLessEqual(np.amax(x_test_adv), 1.0)
            self.assertGreaterEqual(np.amin(x_test_adv), 0.0)

            predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
            self.assertTrue((np.argmax(targets, axis=1) == predictions_adv).any())
            accuracy = np.sum(predictions_adv == np.argmax(targets, axis=1)) / y_test.shape[0]
            logger.info('Targeted EAD success rate of ' + classifier.__class__.__name__ + ' on Iris: %.2f%%',
                        (accuracy * 100))

            # Check that x_test has not been modified by attack and classifier
            self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test))), 0.0, delta=0.00001)


if __name__ == '__main__':
    unittest.main()
