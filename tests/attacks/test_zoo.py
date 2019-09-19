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

from art.attacks import ZooAttack
from art.utils import load_dataset, random_targets, master_seed
from art.utils_test import get_classifier_kr, get_classifier_pt, get_classifier_tf, get_iris_classifier_pt

logger = logging.getLogger('testLogger')

NB_TRAIN = 1
NB_TEST = 1


class TestZooAttack(unittest.TestCase):
    """
    A unittest class for testing the ZOO attack.
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

    def test_failure_attack(self):
        """
        Test the corner case when attack fails.
        :return:
        """
        # Build TensorFlowClassifier
        tfc, sess = get_classifier_tf()

        # Failure attack
        zoo = ZooAttack(classifier=tfc, max_iter=0, binary_search_steps=0, learning_rate=0)
        x_test_adv = zoo.generate(self.x_test)
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
        np.testing.assert_almost_equal(self.x_test, x_test_adv, 3)

        # Clean-up session
        sess.close()

    def test_tfclassifier(self):
        """
        First test with the TensorFlowClassifier.
        :return:
        """
        # Build TensorFlowClassifier
        tfc, sess = get_classifier_tf()

        # Targeted attack
        zoo = ZooAttack(classifier=tfc, targeted=True, max_iter=100, binary_search_steps=10)
        params = {'y': random_targets(self.y_test, tfc.nb_classes())}
        x_test_adv = zoo.generate(self.x_test, **params)
        self.assertFalse((self.x_test == x_test_adv).all())
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        logger.debug('ZOO target: %s', target)
        logger.debug('ZOO actual: %s', y_pred_adv)
        logger.info('ZOO success rate on MNIST: %.2f', (sum(target == y_pred_adv) / float(len(target))))

        # Untargeted attack
        zoo = ZooAttack(classifier=tfc, targeted=False)
        x_test_adv = zoo.generate(self.x_test)
        # self.assertFalse((x_test == x_test_adv).all())
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
        y_pred = np.argmax(tfc.predict(self.x_test), axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        logger.debug('ZOO actual: %s', y_pred_adv)
        logger.info('ZOO success rate on MNIST: %.2f', (sum(y_pred != y_pred_adv) / float(len(y_pred))))

        # Clean-up session
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

        # Targeted attack
        # zoo = ZooAttack(classifier=krc, targeted=True, batch_size=5)
        # params = {'y': random_targets(self.y_test, krc.nb_classes())}
        # x_test_adv = zoo.generate(self.x_test, **params)
        #
        # self.assertFalse((self.x_test == x_test_adv).all())
        # self.assertLessEqual(np.amax(x_test_adv), 1.0)
        # self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
        # target = np.argmax(params['y'], axis=1)
        # y_pred_adv = np.argmax(krc.predict(x_test_adv), axis=1)
        # logger.debug('ZOO target: %s', target)
        # logger.debug('ZOO actual: %s', y_pred_adv)
        # logger.info('ZOO success rate on MNIST: %.2f', (sum(target == y_pred_adv) / float(len(target))))

        # Untargeted attack
        # zoo = ZooAttack(classifier=krc, targeted=False, max_iter=20)
        zoo = ZooAttack(classifier=krc, targeted=False, batch_size=5)
        # x_test_adv = zoo.generate(x_test)
        params = {'y': random_targets(self.y_test, krc.nb_classes())}
        x_test_adv = zoo.generate(self.x_test, **params)

        # x_test_adv_true = [0.00000000e+00, 2.50167388e-04, 1.50529508e-04, 4.69674182e-04,
        #                    0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        #                    1.67321396e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        #                    0.00000000e+00, 2.08451956e-06, 0.00000000e+00, 0.00000000e+00,
        #                    2.53360748e-01, 9.60119188e-01, 9.85227525e-01, 2.53600776e-01,
        #                    0.00000000e+00, 0.00000000e+00, 5.23251540e-04, 0.00000000e+00,
        #                    0.00000000e+00, 0.00000000e+00, 1.08632184e-05, 0.00000000e+00]
        #
        # for i in range(14):
        #     self.assertAlmostEqual(x_test_adv_true[i], x_test_adv[0, 14, i, 0])

        # self.assertFalse((x_test == x_test_adv).all())
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
        y_pred_adv = np.argmax(krc.predict(x_test_adv), axis=1)
        y_pred = np.argmax(krc.predict(self.x_test), axis=1)
        logger.debug('ZOO actual: %s', y_pred_adv)
        logger.info('ZOO success rate on MNIST: %.2f', (sum(y_pred != y_pred_adv) / float(len(y_pred))))

        # Clean-up
        k.clear_session()

    def test_ptclassifier(self):
        """
        Third test with the PyTorchClassifier.
        :return:
        """
        # Build PyTorchClassifier
        ptc = get_classifier_pt()

        # Get MNIST
        x_test = np.swapaxes(self.x_test, 1, 3).astype(np.float32)

        # First attack
        # zoo = ZooAttack(classifier=ptc, targeted=True, max_iter=10, binary_search_steps=10)
        # params = {'y': random_targets(self.y_test, ptc.nb_classes())}
        # x_test_adv = zoo.generate(x_test, **params)
        # self.assertFalse((x_test == x_test_adv).all())
        # self.assertLessEqual(np.amax(x_test_adv), 1.0)
        # self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
        # target = np.argmax(params['y'], axis=1)
        # y_pred_adv = np.argmax(ptc.predict(x_test_adv), axis=1)
        # logger.debug('ZOO target: %s', target)
        # logger.debug('ZOO actual: %s', y_pred_adv)
        # logger.info('ZOO success rate on MNIST: %.2f', (sum(target != y_pred_adv) / float(len(target))))

        # Second attack
        zoo = ZooAttack(classifier=ptc, targeted=False, learning_rate=1e-2, max_iter=15, binary_search_steps=10,
                        abort_early=False, use_resize=False, use_importance=False)
        x_test_adv = zoo.generate(x_test)
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)

        # print(x_test[0, 0, 14, :])
        # print(x_test_adv[0, 0, 14, :])
        # print(np.amax(x_test - x_test_adv))
        x_test_adv_expected = []

    def test_classifier_type_check_fail_classifier(self):
        # Use a useless test classifier to test basic classifier properties
        class ClassifierNoAPI:
            pass

        classifier = ClassifierNoAPI
        with self.assertRaises(TypeError) as context:
            _ = ZooAttack(classifier=classifier)

        self.assertIn('For `ZooAttack` classifier must be an instance of `art.classifiers.classifier.Classifier`, the '
                      'provided classifier is instance of (<class \'object\'>,).', str(context.exception))

    if __name__ == '__main__':
        unittest.main()
