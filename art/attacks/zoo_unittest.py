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

from art.attacks.zoo import ZooAttack
from art.utils import get_classifier_kr, get_classifier_pt, get_classifier_tf
from art.utils import load_mnist, random_targets, master_seed

logger = logging.getLogger('testLogger')

NB_TEST = 6


class TestZooAttack(unittest.TestCase):
    """
    A unittest class for testing the ZOO attack.
    """

    @classmethod
    def setUpClass(cls):
        # Get MNIST
        (_, _), (x_test, y_test), _, _ = load_mnist()
        x_test, y_test = x_test[:NB_TEST], y_test[:NB_TEST]
        cls.mnist = x_test, y_test

    def setUp(self):
        # Set master seed
        master_seed(1234)

    def test_failure_attack(self):
        """
        Test the corner case when attack fails.
        :return:
        """
        # Build TFClassifier
        tfc, sess = get_classifier_tf()

        # Get MNIST
        x_test, _ = self.mnist

        # Failure attack
        zoo = ZooAttack(classifier=tfc, max_iter=0, binary_search_steps=0, learning_rate=0)
        x_test_adv = zoo.generate(x_test)
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())
        np.testing.assert_almost_equal(x_test, x_test_adv, 3)

        # Clean-up session
        sess.close()
        tf.reset_default_graph()

    def test_tfclassifier(self):
        """
        First test with the TFClassifier.
        :return:
        """
        # Build TFClassifier
        tfc, sess = get_classifier_tf()

        # Get MNIST
        x_test, y_test = self.mnist

        # Targeted attack
        zoo = ZooAttack(classifier=tfc, targeted=True)
        params = {'y': random_targets(y_test, tfc.nb_classes)}
        x_test_adv = zoo.generate(x_test, **params)
        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())
        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        logger.debug('ZOO target: %s', target)
        logger.debug('ZOO actual: %s', y_pred_adv)
        logger.info('ZOO success rate: %.2f', (sum(target == y_pred_adv) / float(len(target))))

        # Untargeted attack
        zoo = ZooAttack(classifier=tfc, targeted=False)
        x_test_adv = zoo.generate(x_test)
        # self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())
        y_pred = np.argmax(tfc.predict(x_test), axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        logger.debug('ZOO actual: %s', y_pred_adv)
        logger.info('ZOO success rate: %.2f', (sum(y_pred != y_pred_adv) / float(len(y_pred))))

        # Clean-up session
        sess.close()
        tf.reset_default_graph()

    def test_krclassifier(self):
        """
        Second test with the KerasClassifier.
        :return:
        """
        # Build KerasClassifier
        krc, _ = get_classifier_kr()

        # Get MNIST and test with 3 channels
        x_test, y_test = self.mnist

        # First attack
        zoo = ZooAttack(classifier=krc, targeted=True, batch_size=5)
        params = {'y': random_targets(y_test, krc.nb_classes)}
        x_test_adv = zoo.generate(x_test, **params)
        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())
        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(krc.predict(x_test_adv), axis=1)
        logger.debug('ZOO target: %s', target)
        logger.debug('ZOO actual: %s', y_pred_adv)
        logger.info('ZOO success rate: %.2f', (sum(target == y_pred_adv) / float(len(target))))

        # Second attack
        zoo = ZooAttack(classifier=krc, targeted=False, max_iter=20)
        x_test_adv = zoo.generate(x_test)
        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())
        y_pred_adv = np.argmax(krc.predict(x_test_adv), axis=1)
        y_pred = np.argmax(krc.predict(x_test), axis=1)
        logger.debug('ZOO actual: %s', y_pred_adv)
        logger.info('ZOO success rate: %.2f', (sum(y_pred != y_pred_adv) / float(len(y_pred))))

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
        x_test, y_test = self.mnist
        x_test = np.swapaxes(x_test, 1, 3)

        # First attack
        zoo = ZooAttack(classifier=ptc, targeted=True, max_iter=10)
        params = {'y': random_targets(y_test, ptc.nb_classes)}
        x_test_adv = zoo.generate(x_test, **params)
        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())
        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(ptc.predict(x_test_adv), axis=1)
        logger.debug('ZOO target: %s', target)
        logger.debug('ZOO actual: %s', y_pred_adv)
        logger.info('ZOO success rate: %.2f', (sum(target != y_pred_adv) / float(len(target))))

        # Second attack
        zoo = ZooAttack(classifier=ptc, targeted=False, max_iter=10)
        x_test_adv = zoo.generate(x_test)
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())
        y_pred_adv = np.argmax(ptc.predict(x_test_adv), axis=1)
        y_pred = np.argmax(ptc.predict(x_test), axis=1)
        logger.debug('ZOO actual: %s', y_pred_adv)
        logger.info('ZOO success rate: %.2f', (sum(y_pred != y_pred_adv) / float(len(y_pred))))


if __name__ == '__main__':
    unittest.main()
