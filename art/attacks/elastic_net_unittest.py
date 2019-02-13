from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

import keras.backend as k
import numpy as np
import tensorflow as tf

from art.attacks import ElasticNet
from art.utils import load_mnist, random_targets, master_seed, get_classifier_tf, get_classifier_kr, get_classifier_pt

logger = logging.getLogger('testLogger')

BATCH_SIZE = 100
NB_TRAIN = 500
NB_TEST = 10


class TestElasticNet(unittest.TestCase):
    """
    A unittest class for testing the ElasticNet attack.
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

    def test_failure_attack(self):
        """
        Test the corner case when attack fails.
        :return:
        """
        # Build TFClassifier
        tfc, sess = get_classifier_tf()

        # Get MNIST
        (_, _), (x_test, y_test) = self.mnist

        # Failure attack
        ead = ElasticNet(classifier=tfc, targeted=True, max_iter=0, binary_search_steps=0, learning_rate=0,
                         initial_const=1)
        params = {'y': random_targets(y_test, tfc.nb_classes)}
        x_test_adv = ead.generate(x_test, **params)
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())
        np.testing.assert_almost_equal(x_test, x_test_adv, 3)

        # Kill TF
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
        (_, _), (x_test, y_test) = self.mnist

        # First attack
        ead = ElasticNet(classifier=tfc, targeted=True, max_iter=2)
        params = {'y': random_targets(y_test, tfc.nb_classes)}
        x_test_adv = ead.generate(x_test, **params)
        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())
        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        logger.debug('EAD Target: %s', target)
        logger.debug('EAD Actual: %s', y_pred_adv)
        logger.info('EAD Success Rate: %.2f', (sum(target == y_pred_adv) / float(len(target))))
        self.assertTrue((target == y_pred_adv).any())

        # Second attack
        ead = ElasticNet(classifier=tfc, targeted=False, max_iter=2)
        params = {'y': random_targets(y_test, tfc.nb_classes)}
        x_test_adv = ead.generate(x_test, **params)
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())
        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        logger.debug('EAD Target: %s', target)
        logger.debug('EAD Actual: %s', y_pred_adv)
        logger.info('EAD Success Rate: %.2f', (sum(target != y_pred_adv) / float(len(target))))
        self.assertTrue((target != y_pred_adv).any())

        # Third attack
        ead = ElasticNet(classifier=tfc, targeted=False, max_iter=2)
        params = {}
        x_test_adv = ead.generate(x_test, **params)
        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())
        y_pred = np.argmax(tfc.predict(x_test), axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        logger.debug('EAD Target: %s', y_pred)
        logger.debug('EAD Actual: %s', y_pred_adv)
        logger.info('EAD Success Rate: %.2f', (sum(y_pred != y_pred_adv) / float(len(y_pred))))
        self.assertTrue((y_pred != y_pred_adv).any())

        # First attack without batching
        ead_wob = ElasticNet(classifier=tfc, targeted=True, max_iter=2, batch_size=1)
        params = {'y': random_targets(y_test, tfc.nb_classes)}
        x_test_adv = ead_wob.generate(x_test, **params)
        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())
        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        logger.debug('EAD Target: %s', target)
        logger.debug('EAD Actual: %s', y_pred_adv)
        logger.info('EAD Success Rate: %.2f', (sum(target == y_pred_adv) / float(len(target))))
        self.assertTrue((target == y_pred_adv).any())

        # Second attack without batching
        ead_wob = ElasticNet(classifier=tfc, targeted=False, max_iter=2, batch_size=1)
        params = {'y': random_targets(y_test, tfc.nb_classes)}
        x_test_adv = ead_wob.generate(x_test, **params)
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())
        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        logger.debug('EAD Target: %s', target)
        logger.debug('EAD Actual: %s', y_pred_adv)
        logger.info('EAD Success Rate: %.2f', (sum(target != y_pred_adv) / float(len(target))))
        self.assertTrue((target != y_pred_adv).any())

        # Kill TF
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
        (_, _), (x_test, y_test) = self.mnist

        # First attack
        ead = ElasticNet(classifier=krc, targeted=True, max_iter=2)
        params = {'y': random_targets(y_test, krc.nb_classes)}
        x_test_adv = ead.generate(x_test, **params)
        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())
        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(krc.predict(x_test_adv), axis=1)
        logger.debug('EAD Target: %s', target)
        logger.debug('EAD Actual: %s', y_pred_adv)
        logger.info('EAD Success Rate: %.2f', (sum(target == y_pred_adv) / float(len(target))))
        self.assertTrue((target == y_pred_adv).any())

        # Second attack
        ead = ElasticNet(classifier=krc, targeted=False, max_iter=2)
        params = {'y': random_targets(y_test, krc.nb_classes)}
        x_test_adv = ead.generate(x_test, **params)
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())
        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(krc.predict(x_test_adv), axis=1)
        logger.debug('EAD Target: %s', target)
        logger.debug('EAD Actual: %s', y_pred_adv)
        logger.info('EAD Success Rate: %.2f', (sum(target != y_pred_adv) / float(len(target))))
        self.assertTrue((target != y_pred_adv).any())

        # Kill Keras
        k.clear_session()

    def test_ptclassifier(self):
        """
        Third test with the PyTorchClassifier.
        :return:
        """
        # Build PyTorchClassifier
        ptc = get_classifier_pt()

        # Get MNIST
        (_, _), (x_test, y_test) = self.mnist
        x_test = np.swapaxes(x_test, 1, 3)

        # First attack
        ead = ElasticNet(classifier=ptc, targeted=True, max_iter=2)
        params = {'y': random_targets(y_test, ptc.nb_classes)}
        x_test_adv = ead.generate(x_test, **params)
        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())
        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(ptc.predict(x_test_adv), axis=1)
        self.assertTrue((target == y_pred_adv).any())

        # Second attack
        ead = ElasticNet(classifier=ptc, targeted=False, max_iter=2)
        params = {'y': random_targets(y_test, ptc.nb_classes)}
        x_test_adv = ead.generate(x_test, **params)
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())
        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(ptc.predict(x_test_adv), axis=1)
        self.assertTrue((target != y_pred_adv).any())


if __name__ == '__main__':
    unittest.main()
