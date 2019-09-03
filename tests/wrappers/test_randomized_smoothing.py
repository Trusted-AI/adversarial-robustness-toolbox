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
import tensorflow as tf

from art.attacks import FastGradientMethod
from art.classifiers import KerasClassifier
from art.utils import load_dataset, random_targets, master_seed, compute_accuracy
from art.utils_test import get_classifier_kr, get_iris_classifier_kr
from art.wrappers.randomized_smoothing import RandomizedSmoothing
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
logger = logging.getLogger('testLogger')

BATCH_SIZE = 100
NB_TRAIN = 5000
NB_TEST = 10


class TestRandomizedSmoothing(unittest.TestCase):
    """
    A unittest class for testing Randomized Smoothing as a post-processing step for classifiers.
    """

    @classmethod
    def setUpClass(cls):
        # Get MNIST
        (x_train, y_train), (x_test, y_test), _, _ = load_dataset('mnist')
        x_train, y_train = x_train[:NB_TRAIN], y_train[:NB_TRAIN]
        x_test, y_test = x_test[:NB_TEST], y_test[:NB_TEST]
        cls.mnist = (x_train, y_train), (x_test, y_test)

    def setUp(self):
        # Set master seed
        master_seed(1234)

    @unittest.skipIf(tf.__version__[0] == '2', reason='Skip unittests for Tensorflow v2 until Keras supports Tensorflow'
                                                      ' v2 as backend.')
    def test_krclassifier(self):
        """
        Test with a KerasClassifier.
        :return:
        """
        # Build KerasClassifier
        krc = get_classifier_kr()

        # Get MNIST
        (_, _), (x_test, y_test) = self.mnist

        # First FGSM attack:
        fgsm = FastGradientMethod(classifier=krc, targeted=True)
        params = {'y': random_targets(y_test, krc.nb_classes())}
        x_test_adv = fgsm.generate(x_test, **params)

        # Initialize RS object and attack with FGSM
        rs = RandomizedSmoothing(classifier=krc, sample_size=100, scale=0.01, alpha=0.001)
        fgsm_with_rs = FastGradientMethod(classifier=rs, targeted=True)
        x_test_adv_with_rs = fgsm_with_rs.generate(x_test, **params)

        # Compare results
        # check shapes are equal and values are within a certain range
        self.assertEqual(x_test_adv.shape, x_test_adv_with_rs.shape)
        self.assertTrue((np.abs(x_test_adv - x_test_adv_with_rs) < 0.75).all())

        # Check basic functionality of RS object
        # check predict
        y_test_smooth = rs.predict(x=x_test)
        y_test_base = krc.predict(x=x_test)
        self.assertEqual(y_test_smooth.shape, y_test.shape)
        self.assertTrue((np.sum(y_test_smooth, axis=1) <= np.ones((NB_TEST,))).all())
        self.assertTrue((np.argmax(y_test_smooth, axis=1) == np.argmax(y_test_base, axis=1)).all())

        # check gradients
        grad_smooth1 = rs.loss_gradient(x=x_test, y=y_test)
        grad_smooth2 = rs.class_gradient(x=x_test, label=None)
        grad_smooth3 = rs.class_gradient(x=x_test, label=np.argmax(y_test, axis=1))
        self.assertEqual(grad_smooth1.shape, x_test_adv.shape)
        self.assertEqual(grad_smooth2.shape[0], NB_TEST)
        self.assertEqual(grad_smooth3.shape[0], NB_TEST)

        # check certification
        pred, radius = rs.certify(x=x_test, n=250)
        self.assertEqual(len(pred), NB_TEST)
        self.assertEqual(len(radius), NB_TEST)
        self.assertTrue((radius <= 1).all())
        self.assertTrue((pred < y_test.shape[1]).all())


class TestRandomizedSmoothingVectors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Get Iris
        (x_train, y_train), (x_test, y_test), _, _ = load_dataset('iris')
        cls.iris = (x_train, y_train), (x_test, y_test)

    def setUp(self):
        # Set master seed
        master_seed(1234)

    @unittest.skipIf(tf.__version__[0] == '2', reason='Skip unittests for Tensorflow v2 until Keras supports Tensorflow'
                                                      ' v2 as backend.')
    def test_iris_clipped(self):
        (_, _), (x_test, y_test) = self.iris

        krc, _ = get_iris_classifier_kr()
        rs = RandomizedSmoothing(classifier=krc, sample_size=100, scale=0.01, alpha=0.001)

        # Test untargeted attack
        attack = FastGradientMethod(krc, eps=.1)
        x_test_adv = attack.generate(x_test)
        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1).all())
        self.assertTrue((x_test_adv >= 0).all())

        preds_base = np.argmax(rs.predict(x_test), axis=1)
        preds_smooth = np.argmax(rs.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(y_test, axis=1) == preds_smooth).all())

        pred = rs.predict(x_test)
        pred2 = rs.predict(x_test_adv)
        acc, cov = compute_accuracy(pred, y_test)
        acc2, cov2 = compute_accuracy(pred2, y_test)
        logger.info('Accuracy on Iris with smoothing on adversarial examples: %.2f%%', (acc * 100))
        logger.info('Coverage on Iris with smoothing on adversarial examples: %.2f%%', (cov * 100))
        logger.info('Accuracy on Iris with smoothing: %.2f%%', (acc2 * 100))
        logger.info('Coverage on Iris with smoothing: %.2f%%', (cov2 * 100))


        # Check basic functionality of RS object
        # check predict
        y_test_smooth = rs.predict(x=x_test)
        self.assertEqual(y_test_smooth.shape, y_test.shape)
        self.assertTrue((np.sum(y_test_smooth, axis=1) <= 1).all())
        
        # check gradients
        grad_smooth1 = rs.loss_gradient(x=x_test, y=y_test)
        grad_smooth2 = rs.class_gradient(x=x_test, label=None)
        grad_smooth3 = rs.class_gradient(x=x_test, label=np.argmax(y_test, axis=1))
        self.assertEqual(grad_smooth1.shape, x_test_adv.shape)
        self.assertEqual(grad_smooth2.shape[0], len(x_test))
        self.assertEqual(grad_smooth3.shape[0], len(x_test))
        
        # check certification
        pred, radius = rs.certify(x=x_test, n=250)
        self.assertEqual(len(pred), len(x_test))
        self.assertEqual(len(radius), len(x_test))
        self.assertTrue((radius <= 1).all())
        self.assertTrue((pred < y_test.shape[1]).all())

    @unittest.skipIf(tf.__version__[0] == '2', reason='Skip unittests for Tensorflow v2 until Keras supports Tensorflow'
                                                      ' v2 as backend.')
    def test_iris_unbounded(self):
        (_, _), (x_test, y_test) = self.iris
        classifier, _ = get_iris_classifier_kr()

        # Recreate a classifier without clip values
        krc = KerasClassifier(model=classifier._model, use_logits=False, channel_index=1)
        rs = RandomizedSmoothing(classifier=krc, sample_size=100, scale=0.01, alpha=0.001)
        attack = FastGradientMethod(rs, eps=1)
        x_test_adv = attack.generate(x_test)
        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv > 1).any())
        self.assertTrue((x_test_adv < 0).any())

        preds_base = np.argmax(rs.predict(x_test), axis=1)
        preds_smooth = np.argmax(rs.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(y_test, axis=1) == preds_smooth).all())

        pred = rs.predict(x_test)
        pred2 = rs.predict(x_test_adv)
        acc, cov = compute_accuracy(pred, y_test)
        acc2, cov2 = compute_accuracy(pred2, y_test)
        logger.info('Accuracy on Iris with smoothing on adversarial examples: %.2f%%', (acc * 100))
        logger.info('Coverage on Iris with smoothing on adversarial examples: %.2f%%', (cov * 100))
        logger.info('Accuracy on Iris with smoothing: %.2f%%', (acc2 * 100))
        logger.info('Coverage on Iris with smoothing: %.2f%%', (cov2 * 100))


if __name__ == '__main__':
    unittest.main()
