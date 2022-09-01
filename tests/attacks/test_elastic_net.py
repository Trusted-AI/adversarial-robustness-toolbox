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

from art.attacks.evasion.elastic_net import ElasticNet
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassGradientsMixin
from art.estimators.classification.keras import KerasClassifier
from art.utils import random_targets, to_categorical
from tests.attacks.utils import backend_test_classifier_type_check_fail
from tests.utils import (
    TestBase,
    get_image_classifier_kr,
    get_image_classifier_pt,
    get_image_classifier_tf,
    get_tabular_classifier_kr,
    get_tabular_classifier_pt,
    get_tabular_classifier_tf,
    master_seed,
)

logger = logging.getLogger(__name__)


class TestElasticNet(TestBase):
    """
    A unittest class for testing the ElasticNet attack.
    """

    @classmethod
    def setUpClass(cls):
        master_seed(seed=1234)
        super().setUpClass()

        cls.n_train = 500
        cls.n_test = 10
        cls.x_train_mnist = cls.x_train_mnist[0 : cls.n_train]
        cls.y_train_mnist = cls.y_train_mnist[0 : cls.n_train]
        cls.x_test_mnist = cls.x_test_mnist[0 : cls.n_test]
        cls.y_test_mnist = cls.y_test_mnist[0 : cls.n_test]

    def setUp(self):
        master_seed(seed=1234)
        super().setUp()

    def test_2_tensorflow_failure_attack(self):
        """
        Test the corner case when attack fails.
        :return:
        """
        # Build TensorFlowClassifier
        tfc, sess = get_image_classifier_tf()

        # Failure attack
        ead = ElasticNet(
            classifier=tfc,
            targeted=True,
            max_iter=0,
            binary_search_steps=0,
            learning_rate=0,
            initial_const=1,
            verbose=False,
        )
        params = {"y": random_targets(self.y_test_mnist, tfc.nb_classes)}
        x_test_adv = ead.generate(self.x_test_mnist, **params)
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
        np.testing.assert_almost_equal(self.x_test_mnist, x_test_adv, 3)

        # Clean-up session
        if sess is not None:
            sess.close()

    def test_4_tensorflow_mnist(self):
        """
        First test with the TensorFlowClassifier.
        :return:
        """
        x_test_original = self.x_test_mnist.copy()

        # Build TensorFlowClassifier
        tfc, sess = get_image_classifier_tf(from_logits=True)

        # First attack
        ead = ElasticNet(classifier=tfc, targeted=True, max_iter=2, verbose=False)
        params = {"y": random_targets(self.y_test_mnist, tfc.nb_classes)}
        x_test_adv = ead.generate(self.x_test_mnist, **params)
        expected_x_test_adv = np.asarray(
            [
                0.45284095,
                0.43225235,
                0.577448,
                1.0,
                0.16554962,
                0.3587564,
                0.19202651,
                0.04293771,
                0.0,
                0.25811037,
                0.0,
                0.0290696,
                0.0,
                0.136172,
                0.6153389,
                0.12244645,
                0.0,
                0.7586619,
                0.8366919,
                0.22206311,
                0.12455986,
                0.02862802,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        np.testing.assert_array_almost_equal(x_test_adv[0, 14, :, 0], expected_x_test_adv, decimal=6)
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
        target = np.argmax(params["y"], axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        logger.debug("EAD target: %s", target)
        logger.debug("EAD actual: %s", y_pred_adv)
        logger.info("EAD success rate on MNIST: %.2f%%", (100 * sum(target == y_pred_adv) / len(target)))
        self.assertTrue((target == y_pred_adv).any())

        # Second attack
        ead = ElasticNet(classifier=tfc, targeted=False, max_iter=2, verbose=False)
        params = {"y": random_targets(self.y_test_mnist, tfc.nb_classes)}
        x_test_adv = ead.generate(self.x_test_mnist, **params)
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
        target = np.argmax(params["y"], axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        logger.debug("EAD target: %s", target)
        logger.debug("EAD actual: %s", y_pred_adv)
        logger.info("EAD success rate on MNIST: %.2f%%", (100 * sum(target != y_pred_adv) / float(len(target))))
        np.testing.assert_array_equal(y_pred_adv, np.asarray([7, 1, 1, 4, 4, 1, 4, 4, 4, 4]))

        # Third attack
        ead = ElasticNet(classifier=tfc, targeted=False, max_iter=2, verbose=False)
        params = {}
        x_test_adv = ead.generate(self.x_test_mnist, **params)
        expected_x_test_adv = np.asarray(
            [
                0.22766514,
                0.21726893,
                0.22802338,
                0.06168516,
                0.0,
                0.0,
                0.04722975,
                0.0,
                0.0,
                0.0,
                0.05455382,
                0.0,
                0.0,
                0.0,
                0.38886347,
                0.10553087,
                0.32285708,
                0.9794307,
                0.7589039,
                0.16586718,
                0.15969527,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        np.testing.assert_array_almost_equal(x_test_adv[0, 14, :, 0], expected_x_test_adv, decimal=6)
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
        y_pred = np.argmax(tfc.predict(self.x_test_mnist), axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        logger.debug("EAD target: %s", y_pred)
        logger.debug("EAD actual: %s", y_pred_adv)
        logger.info("EAD success rate: %.2f%%", (100 * sum(y_pred != y_pred_adv) / float(len(y_pred))))
        np.testing.assert_array_equal(y_pred_adv, np.asarray([0, 4, 7, 9, 0, 7, 7, 3, 0, 7]))

        # First attack without batching
        ead_wob = ElasticNet(classifier=tfc, targeted=True, max_iter=2, batch_size=1, verbose=False)
        params = {"y": random_targets(self.y_test_mnist, tfc.nb_classes)}
        x_test_adv = ead_wob.generate(self.x_test_mnist, **params)
        expected_x_test_adv = np.asarray(
            [
                0.32568744,
                0.31085464,
                0.43235543,
                1.0,
                0.2433605,
                0.3411813,
                0.49469775,
                0.2603619,
                0.0,
                0.35584357,
                0.0,
                0.2046029,
                0.0,
                0.08249304,
                1.0,
                0.35813788,
                0.62945133,
                1.0,
                0.32154015,
                0.3497113,
                0.7613426,
                0.36533928,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        np.testing.assert_array_almost_equal(x_test_adv[0, 14, :, 0], expected_x_test_adv, decimal=6)
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
        target = np.argmax(params["y"], axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        logger.debug("EAD target: %s", target)
        logger.debug("EAD actual: %s", y_pred_adv)
        logger.info("EAD success rate: %.2f%%", (100 * sum(target == y_pred_adv) / float(len(target))))
        self.assertTrue((target == y_pred_adv).any())

        # Second attack without batching
        ead_wob = ElasticNet(classifier=tfc, targeted=False, max_iter=2, batch_size=1, verbose=False)
        params = {"y": random_targets(self.y_test_mnist, tfc.nb_classes)}
        x_test_adv = ead_wob.generate(self.x_test_mnist, **params)
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
        target = np.argmax(params["y"], axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        logger.debug("EAD target: %s", target)
        logger.debug("EAD actual: %s", y_pred_adv)
        logger.info("EAD success rate: %.2f%%", (100 * sum(target != y_pred_adv) / float(len(target))))
        np.testing.assert_array_equal(y_pred_adv, np.asarray([7, 1, 1, 4, 4, 1, 4, 4, 4, 4]))

        # Check that x_test has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - self.x_test_mnist))), 0.0, delta=0.00001)

        ead = ElasticNet(classifier=tfc, targeted=False, max_iter=2, verbose=False, decision_rule="L1")
        _ = ead.generate(self.x_test_mnist, **params)
        ead = ElasticNet(classifier=tfc, targeted=False, max_iter=2, verbose=False, decision_rule="L2")
        _ = ead.generate(self.x_test_mnist, **params)
        with self.assertRaises(ValueError):
            _ = ElasticNet(classifier=tfc, targeted=False, max_iter=2, verbose=False, decision_rule="L3")

        # Close session
        if sess is not None:
            sess.close()

    def test_9a_keras_mnist(self):
        """
        Second test with the KerasClassifier.
        :return:
        """
        x_test_original = self.x_test_mnist.copy()

        # Build KerasClassifier
        krc = get_image_classifier_kr()

        # First attack
        ead = ElasticNet(classifier=krc, targeted=True, max_iter=2, verbose=False)
        y_target = to_categorical(np.asarray([6, 6, 7, 4, 9, 7, 9, 0, 1, 0]), nb_classes=10)
        x_test_adv = ead.generate(self.x_test_mnist, y=y_target)
        expected_x_test_adv = np.asarray(
            [
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                7.8193319e-04,
                5.5843666e-03,
                0.0000000e00,
                0.0000000e00,
                4.9869284e-01,
                1.0000000e00,
                6.4663666e-01,
                3.4855194e-03,
                3.5087438e-03,
                0.0000000e00,
                9.8862723e-03,
                3.8835173e-03,
                3.0151173e-02,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
            ]
        )
        np.testing.assert_array_almost_equal(x_test_adv[2, 14, :, 0], expected_x_test_adv, decimal=6)
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
        target = np.argmax(y_target, axis=1)
        y_pred_adv = np.argmax(krc.predict(x_test_adv), axis=1)
        logger.debug("EAD target: %s", target)
        logger.debug("EAD actual: %s", y_pred_adv)
        logger.info("EAD success rate: %.2f%%", (100 * sum(target == y_pred_adv) / float(len(target))))
        self.assertTrue((target == y_pred_adv).any())

        # Second attack
        ead = ElasticNet(classifier=krc, targeted=False, max_iter=2, verbose=False)
        y_target = to_categorical(np.asarray([9, 5, 6, 7, 1, 6, 1, 5, 8, 5]), nb_classes=10)
        x_test_adv = ead.generate(self.x_test_mnist, y=y_target)
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
        y_pred_adv = np.argmax(krc.predict(x_test_adv), axis=1)
        logger.debug("EAD target: %s", y_target)
        logger.debug("EAD actual: %s", y_pred_adv)
        logger.info("EAD success rate: %.2f", (100 * sum(target != y_pred_adv) / float(len(target))))
        self.assertTrue((target != y_pred_adv).any())
        np.testing.assert_array_equal(y_pred_adv, np.asarray([7, 1, 1, 4, 4, 1, 4, 4, 4, 4]))

        # Check that x_test has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - self.x_test_mnist))), 0.0, delta=0.00001)

        k.clear_session()

    def test_6_pytorch_mnist(self):
        """
        Third test with the PyTorchClassifier.
        :return:
        """
        x_test = np.reshape(self.x_test_mnist, (self.x_test_mnist.shape[0], 1, 28, 28)).astype(np.float32)
        x_test_original = x_test.copy()

        # Build PyTorchClassifier
        ptc = get_image_classifier_pt(from_logits=False)

        # First attack
        ead = ElasticNet(classifier=ptc, targeted=True, max_iter=2, verbose=False)
        params = {"y": random_targets(self.y_test_mnist, ptc.nb_classes)}
        x_test_adv = ead.generate(x_test, **params)
        expected_x_test_adv = np.asarray(
            [
                0.01758931,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.00698278,
                0.0,
                0.11318438,
                0.36223832,
                0.54720753,
                0.93125045,
                1.0,
                0.9999359,
                0.8638486,
                0.6354147,
                0.5600332,
                0.24081531,
                0.25882354,
                0.00899846,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        np.testing.assert_array_almost_equal(x_test_adv[2, 0, :, 14], expected_x_test_adv, decimal=6)
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
        target = np.argmax(params["y"], axis=1)
        y_pred_adv = np.argmax(ptc.predict(x_test_adv), axis=1)
        self.assertTrue((target == y_pred_adv).any())

        # Second attack
        ead = ElasticNet(classifier=ptc, targeted=False, max_iter=2, verbose=False)
        params = {"y": random_targets(self.y_test_mnist, ptc.nb_classes)}
        x_test_adv = ead.generate(x_test, **params)
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
        target = np.argmax(params["y"], axis=1)
        y_pred_adv = np.argmax(ptc.predict(x_test_adv), axis=1)
        self.assertTrue((target != y_pred_adv).any())
        np.testing.assert_array_equal(y_pred_adv, np.asarray([7, 1, 1, 4, 4, 1, 4, 4, 4, 4]))

        # Check that x_test has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test))), 0.0, delta=0.00001)

    def test_1_classifier_type_check_fail(self):
        backend_test_classifier_type_check_fail(ElasticNet, [BaseEstimator, ClassGradientsMixin])

    def test_8_keras_iris_clipped(self):
        classifier = get_tabular_classifier_kr()
        attack = ElasticNet(classifier, targeted=False, max_iter=10, verbose=False)
        x_test_adv = attack.generate(self.x_test_iris)
        expected_x_test_adv = np.asarray([0.8670352, 0.4624909, 0.6453267, 0.23096858])
        np.testing.assert_array_almost_equal(x_test_adv[0, :], expected_x_test_adv, decimal=6)
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)
        predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        np.testing.assert_array_equal(
            predictions_adv,
            np.asarray(
                [
                    1,
                    1,
                    1,
                    2,
                    1,
                    1,
                    1,
                    2,
                    1,
                    2,
                    1,
                    1,
                    1,
                    2,
                    1,
                    1,
                    2,
                    2,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    2,
                    1,
                    2,
                    1,
                    2,
                    1,
                    0,
                    1,
                    1,
                    1,
                    2,
                    2,
                    2,
                    2,
                    1,
                    1,
                    2,
                ]
            ),
        )
        accuracy = 1.0 - np.sum(predictions_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
        logger.info("EAD success rate on Iris: %.2f%%", (accuracy * 100))

        k.clear_session()

    def test_9_keras_iris_unbounded(self):
        classifier = get_tabular_classifier_kr()

        # Recreate a classifier without clip values
        classifier = KerasClassifier(model=classifier._model, use_logits=False, channels_first=True)
        attack = ElasticNet(classifier, targeted=False, max_iter=10, verbose=False)
        x_test_adv = attack.generate(self.x_test_iris)
        expected_x_test_adv = np.asarray([0.8670352, 0.4624909, 0.6453267, 0.23096858])
        np.testing.assert_array_almost_equal(x_test_adv[0, :], expected_x_test_adv, decimal=6)
        predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        np.testing.assert_array_equal(
            predictions_adv,
            np.asarray(
                [
                    1,
                    1,
                    1,
                    2,
                    1,
                    1,
                    1,
                    2,
                    1,
                    2,
                    1,
                    1,
                    1,
                    2,
                    1,
                    1,
                    2,
                    2,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    2,
                    1,
                    2,
                    1,
                    2,
                    1,
                    0,
                    1,
                    1,
                    1,
                    2,
                    2,
                    2,
                    2,
                    1,
                    1,
                    2,
                ]
            ),
        )
        accuracy = 1.0 - np.sum(predictions_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
        logger.info("EAD success rate on Iris: %.2f%%", (accuracy * 100))

        k.clear_session()

    def test_3_tensorflow_iris(self):
        classifier, sess = get_tabular_classifier_tf()

        # Test untargeted attack
        attack = ElasticNet(classifier, targeted=False, max_iter=10, verbose=False)
        x_test_adv = attack.generate(self.x_test_iris)
        expected_x_test_adv = np.asarray([0.84810126, 0.43320203, 0.70404345, 0.29160658])
        np.testing.assert_array_almost_equal(x_test_adv[0, :], expected_x_test_adv, decimal=6)
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)

        predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        np.testing.assert_array_equal(
            predictions_adv,
            np.asarray(
                [
                    1,
                    2,
                    2,
                    2,
                    1,
                    1,
                    1,
                    2,
                    1,
                    2,
                    1,
                    1,
                    1,
                    2,
                    2,
                    2,
                    2,
                    2,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    2,
                    2,
                    2,
                    2,
                    1,
                    2,
                    1,
                    2,
                    1,
                    0,
                    1,
                    2,
                    1,
                    2,
                    0,
                    2,
                    2,
                    1,
                    1,
                    2,
                ]
            ),
        )
        accuracy = 1.0 - np.sum(predictions_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
        logger.info("EAD success rate on Iris: %.2f%%", (accuracy * 100))

        # Test targeted attack
        targets = random_targets(self.y_test_iris, nb_classes=3)
        attack = ElasticNet(classifier, targeted=True, max_iter=10, verbose=False)
        x_test_adv = attack.generate(self.x_test_iris, **{"y": targets})
        expected_x_test_adv = np.asarray([0.88713187, 0.5239736, 0.49900988, 0.05677444])
        np.testing.assert_array_almost_equal(x_test_adv[0, :], expected_x_test_adv, decimal=6)
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)

        predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        np.testing.assert_array_equal(
            predictions_adv,
            np.asarray(
                [
                    0,
                    0,
                    0,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    0,
                    2,
                    0,
                    0,
                    2,
                    2,
                    0,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    0,
                    0,
                    0,
                    2,
                    0,
                    2,
                    2,
                    2,
                    2,
                    2,
                    0,
                    0,
                    0,
                    2,
                    2,
                    2,
                    2,
                    2,
                    0,
                    2,
                ]
            ),
        )

        accuracy = np.sum(predictions_adv == np.argmax(targets, axis=1)) / self.y_test_iris.shape[0]
        logger.info("Targeted EAD success rate on Iris: %.2f%%", (accuracy * 100))

        # Close session
        if sess is not None:
            sess.close()

    def test_5_pytorch_iris(self):
        classifier = get_tabular_classifier_pt()
        attack = ElasticNet(classifier, targeted=False, max_iter=10, verbose=False)
        x_test_adv = attack.generate(self.x_test_iris.astype(np.float32))
        expected_x_test_adv = np.asarray([0.84810126, 0.43320203, 0.70404345, 0.29160658])
        np.testing.assert_array_almost_equal(x_test_adv[0, :], expected_x_test_adv, decimal=6)
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)

        predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        np.testing.assert_array_equal(
            predictions_adv,
            np.asarray(
                [
                    1,
                    2,
                    2,
                    2,
                    1,
                    1,
                    1,
                    2,
                    1,
                    2,
                    1,
                    1,
                    1,
                    2,
                    2,
                    2,
                    2,
                    2,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    2,
                    2,
                    2,
                    2,
                    1,
                    2,
                    1,
                    2,
                    1,
                    0,
                    1,
                    2,
                    1,
                    2,
                    0,
                    2,
                    2,
                    1,
                    1,
                    2,
                ]
            ),
        )

        accuracy = 1.0 - np.sum(predictions_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
        logger.info("EAD success rate on Iris: %.2f%%", (accuracy * 100))

    def test_7_scikitlearn(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC, LinearSVC

        from art.estimators.classification.scikitlearn import SklearnClassifier

        scikitlearn_test_cases = [
            LogisticRegression(solver="lbfgs", multi_class="auto"),
            SVC(gamma="auto"),
            LinearSVC(),
        ]

        x_test_original = self.x_test_iris.copy()

        for model in scikitlearn_test_cases:
            classifier = SklearnClassifier(model=model, clip_values=(0, 1))
            classifier.fit(x=self.x_test_iris, y=self.y_test_iris)

            # Test untargeted attack
            attack = ElasticNet(classifier, targeted=False, max_iter=2, verbose=False)
            x_test_adv = attack.generate(self.x_test_iris)
            self.assertFalse((self.x_test_iris == x_test_adv).all())
            self.assertLessEqual(np.amax(x_test_adv), 1.0)
            self.assertGreaterEqual(np.amin(x_test_adv), 0.0)

            predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
            self.assertFalse((np.argmax(self.y_test_iris, axis=1) == predictions_adv).all())
            accuracy = 1.0 - np.sum(predictions_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
            logger.info("EAD success rate  of " + classifier.__class__.__name__ + " on Iris: %.2f%%", (accuracy * 100))

            # Test targeted attack
            targets = random_targets(self.y_test_iris, nb_classes=3)
            attack = ElasticNet(classifier, targeted=True, max_iter=2, verbose=False)
            x_test_adv = attack.generate(self.x_test_iris, **{"y": targets})
            self.assertFalse((self.x_test_iris == x_test_adv).all())
            self.assertLessEqual(np.amax(x_test_adv), 1.0)
            self.assertGreaterEqual(np.amin(x_test_adv), 0.0)

            predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
            self.assertTrue((np.argmax(targets, axis=1) == predictions_adv).any())
            accuracy = np.sum(predictions_adv == np.argmax(targets, axis=1)) / self.y_test_iris.shape[0]
            logger.info(
                "Targeted EAD success rate of " + classifier.__class__.__name__ + " on Iris: %.2f%%", (accuracy * 100)
            )

            # Check that x_test has not been modified by attack and classifier
            self.assertAlmostEqual(float(np.max(np.abs(x_test_original - self.x_test_iris))), 0.0, delta=0.00001)

    def test_check_params(self):

        ptc = get_image_classifier_pt(from_logits=True)

        with self.assertRaises(ValueError):
            _ = ElasticNet(ptc, binary_search_steps="1.0")
        with self.assertRaises(ValueError):
            _ = ElasticNet(ptc, binary_search_steps=-1)

        with self.assertRaises(ValueError):
            _ = ElasticNet(ptc, max_iter="1.0")
        with self.assertRaises(ValueError):
            _ = ElasticNet(ptc, max_iter=-1)

        with self.assertRaises(ValueError):
            _ = ElasticNet(ptc, batch_size="1.0")
        with self.assertRaises(ValueError):
            _ = ElasticNet(ptc, batch_size=-1)

        with self.assertRaises(ValueError):
            _ = ElasticNet(ptc, decision_rule=1.0)

        with self.assertRaises(ValueError):
            _ = ElasticNet(ptc, verbose="True")


if __name__ == "__main__":
    unittest.main()
