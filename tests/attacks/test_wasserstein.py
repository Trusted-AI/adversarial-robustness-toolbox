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

import numpy as np

from art.attacks.evasion.wasserstein import Wasserstein
from art.estimators.classification.keras import KerasClassifier
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.utils import get_labels_np_array, random_targets

from tests.utils import TestBase
from tests.utils import get_image_classifier_tf, get_image_classifier_kr, get_image_classifier_pt
from tests.utils import get_tabular_classifier_tf, get_tabular_classifier_kr, get_tabular_classifier_pt
from tests.attacks.utils import backend_test_classifier_type_check_fail

logger = logging.getLogger(__name__)


class TestWasserstein(TestBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.n_train = 10
        cls.n_test = 10
        cls.x_train_mnist = cls.x_train_mnist[0 : cls.n_train]
        cls.y_train_mnist = cls.y_train_mnist[0 : cls.n_train]
        cls.x_test_mnist = cls.x_test_mnist[0 : cls.n_test]
        cls.y_test_mnist = cls.y_test_mnist[0 : cls.n_test]

    # def test_keras_mnist(self):
    #     classifier = get_image_classifier_kr()
    #
    #     scores = classifier._model.evaluate(self.x_train_mnist, self.y_train_mnist)
    #     logger.info("[Keras, MNIST] Accuracy on training set: %.2f%%", scores[1] * 100)
    #     scores = classifier._model.evaluate(self.x_test_mnist, self.y_test_mnist)
    #     logger.info("[Keras, MNIST] Accuracy on test set: %.2f%%", scores[1] * 100)
    #
    #     self._test_backend_mnist(
    #         classifier, self.x_train_mnist, self.y_train_mnist, self.x_test_mnist, self.y_test_mnist
    #     )
    #
    def test_tensorflow_mnist(self):
        classifier, sess = get_image_classifier_tf()

        scores = get_labels_np_array(classifier.predict(self.x_train_mnist))
        acc = np.sum(np.argmax(scores, axis=1) == np.argmax(self.y_train_mnist, axis=1)) / self.y_train_mnist.shape[0]
        logger.info("[TF, MNIST] Accuracy on training set: %.2f%%", acc * 100)

        scores = get_labels_np_array(classifier.predict(self.x_test_mnist))
        acc = np.sum(np.argmax(scores, axis=1) == np.argmax(self.y_test_mnist, axis=1)) / self.y_test_mnist.shape[0]
        logger.info("[TF, MNIST] Accuracy on test set: %.2f%%", acc * 100)

        self._test_backend_mnist(
            classifier, self.x_train_mnist, self.y_train_mnist, self.x_test_mnist, self.y_test_mnist
        )

    # def test_pytorch_mnist(self):
    #     x_train_mnist = np.swapaxes(self.x_train_mnist, 1, 3).astype(np.float32)
    #     x_test_mnist = np.swapaxes(self.x_test_mnist, 1, 3).astype(np.float32)
    #     classifier = get_image_classifier_pt()
    #
    #     scores = get_labels_np_array(classifier.predict(x_train_mnist))
    #     acc = np.sum(np.argmax(scores, axis=1) == np.argmax(self.y_train_mnist, axis=1)) / self.y_train_mnist.shape[0]
    #     logger.info("[PyTorch, MNIST] Accuracy on training set: %.2f%%", acc * 100)
    #
    #
    #     scores = get_labels_np_array(classifier.predict(x_test_mnist))
    #     acc = np.sum(np.argmax(scores, axis=1) == np.argmax(self.y_test_mnist, axis=1)) / self.y_test_mnist.shape[0]
    #     logger.info("[PyTorch, MNIST] Accuracy on test set: %.2f%%", acc * 100)
    #     print(np.argmax(scores, axis=1))
    #
    #     self._test_backend_mnist(classifier, x_train_mnist, self.y_train_mnist, x_test_mnist, self.y_test_mnist)

    def _test_backend_mnist(self, classifier, x_train, y_train, x_test, y_test):

        # Test Wasserstein with wasserstein ball and wasserstein norm
        attack = Wasserstein(
            classifier,
            regularization=100,
            max_iter=5,
            conjugate_sinkhorn_max_iter=5,
            projected_sinkhorn_max_iter=5,
            norm='wasserstein',
            ball='wasserstein',
            targeted=False,
            p=2,
            eps_iter=2,
            eps_factor=1.05,
            eps=0.3,
            eps_step=0.1,
            kernel_size=5,
            batch_size=3,
        )

        x_train_adv = attack.generate(x_train)
        x_test_adv = attack.generate(x_test)

        self.assertFalse((x_train == x_train_adv).all())
        self.assertFalse((x_test == x_test_adv).all())

        train_y_pred = get_labels_np_array(classifier.predict(x_train_adv)).astype(float)
        test_y_pred = get_labels_np_array(classifier.predict(x_test_adv)).astype(float)

        self.assertFalse((y_train == train_y_pred).all())
        self.assertFalse((y_test == test_y_pred).all())

        acc1 = np.sum(np.argmax(train_y_pred, axis=1) == np.argmax(y_train, axis=1)) / y_train.shape[0]
        logger.info("Accuracy on adversarial train examples: %.2f%%", acc1 * 100)

        acc2 = np.sum(np.argmax(test_y_pred, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info("Accuracy on adversarial test examples: %.2f%%", acc2 * 100)

        train_success_rate = (
                np.sum(np.argmax(train_y_pred, axis=1) != np.argmax(classifier.predict(x_train), axis=1)) /
                y_train.shape[0]
        )
        self.assertTrue(train_success_rate >= 0.3)

        test_success_rate = (
                np.sum(np.argmax(test_y_pred, axis=1) != np.argmax(classifier.predict(x_test), axis=1)) /
                y_test.shape[0]
        )
        self.assertTrue(test_success_rate >= 0.3)

        # Test Wasserstein with wasserstein ball and l2 norm
        attack = Wasserstein(
            classifier,
            regularization=100,
            max_iter=5,
            conjugate_sinkhorn_max_iter=5,
            projected_sinkhorn_max_iter=5,
            norm='2',
            ball='wasserstein',
            targeted=False,
            p=2,
            eps_iter=2,
            eps_factor=1.05,
            eps=0.3,
            eps_step=0.1,
            kernel_size=5,
            batch_size=3,
        )

        x_train_adv = attack.generate(x_train)
        x_test_adv = attack.generate(x_test)

        self.assertFalse((x_train == x_train_adv).all())
        self.assertFalse((x_test == x_test_adv).all())

        train_y_pred = get_labels_np_array(classifier.predict(x_train_adv)).astype(float)
        test_y_pred = get_labels_np_array(classifier.predict(x_test_adv)).astype(float)

        self.assertFalse((y_train == train_y_pred).all())
        self.assertFalse((y_test == test_y_pred).all())

        acc1 = np.sum(np.argmax(train_y_pred, axis=1) == np.argmax(y_train, axis=1)) / y_train.shape[0]
        logger.info("Accuracy on adversarial train examples: %.2f%%", acc1 * 100)

        acc2 = np.sum(np.argmax(test_y_pred, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info("Accuracy on adversarial test examples: %.2f%%", acc2 * 100)

        train_success_rate = (
                np.sum(np.argmax(train_y_pred, axis=1) != np.argmax(classifier.predict(x_train), axis=1)) /
                y_train.shape[0]
        )
        self.assertTrue(train_success_rate >= 0.3)

        test_success_rate = (
                np.sum(np.argmax(test_y_pred, axis=1) != np.argmax(classifier.predict(x_test), axis=1)) /
                y_test.shape[0]
        )
        self.assertTrue(test_success_rate >= 0.3)

        # Test Wasserstein with wasserstein ball and inf norm
        attack = Wasserstein(
            classifier,
            regularization=100,
            max_iter=5,
            conjugate_sinkhorn_max_iter=5,
            projected_sinkhorn_max_iter=5,
            norm='inf',
            ball='wasserstein',
            targeted=False,
            p=2,
            eps_iter=2,
            eps_factor=1.05,
            eps=0.3,
            eps_step=0.1,
            kernel_size=5,
            batch_size=3,
        )

        x_train_adv = attack.generate(x_train)
        x_test_adv = attack.generate(x_test)

        self.assertFalse((x_train == x_train_adv).all())
        self.assertFalse((x_test == x_test_adv).all())

        train_y_pred = get_labels_np_array(classifier.predict(x_train_adv)).astype(float)
        test_y_pred = get_labels_np_array(classifier.predict(x_test_adv)).astype(float)

        self.assertFalse((y_train == train_y_pred).all())
        self.assertFalse((y_test == test_y_pred).all())

        acc1 = np.sum(np.argmax(train_y_pred, axis=1) == np.argmax(y_train, axis=1)) / y_train.shape[0]
        logger.info("Accuracy on adversarial train examples: %.2f%%", acc1 * 100)

        acc2 = np.sum(np.argmax(test_y_pred, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info("Accuracy on adversarial test examples: %.2f%%", acc2 * 100)

        train_success_rate = (
                np.sum(np.argmax(train_y_pred, axis=1) != np.argmax(classifier.predict(x_train), axis=1)) /
                y_train.shape[0]
        )
        self.assertTrue(train_success_rate >= 0.3)

        test_success_rate = (
                np.sum(np.argmax(test_y_pred, axis=1) != np.argmax(classifier.predict(x_test), axis=1)) /
                y_test.shape[0]
        )
        self.assertTrue(test_success_rate >= 0.3)

        # Test Wasserstein with wasserstein ball and l1 norm
        attack = Wasserstein(
            classifier,
            regularization=100,
            max_iter=5,
            conjugate_sinkhorn_max_iter=5,
            projected_sinkhorn_max_iter=5,
            norm='1',
            ball='wasserstein',
            targeted=False,
            p=2,
            eps_iter=2,
            eps_factor=1.05,
            eps=0.3,
            eps_step=0.1,
            kernel_size=5,
            batch_size=3,
        )

        x_train_adv = attack.generate(x_train)
        x_test_adv = attack.generate(x_test)

        self.assertFalse((x_train == x_train_adv).all())
        self.assertFalse((x_test == x_test_adv).all())

        train_y_pred = get_labels_np_array(classifier.predict(x_train_adv)).astype(float)
        test_y_pred = get_labels_np_array(classifier.predict(x_test_adv)).astype(float)

        self.assertFalse((y_train == train_y_pred).all())
        self.assertFalse((y_test == test_y_pred).all())

        acc1 = np.sum(np.argmax(train_y_pred, axis=1) == np.argmax(y_train, axis=1)) / y_train.shape[0]
        logger.info("Accuracy on adversarial train examples: %.2f%%", acc1 * 100)

        acc2 = np.sum(np.argmax(test_y_pred, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info("Accuracy on adversarial test examples: %.2f%%", acc2 * 100)

        train_success_rate = (
                np.sum(np.argmax(train_y_pred, axis=1) != np.argmax(classifier.predict(x_train), axis=1)) /
                y_train.shape[0]
        )
        self.assertTrue(train_success_rate >= 0.3)

        test_success_rate = (
                np.sum(np.argmax(test_y_pred, axis=1) != np.argmax(classifier.predict(x_test), axis=1)) /
                y_test.shape[0]
        )
        self.assertTrue(test_success_rate >= 0.3)


        # Test Wasserstein with wasserstein ball and inf norm
        attack = Wasserstein(
            classifier,
            regularization=100,
            max_iter=5,
            conjugate_sinkhorn_max_iter=5,
            projected_sinkhorn_max_iter=5,
            norm='inf',
            ball='wasserstein',
            targeted=False,
            p=2,
            eps_iter=2,
            eps_factor=1.05,
            eps=0.3,
            eps_step=0.1,
            kernel_size=5,
            batch_size=3,
        )

        x_train_adv = attack.generate(x_train)
        x_test_adv = attack.generate(x_test)

        self.assertFalse((x_train == x_train_adv).all())
        self.assertFalse((x_test == x_test_adv).all())

        train_y_pred = get_labels_np_array(classifier.predict(x_train_adv)).astype(float)
        test_y_pred = get_labels_np_array(classifier.predict(x_test_adv)).astype(float)

        self.assertFalse((y_train == train_y_pred).all())
        self.assertFalse((y_test == test_y_pred).all())

        acc1 = np.sum(np.argmax(train_y_pred, axis=1) == np.argmax(y_train, axis=1)) / y_train.shape[0]
        logger.info("Accuracy on adversarial train examples: %.2f%%", acc1 * 100)

        acc2 = np.sum(np.argmax(test_y_pred, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info("Accuracy on adversarial test examples: %.2f%%", acc2 * 100)

        train_success_rate = (
                np.sum(np.argmax(train_y_pred, axis=1) != np.argmax(classifier.predict(x_train), axis=1)) /
                y_train.shape[0]
        )
        self.assertTrue(train_success_rate >= 0.3)

        test_success_rate = (
                np.sum(np.argmax(test_y_pred, axis=1) != np.argmax(classifier.predict(x_test), axis=1)) /
                y_test.shape[0]
        )
        self.assertTrue(test_success_rate >= 0.3)


    #
    # def test_classifier_type_check_fail(self):
    #     backend_test_classifier_type_check_fail(ProjectedGradientDescent, [BaseEstimator, LossGradientsMixin])


if __name__ == "__main__":
    unittest.main()
