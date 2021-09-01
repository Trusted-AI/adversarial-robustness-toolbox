# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
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
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import get_labels_np_array, random_targets

from tests.utils import TestBase, master_seed
from tests.utils import get_image_classifier_tf, get_image_classifier_pt
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

    def test_pytorch_mnist(self):
        x_train_mnist = np.swapaxes(self.x_train_mnist, 1, 3).astype(np.float32)
        x_test_mnist = np.swapaxes(self.x_test_mnist, 1, 3).astype(np.float32)
        classifier = get_image_classifier_pt()

        scores = get_labels_np_array(classifier.predict(x_train_mnist))
        acc = np.sum(np.argmax(scores, axis=1) == np.argmax(self.y_train_mnist, axis=1)) / self.y_train_mnist.shape[0]
        logger.info("[PyTorch, MNIST] Accuracy on training set: %.2f%%", acc * 100)

        scores = get_labels_np_array(classifier.predict(x_test_mnist))
        acc = np.sum(np.argmax(scores, axis=1) == np.argmax(self.y_test_mnist, axis=1)) / self.y_test_mnist.shape[0]
        logger.info("[PyTorch, MNIST] Accuracy on test set: %.2f%%", acc * 100)

        self._test_backend_mnist(classifier, x_train_mnist, self.y_train_mnist, x_test_mnist, self.y_test_mnist)

    def _test_backend_mnist(self, classifier, x_train, y_train, x_test, y_test):

        base_success_rate = 0.1
        num_iter = 5
        regularization = 100
        batch_size = 5
        eps = 0.3

        # Test Wasserstein with wasserstein ball and wasserstein norm
        attack = Wasserstein(
            classifier,
            regularization=regularization,
            max_iter=num_iter,
            conjugate_sinkhorn_max_iter=num_iter,
            projected_sinkhorn_max_iter=num_iter,
            norm="wasserstein",
            ball="wasserstein",
            targeted=False,
            p=2,
            eps_iter=2,
            eps_factor=1.05,
            eps=eps,
            eps_step=0.1,
            kernel_size=5,
            batch_size=batch_size,
            verbose=False,
        )

        x_train_adv = attack.generate(x_train)
        x_test_adv = attack.generate(x_test)

        self.assertFalse((x_train_adv == x_train).all())
        self.assertFalse((x_test_adv == x_test).all())

        train_y_pred = get_labels_np_array(classifier.predict(x_train_adv)).astype(float)
        test_y_pred = get_labels_np_array(classifier.predict(x_test_adv)).astype(float)

        train_success_rate = (
            np.sum(np.argmax(train_y_pred, axis=1) != np.argmax(classifier.predict(x_train), axis=1)) / y_train.shape[0]
        )
        self.assertGreaterEqual(train_success_rate, base_success_rate)

        test_success_rate = (
            np.sum(np.argmax(test_y_pred, axis=1) != np.argmax(classifier.predict(x_test), axis=1)) / y_test.shape[0]
        )
        self.assertGreaterEqual(test_success_rate, base_success_rate)

        # Test Wasserstein with wasserstein ball and l2 norm
        attack = Wasserstein(
            classifier,
            regularization=regularization,
            max_iter=num_iter,
            conjugate_sinkhorn_max_iter=num_iter,
            projected_sinkhorn_max_iter=num_iter,
            norm="2",
            ball="wasserstein",
            targeted=False,
            p=2,
            eps_iter=2,
            eps_factor=1.05,
            eps=eps,
            eps_step=0.1,
            kernel_size=5,
            batch_size=batch_size,
            verbose=False,
        )

        x_train_adv = attack.generate(x_train)
        x_test_adv = attack.generate(x_test)

        train_y_pred = get_labels_np_array(classifier.predict(x_train_adv)).astype(float)
        test_y_pred = get_labels_np_array(classifier.predict(x_test_adv)).astype(float)

        train_success_rate = (
            np.sum(np.argmax(train_y_pred, axis=1) != np.argmax(classifier.predict(x_train), axis=1)) / y_train.shape[0]
        )
        self.assertGreaterEqual(train_success_rate, base_success_rate)

        test_success_rate = (
            np.sum(np.argmax(test_y_pred, axis=1) != np.argmax(classifier.predict(x_test), axis=1)) / y_test.shape[0]
        )
        self.assertGreaterEqual(test_success_rate, 0)

        # Test Wasserstein with wasserstein ball and inf norm
        attack = Wasserstein(
            classifier,
            regularization=regularization,
            max_iter=num_iter,
            conjugate_sinkhorn_max_iter=num_iter,
            projected_sinkhorn_max_iter=num_iter,
            norm="inf",
            ball="wasserstein",
            targeted=False,
            p=2,
            eps_iter=2,
            eps_factor=1.05,
            eps=eps,
            eps_step=0.1,
            kernel_size=5,
            batch_size=batch_size,
            verbose=False,
        )

        x_train_adv = attack.generate(x_train)
        x_test_adv = attack.generate(x_test)

        train_y_pred = get_labels_np_array(classifier.predict(x_train_adv)).astype(float)
        test_y_pred = get_labels_np_array(classifier.predict(x_test_adv)).astype(float)

        train_success_rate = (
            np.sum(np.argmax(train_y_pred, axis=1) != np.argmax(classifier.predict(x_train), axis=1)) / y_train.shape[0]
        )
        self.assertGreaterEqual(train_success_rate, base_success_rate)

        test_success_rate = (
            np.sum(np.argmax(test_y_pred, axis=1) != np.argmax(classifier.predict(x_test), axis=1)) / y_test.shape[0]
        )
        self.assertGreaterEqual(test_success_rate, 0)

        # Test Wasserstein with wasserstein ball and l1 norm
        attack = Wasserstein(
            classifier,
            regularization=regularization,
            max_iter=num_iter,
            conjugate_sinkhorn_max_iter=num_iter,
            projected_sinkhorn_max_iter=num_iter,
            norm="1",
            ball="wasserstein",
            targeted=False,
            p=2,
            eps_iter=2,
            eps_factor=1.05,
            eps=eps,
            eps_step=0.1,
            kernel_size=5,
            batch_size=batch_size,
            verbose=False,
        )

        x_train_adv = attack.generate(x_train)
        x_test_adv = attack.generate(x_test)

        train_y_pred = get_labels_np_array(classifier.predict(x_train_adv)).astype(float)
        test_y_pred = get_labels_np_array(classifier.predict(x_test_adv)).astype(float)

        train_success_rate = (
            np.sum(np.argmax(train_y_pred, axis=1) != np.argmax(classifier.predict(x_train), axis=1)) / y_train.shape[0]
        )
        self.assertGreaterEqual(train_success_rate, base_success_rate)

        test_success_rate = (
            np.sum(np.argmax(test_y_pred, axis=1) != np.argmax(classifier.predict(x_test), axis=1)) / y_test.shape[0]
        )
        self.assertGreaterEqual(test_success_rate, 0)

        # Test Wasserstein with l2 ball and wasserstein norm
        attack = Wasserstein(
            classifier,
            regularization=regularization,
            max_iter=num_iter,
            conjugate_sinkhorn_max_iter=num_iter,
            projected_sinkhorn_max_iter=num_iter,
            norm="wasserstein",
            ball="2",
            targeted=False,
            p=2,
            eps_iter=2,
            eps_factor=1.05,
            eps=eps,
            eps_step=0.05,
            kernel_size=5,
            batch_size=batch_size,
            verbose=False,
        )

        x_train_adv = attack.generate(x_train)
        x_test_adv = attack.generate(x_test)

        train_y_pred = get_labels_np_array(classifier.predict(x_train_adv)).astype(float)
        test_y_pred = get_labels_np_array(classifier.predict(x_test_adv)).astype(float)

        train_success_rate = (
            np.sum(np.argmax(train_y_pred, axis=1) != np.argmax(classifier.predict(x_train), axis=1)) / y_train.shape[0]
        )
        self.assertGreaterEqual(train_success_rate, 0)

        test_success_rate = (
            np.sum(np.argmax(test_y_pred, axis=1) != np.argmax(classifier.predict(x_test), axis=1)) / y_test.shape[0]
        )
        self.assertGreaterEqual(test_success_rate, 0)

        # Test Wasserstein with l1 ball and wasserstein norm
        attack = Wasserstein(
            classifier,
            regularization=regularization,
            max_iter=num_iter,
            conjugate_sinkhorn_max_iter=num_iter,
            projected_sinkhorn_max_iter=num_iter,
            norm="wasserstein",
            ball="1",
            targeted=False,
            p=2,
            eps_iter=2,
            eps_factor=1.05,
            eps=eps,
            eps_step=0.1,
            kernel_size=5,
            batch_size=batch_size,
            verbose=False,
        )

        x_train_adv = attack.generate(x_train)
        x_test_adv = attack.generate(x_test)

        train_y_pred = get_labels_np_array(classifier.predict(x_train_adv)).astype(float)
        test_y_pred = get_labels_np_array(classifier.predict(x_test_adv)).astype(float)

        train_success_rate = (
            np.sum(np.argmax(train_y_pred, axis=1) != np.argmax(classifier.predict(x_train), axis=1)) / y_train.shape[0]
        )
        self.assertGreaterEqual(train_success_rate, 0)

        test_success_rate = (
            np.sum(np.argmax(test_y_pred, axis=1) != np.argmax(classifier.predict(x_test), axis=1)) / y_test.shape[0]
        )
        self.assertGreaterEqual(test_success_rate, 0)

        # Test Wasserstein with inf ball and Wasserstein norm
        attack = Wasserstein(
            classifier,
            regularization=regularization,
            max_iter=num_iter,
            conjugate_sinkhorn_max_iter=num_iter,
            projected_sinkhorn_max_iter=num_iter,
            norm="wasserstein",
            ball="inf",
            targeted=False,
            p=2,
            eps_iter=2,
            eps_factor=1.05,
            eps=eps,
            eps_step=0.1,
            kernel_size=5,
            batch_size=batch_size,
            verbose=False,
        )

        x_train_adv = attack.generate(x_train)
        x_test_adv = attack.generate(x_test)

        train_y_pred = get_labels_np_array(classifier.predict(x_train_adv)).astype(float)
        test_y_pred = get_labels_np_array(classifier.predict(x_test_adv)).astype(float)

        train_success_rate = (
            np.sum(np.argmax(train_y_pred, axis=1) != np.argmax(classifier.predict(x_train), axis=1)) / y_train.shape[0]
        )
        self.assertGreaterEqual(train_success_rate, base_success_rate)

        test_success_rate = (
            np.sum(np.argmax(test_y_pred, axis=1) != np.argmax(classifier.predict(x_test), axis=1)) / y_test.shape[0]
        )
        self.assertGreaterEqual(test_success_rate, base_success_rate)

        # Test Wasserstein with targeted attack
        master_seed(1234)
        attack = Wasserstein(
            classifier,
            regularization=regularization,
            max_iter=num_iter,
            conjugate_sinkhorn_max_iter=num_iter,
            projected_sinkhorn_max_iter=num_iter,
            norm="wasserstein",
            ball="wasserstein",
            targeted=True,
            p=2,
            eps_iter=2,
            eps_factor=1.05,
            eps=eps,
            eps_step=0.1,
            kernel_size=5,
            batch_size=batch_size,
            verbose=False,
        )

        train_y_rand = random_targets(y_train, nb_classes=10)
        test_y_rand = random_targets(y_test, nb_classes=10)

        x_train_adv = attack.generate(x_train, train_y_rand)
        x_test_adv = attack.generate(x_test, test_y_rand)

        train_y_pred = get_labels_np_array(classifier.predict(x_train_adv)).astype(float)
        test_y_pred = get_labels_np_array(classifier.predict(x_test_adv)).astype(float)

        train_success_rate = (
            np.sum(np.argmax(train_y_pred, axis=1) == np.argmax(train_y_rand, axis=1)) / y_train.shape[0]
        )
        self.assertGreaterEqual(train_success_rate, base_success_rate)

        test_success_rate = np.sum(np.argmax(test_y_pred, axis=1) == np.argmax(test_y_rand, axis=1)) / y_test.shape[0]
        self.assertGreaterEqual(test_success_rate, 0)

        # Test Wasserstein with p-wasserstein=1 and kernel_size=3
        attack = Wasserstein(
            classifier,
            regularization=regularization,
            max_iter=num_iter,
            conjugate_sinkhorn_max_iter=num_iter,
            projected_sinkhorn_max_iter=num_iter,
            norm="wasserstein",
            ball="wasserstein",
            targeted=False,
            p=1,
            eps_iter=2,
            eps_factor=1.05,
            eps=eps,
            eps_step=0.1,
            kernel_size=3,
            batch_size=batch_size,
            verbose=False,
        )

        x_train_adv = attack.generate(x_train)
        x_test_adv = attack.generate(x_test)

        train_y_pred = get_labels_np_array(classifier.predict(x_train_adv)).astype(float)
        test_y_pred = get_labels_np_array(classifier.predict(x_test_adv)).astype(float)

        train_success_rate = (
            np.sum(np.argmax(train_y_pred, axis=1) != np.argmax(classifier.predict(x_train), axis=1)) / y_train.shape[0]
        )
        self.assertTrue(train_success_rate >= base_success_rate)

        test_success_rate = (
            np.sum(np.argmax(test_y_pred, axis=1) != np.argmax(classifier.predict(x_test), axis=1)) / y_test.shape[0]
        )
        self.assertTrue(test_success_rate >= base_success_rate)

    def test_unsquared_images(self):
        from art.estimators.estimator import (
            BaseEstimator,
            LossGradientsMixin,
            NeuralNetworkMixin,
        )

        from art.estimators.classification.classifier import (
            ClassGradientsMixin,
            ClassifierMixin,
        )

        class DummyClassifier(
            ClassGradientsMixin, ClassifierMixin, NeuralNetworkMixin, LossGradientsMixin, BaseEstimator
        ):
            estimator_params = (
                BaseEstimator.estimator_params + NeuralNetworkMixin.estimator_params + ClassifierMixin.estimator_params
            )

            def __init__(self):
                super(DummyClassifier, self).__init__(model=None, clip_values=None, channels_first=True)
                self._nb_classes = 10

            def class_gradient(self):
                return None

            def fit(self):
                pass

            def loss_gradient(self, x, y):
                return np.random.normal(size=(1, 3, 33, 32))

            def predict(self, x, batch_size=1):
                return np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])

            def get_activations(self):
                return None

            def save(self):
                pass

            def compute_loss(self, x, y, **kwargs):
                pass

            def input_shape(self):
                pass

        classifier = DummyClassifier()
        attack = Wasserstein(
            classifier,
            regularization=1,
            kernel_size=3,
            max_iter=1,
            conjugate_sinkhorn_max_iter=10,
            projected_sinkhorn_max_iter=10,
        )

        x = np.random.normal(size=(1, 3, 33, 32))
        x_adv = attack.generate(x)

        self.assertTrue(x_adv.shape == x.shape)

    def test_check_params(self):

        ptc = get_image_classifier_pt(from_logits=True)

        with self.assertRaises(ValueError):
            _ = Wasserstein(ptc, targeted="true")

        with self.assertRaises(ValueError):
            _ = Wasserstein(ptc, regularization=-1)

        with self.assertRaises(TypeError):
            _ = Wasserstein(ptc, p=1.0)
        with self.assertRaises(ValueError):
            _ = Wasserstein(ptc, p=-1)

        with self.assertRaises(TypeError):
            _ = Wasserstein(ptc, kernel_size=1.0)
        with self.assertRaises(ValueError):
            _ = Wasserstein(ptc, kernel_size=2)

        with self.assertRaises(ValueError):
            _ = Wasserstein(ptc, norm=0)

        with self.assertRaises(ValueError):
            _ = Wasserstein(ptc, ball=0)

        with self.assertRaises(ValueError):
            _ = Wasserstein(ptc, eps=-1)

        with self.assertRaises(ValueError):
            _ = Wasserstein(ptc, eps_step=-1)

        with self.assertRaises(ValueError):
            _ = Wasserstein(ptc, norm="inf", eps=1, eps_step=2)

        with self.assertRaises(ValueError):
            _ = Wasserstein(ptc, eps_iter=-1)

        with self.assertRaises(ValueError):
            _ = Wasserstein(ptc, eps_factor=-1)

        with self.assertRaises(ValueError):
            _ = Wasserstein(ptc, max_iter=-1)

        with self.assertRaises(ValueError):
            _ = Wasserstein(ptc, conjugate_sinkhorn_max_iter=-1)

        with self.assertRaises(ValueError):
            _ = Wasserstein(ptc, projected_sinkhorn_max_iter=-1)

        with self.assertRaises(ValueError):
            _ = Wasserstein(ptc, batch_size=-1)

        with self.assertRaises(ValueError):
            _ = Wasserstein(ptc, verbose="true")

    def test_classifier_type_check_fail(self):
        backend_test_classifier_type_check_fail(Wasserstein, (BaseEstimator, LossGradientsMixin, ClassifierMixin))


if __name__ == "__main__":
    unittest.main()
