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

import keras
import numpy as np

from art.attacks.evasion.deepfool import DeepFool
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassGradientsMixin
from art.estimators.classification.keras import KerasClassifier
from art.utils import get_labels_np_array
from tests.attacks.utils import backend_test_classifier_type_check_fail
from tests.utils import (
    TestBase,
    get_image_classifier_kr,
    get_image_classifier_pt,
    get_image_classifier_tf,
    get_tabular_classifier_kr,
    get_tabular_classifier_pt,
    get_tabular_classifier_tf,
)

logger = logging.getLogger(__name__)


class TestDeepFool(TestBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.n_train = 100
        cls.n_test = 11
        cls.x_train_mnist = cls.x_train_mnist[0 : cls.n_train]
        cls.y_train_mnist = cls.y_train_mnist[0 : cls.n_train]
        cls.x_test_mnist = cls.x_test_mnist[0 : cls.n_test]
        cls.y_test_mnist = cls.y_test_mnist[0 : cls.n_test]

    @unittest.skipIf(
        not (int(keras.__version__.split(".")[0]) == 2 and int(keras.__version__.split(".")[1]) >= 3),
        reason="Minimal version of Keras or TensorFlow required.",
    )
    def test_8_keras_mnist(self):
        x_test_original = self.x_test_mnist.copy()

        # Keras classifier
        classifier = get_image_classifier_kr(from_logits=True)

        scores = classifier._model.evaluate(self.x_train_mnist, self.y_train_mnist)
        logger.info("[Keras, MNIST] Accuracy on training set: %.2f%%", (scores[1] * 100))
        scores = classifier._model.evaluate(self.x_test_mnist, self.y_test_mnist)
        logger.info("[Keras, MNIST] Accuracy on test set: %.2f%%", (scores[1] * 100))

        attack = DeepFool(classifier, max_iter=5, batch_size=11, verbose=False)
        x_train_adv = attack.generate(self.x_train_mnist)
        x_test_adv = attack.generate(self.x_test_mnist)

        self.assertFalse((self.x_train_mnist == x_train_adv).all())
        self.assertFalse((self.x_test_mnist == x_test_adv).all())

        train_y_pred = get_labels_np_array(classifier.predict(x_train_adv))
        test_y_pred = get_labels_np_array(classifier.predict(x_test_adv))

        self.assertFalse((self.y_train_mnist == train_y_pred).all())
        self.assertFalse((self.y_test_mnist == test_y_pred).all())

        sum_0 = np.sum(np.argmax(train_y_pred, axis=1) == np.argmax(self.y_train_mnist, axis=1))
        accuracy_0 = sum_0 / self.y_train_mnist.shape[0]
        logger.info("Accuracy on adversarial train examples: %.2f%%", (accuracy_0 * 100))

        sum_1 = np.sum(np.argmax(test_y_pred, axis=1) == np.argmax(self.y_test_mnist, axis=1))
        accuracy_1 = sum_1 / self.y_test_mnist.shape[0]
        logger.info("Accuracy on adversarial test examples: %.2f%%", (accuracy_1 * 100))

        # Check that x_test has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - self.x_test_mnist))), 0.0, delta=0.00001)

    def test_3_tensorflow_mnist(self):
        x_test_original = self.x_test_mnist.copy()

        # Create basic CNN on MNIST using TensorFlow
        classifier, sess = get_image_classifier_tf(from_logits=True)

        scores = get_labels_np_array(classifier.predict(self.x_train_mnist))
        sum2 = np.sum(np.argmax(scores, axis=1) == np.argmax(self.y_train_mnist, axis=1))
        accuracy = sum2 / self.y_train_mnist.shape[0]
        logger.info("[TF, MNIST] Accuracy on training set: %.2f%%", (accuracy * 100))

        scores = get_labels_np_array(classifier.predict(self.x_test_mnist))
        sum3 = np.sum(np.argmax(scores, axis=1) == np.argmax(self.y_test_mnist, axis=1))
        accuracy = sum3 / self.y_test_mnist.shape[0]
        logger.info("[TF, MNIST] Accuracy on test set: %.2f%%", (accuracy * 100))

        attack = DeepFool(classifier, max_iter=5, batch_size=11, verbose=False)
        x_train_adv = attack.generate(self.x_train_mnist)
        x_test_adv = attack.generate(self.x_test_mnist)

        self.assertFalse((self.x_train_mnist == x_train_adv).all())
        self.assertFalse((self.x_test_mnist == x_test_adv).all())

        train_y_pred = get_labels_np_array(classifier.predict(x_train_adv))
        test_y_pred = get_labels_np_array(classifier.predict(x_test_adv))

        self.assertFalse((self.y_train_mnist == train_y_pred).all())
        self.assertFalse((self.y_test_mnist == test_y_pred).all())

        sum4 = np.sum(np.argmax(train_y_pred, axis=1) == np.argmax(self.y_train_mnist, axis=1))
        accuracy = sum4 / self.y_train_mnist.shape[0]
        logger.info("Accuracy on adversarial train examples: %.2f%%", (accuracy * 100))

        sum5 = np.sum(np.argmax(test_y_pred, axis=1) == np.argmax(self.y_test_mnist, axis=1))
        accuracy = sum5 / self.y_test_mnist.shape[0]
        logger.info("Accuracy on adversarial test examples: %.2f%%", (accuracy * 100))

        # Check that x_test has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - self.x_test_mnist))), 0.0, delta=0.00001)

    def test_5_pytorch_mnist(self):
        x_train = np.reshape(self.x_train_mnist, (self.x_train_mnist.shape[0], 1, 28, 28)).astype(np.float32)
        x_test = np.reshape(self.x_test_mnist, (self.x_test_mnist.shape[0], 1, 28, 28)).astype(np.float32)
        x_test_original = x_test.copy()

        # Create basic PyTorch model
        classifier = get_image_classifier_pt(from_logits=True)

        scores = get_labels_np_array(classifier.predict(x_train))
        sum6 = np.sum(np.argmax(scores, axis=1) == np.argmax(self.y_train_mnist, axis=1))
        accuracy = sum6 / self.y_train_mnist.shape[0]
        logger.info("[PyTorch, MNIST] Accuracy on training set: %.2f%%", (accuracy * 100))

        scores = get_labels_np_array(classifier.predict(x_test))
        sum7 = np.sum(np.argmax(scores, axis=1) == np.argmax(self.y_test_mnist, axis=1))
        accuracy = sum7 / self.y_test_mnist.shape[0]
        logger.info("[PyTorch, MNIST] Accuracy on test set: %.2f%%", (accuracy * 100))

        attack = DeepFool(classifier, max_iter=5, batch_size=11, verbose=False)
        x_train_adv = attack.generate(x_train)
        x_test_adv = attack.generate(x_test)

        self.assertFalse((x_train == x_train_adv).all())
        self.assertFalse((x_test == x_test_adv).all())

        train_y_pred = get_labels_np_array(classifier.predict(x_train_adv))
        test_y_pred = get_labels_np_array(classifier.predict(x_test_adv))

        self.assertFalse((self.y_train_mnist == train_y_pred).all())
        self.assertFalse((self.y_test_mnist == test_y_pred).all())

        sum8 = np.sum(np.argmax(train_y_pred, axis=1) == np.argmax(self.y_train_mnist, axis=1))
        accuracy = sum8 / self.y_train_mnist.shape[0]
        logger.info("Accuracy on adversarial train examples: %.2f%%", (accuracy * 100))

        sum9 = np.sum(np.argmax(test_y_pred, axis=1) == np.argmax(self.y_test_mnist, axis=1))
        accuracy = sum9 / self.y_test_mnist.shape[0]
        logger.info("Accuracy on adversarial test examples: %.2f%%", (accuracy * 100))

        # Check that x_test has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test))), 0.0, delta=0.00001)

    @unittest.skipIf(
        not (int(keras.__version__.split(".")[0]) == 2 and int(keras.__version__.split(".")[1]) >= 3),
        reason="Minimal version of Keras or TensorFlow required.",
    )
    def test_9_keras_mnist_partial_grads(self):
        classifier = get_image_classifier_kr(from_logits=True)
        attack = DeepFool(classifier, max_iter=2, nb_grads=3, verbose=False)
        x_test_adv = attack.generate(self.x_test_mnist)
        self.assertFalse((self.x_test_mnist == x_test_adv).all())

        test_y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        self.assertFalse((self.y_test_mnist == test_y_pred).all())
        sum10 = np.sum(np.argmax(test_y_pred, axis=1) == np.argmax(self.y_test_mnist, axis=1))
        accuracy = sum10 / self.y_test_mnist.shape[0]
        logger.info("Accuracy on adversarial test examples: %.2f%%", (accuracy * 100))

    def test_1_classifier_type_check_fail(self):
        backend_test_classifier_type_check_fail(DeepFool, [BaseEstimator, ClassGradientsMixin])

    def test_6_keras_iris_clipped(self):
        classifier = get_tabular_classifier_kr()

        attack = DeepFool(classifier, max_iter=5, verbose=False)
        x_test_adv = attack.generate(self.x_test_iris)
        self.assertFalse((self.x_test_iris == x_test_adv).all())
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)

        predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test_iris, axis=1) == predictions_adv).all())
        accuracy = np.sum(predictions_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
        logger.info("Accuracy on Iris with DeepFool adversarial examples: %.2f%%", (accuracy * 100))

    def test_7_keras_iris_unbounded(self):
        classifier = get_tabular_classifier_kr()

        # Recreate a classifier without clip values
        classifier = KerasClassifier(model=classifier._model, use_logits=False, channels_first=True)
        attack = DeepFool(classifier, max_iter=5, batch_size=128, verbose=False)
        x_test_adv = attack.generate(self.x_test_iris)
        self.assertFalse((self.x_test_iris == x_test_adv).all())

        predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test_iris, axis=1) == predictions_adv).all())
        accuracy = np.sum(predictions_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
        logger.info("Accuracy on Iris with DeepFool adversarial examples: %.2f%%", (accuracy * 100))

    def test_2_tensorflow_iris(self):
        classifier, _ = get_tabular_classifier_tf()

        attack = DeepFool(classifier, max_iter=5, batch_size=128, verbose=False)
        x_test_adv = attack.generate(self.x_test_iris)
        self.assertFalse((self.x_test_iris == x_test_adv).all())
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)

        predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test_iris, axis=1) == predictions_adv).all())
        accuracy = np.sum(predictions_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
        logger.info("Accuracy on Iris with DeepFool adversarial examples: %.2f%%", (accuracy * 100))

    def test_4_pytorch_iris(self):
        classifier = get_tabular_classifier_pt()

        attack = DeepFool(classifier, max_iter=5, batch_size=128, verbose=False)
        x_test_adv = attack.generate(self.x_test_iris)
        self.assertFalse((self.x_test_iris == x_test_adv).all())
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)

        predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test_iris, axis=1) == predictions_adv).all())
        accuracy = np.sum(predictions_adv == np.argmax(self.y_test_iris, axis=1)) / self.y_test_iris.shape[0]
        logger.info("Accuracy on Iris with DeepFool adversarial examples: %.2f%%", (accuracy * 100))


if __name__ == "__main__":
    unittest.main()
