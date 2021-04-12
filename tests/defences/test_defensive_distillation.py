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

from art.defences.transformer.evasion import DefensiveDistillation

from tests.utils import master_seed, TestBase
from tests.utils import get_image_classifier_tf, get_image_classifier_pt, get_image_classifier_kr
from tests.utils import get_tabular_classifier_tf, get_tabular_classifier_kr, get_tabular_classifier_pt

logger = logging.getLogger(__name__)

BATCH_SIZE = 10
NB_EPOCHS = 30


def cross_entropy(prob1, prob2, eps=1e-10):
    """
    Compute cross-entropy between two probability distributions.

    :param prob1: First probability distribution.
    :type prob1: `np.ndarray`
    :param prob2: Second probability distribution.
    :type prob2: `np.ndarray`
    :param eps: A small amount to avoid the possibility of having a log of zero.
    :type eps: `float`
    :return: Cross entropy.
    :rtype: `float`
    """
    prob1 = np.clip(prob1, eps, 1.0 - eps)
    size = prob1.shape[0]
    result = -np.sum(prob2 * np.log(prob1 + eps)) / size

    return result


class TestDefensiveDistillation(TestBase):
    """
    A unittest class for testing the DefensiveDistillation transformer on image data.
    """

    @classmethod
    def setUpClass(cls):
        master_seed(seed=1234, set_tensorflow=True)
        super().setUpClass()

    def setUp(self):
        super().setUp()

    def test_1_tensorflow_classifier(self):
        """
        First test with the TensorFlowClassifier.
        :return:
        """
        # Create the trained classifier
        trained_classifier, sess = get_image_classifier_tf()

        # Create the modified classifier
        transformed_classifier, _ = get_image_classifier_tf(load_init=False, sess=sess)

        # Create defensive distillation transformer
        transformer = DefensiveDistillation(classifier=trained_classifier, batch_size=BATCH_SIZE, nb_epochs=NB_EPOCHS)

        # Perform the transformation
        transformed_classifier = transformer(x=self.x_train_mnist, transformed_classifier=transformed_classifier)

        # Compare the 2 outputs
        preds1 = trained_classifier.predict(x=self.x_train_mnist, batch_size=BATCH_SIZE)

        preds2 = transformed_classifier.predict(x=self.x_train_mnist, batch_size=BATCH_SIZE)

        preds1 = np.argmax(preds1, axis=1)
        preds2 = np.argmax(preds2, axis=1)
        acc = np.sum(preds1 == preds2) / len(preds1)

        self.assertGreater(acc, 0.5)

        ce = cross_entropy(preds1, preds2)

        self.assertLess(ce, 10)
        self.assertGreaterEqual(ce, 0)

        # Clean-up session
        if sess is not None:
            sess.close()

    def test_3_pytorch_classifier(self):
        """
        Second test with the PyTorchClassifier.
        :return:
        """
        self.x_train_mnist = np.reshape(self.x_train_mnist, (self.x_train_mnist.shape[0], 1, 28, 28)).astype(np.float32)

        # Create the trained classifier
        trained_classifier = get_image_classifier_pt()

        # Create the modified classifier
        transformed_classifier = get_image_classifier_pt(load_init=False)

        # Create defensive distillation transformer
        transformer = DefensiveDistillation(classifier=trained_classifier, batch_size=BATCH_SIZE, nb_epochs=NB_EPOCHS)

        # Perform the transformation
        transformed_classifier = transformer(x=self.x_train_mnist, transformed_classifier=transformed_classifier)

        # Compare the 2 outputs
        preds1 = trained_classifier.predict(x=self.x_train_mnist, batch_size=BATCH_SIZE)

        preds2 = transformed_classifier.predict(x=self.x_train_mnist, batch_size=BATCH_SIZE)

        preds1 = np.argmax(preds1, axis=1)
        preds2 = np.argmax(preds2, axis=1)
        acc = np.sum(preds1 == preds2) / len(preds1)

        self.assertGreater(acc, 0.5)

        ce = cross_entropy(preds1, preds2)

        self.assertLess(ce, 10)
        self.assertGreaterEqual(ce, 0)

        self.x_train_mnist = np.reshape(self.x_train_mnist, (self.x_train_mnist.shape[0], 28, 28, 1)).astype(np.float32)

    def test_5_keras_classifier(self):
        """
        Third test with the KerasClassifier.
        :return:
        """
        # Create the trained classifier
        trained_classifier = get_image_classifier_kr()

        # Create the modified classifier
        transformed_classifier = get_image_classifier_kr(load_init=False)

        # Create defensive distillation transformer
        transformer = DefensiveDistillation(classifier=trained_classifier, batch_size=BATCH_SIZE, nb_epochs=NB_EPOCHS)

        # Perform the transformation
        transformed_classifier = transformer(x=self.x_train_mnist, transformed_classifier=transformed_classifier)

        # Compare the 2 outputs
        preds1 = trained_classifier.predict(x=self.x_train_mnist, batch_size=BATCH_SIZE)

        preds2 = transformed_classifier.predict(x=self.x_train_mnist, batch_size=BATCH_SIZE)

        preds1 = np.argmax(preds1, axis=1)
        preds2 = np.argmax(preds2, axis=1)
        acc = np.sum(preds1 == preds2) / len(preds1)

        self.assertGreater(acc, 0.5)

        ce = cross_entropy(preds1, preds2)

        self.assertLess(ce, 10)
        self.assertGreaterEqual(ce, 0)

    def test_2_tensorflow_iris(self):
        """
        First test for TensorFlow.
        :return:
        """
        # Create the trained classifier
        trained_classifier, sess = get_tabular_classifier_tf()

        # Create the modified classifier
        transformed_classifier, _ = get_tabular_classifier_tf(load_init=False, sess=sess)

        # Create defensive distillation transformer
        transformer = DefensiveDistillation(classifier=trained_classifier, batch_size=BATCH_SIZE, nb_epochs=NB_EPOCHS)

        # Perform the transformation
        with self.assertRaises(ValueError) as context:
            _ = transformer(x=self.x_train_iris, transformed_classifier=transformed_classifier)

        self.assertIn("The input trained classifier do not produce probability outputs.", str(context.exception))

        # Clean-up session
        if sess is not None:
            sess.close()

    def test_6_keras_iris(self):
        """
        Second test for Keras.
        :return:
        """
        # Create the trained classifier
        trained_classifier = get_tabular_classifier_kr()

        # Create the modified classifier
        transformed_classifier = get_tabular_classifier_kr(load_init=False)

        # Create defensive distillation transformer
        transformer = DefensiveDistillation(classifier=trained_classifier, batch_size=BATCH_SIZE, nb_epochs=NB_EPOCHS)

        # Perform the transformation
        transformed_classifier = transformer(x=self.x_train_iris, transformed_classifier=transformed_classifier)

        # Compare the 2 outputs
        preds1 = trained_classifier.predict(x=self.x_train_iris, batch_size=BATCH_SIZE)

        preds2 = transformed_classifier.predict(x=self.x_train_iris, batch_size=BATCH_SIZE)

        preds1 = np.argmax(preds1, axis=1)
        preds2 = np.argmax(preds2, axis=1)
        acc = np.sum(preds1 == preds2) / len(preds1)

        self.assertGreater(acc, 0.2)

        ce = cross_entropy(preds1, preds2)

        self.assertLess(ce, 20)
        self.assertGreaterEqual(ce, 0)

    def test_4_pytorch_iris(self):
        """
        Third test for PyTorch.
        :return:
        """
        # Create the trained classifier
        trained_classifier = get_tabular_classifier_pt()

        # Create the modified classifier
        transformed_classifier = get_tabular_classifier_pt(load_init=False)

        # Create defensive distillation transformer
        transformer = DefensiveDistillation(classifier=trained_classifier, batch_size=BATCH_SIZE, nb_epochs=NB_EPOCHS)

        # Perform the transformation
        with self.assertRaises(ValueError) as context:
            _ = transformer(x=self.x_train_iris, transformed_classifier=transformed_classifier)

        self.assertIn("The input trained classifier do not produce probability outputs.", str(context.exception))


if __name__ == "__main__":
    unittest.main()
