# MIT License
#
# Copyright (C) IBM Corporation 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the  rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

import numpy as np

from tests.utils import TestBase
from tests.utils import get_classifier_tf, get_classifier_kr, get_classifier_pt

from art.attacks import PixelAttack
from art.utils import get_labels_np_array, to_categorical

logger = logging.getLogger(__name__)


class TestPixelAttack(TestBase):
    """
    TODO: Write Comment
    """

    @classmethod
    def setUpClass(cls):
        """
        TODO: Write Comment
        """
        super().setUpClass()

        cls.n_train = 100
        cls.n_test = 2
        cls.x_train_mnist = cls.x_train_mnist[0:cls.n_train]
        cls.y_train_mnist = cls.y_train_mnist[0:cls.n_train]
        cls.x_test_mnist = cls.x_test_mnist[0:cls.n_test]
        cls.y_test_mnist = cls.y_test_mnist[0:cls.n_test]

    def test_keras_mnist(self):
        """
        TODO: Write Comment
        """
        x_test_original = self.x_test_mnist.copy()

        # Keras classifier
        classifier = get_classifier_kr()

        scores = classifier._model.evaluate(
            self.x_train_mnist, self.y_train_mnist)
        logger.info(
            '[Keras, MNIST] Accuracy on training set: %.2f%%',
            (scores[1] * 100))

        scores = classifier._model.evaluate(
            self.x_test_mnist, self.y_test_mnist)
        logger.info(
            '[Keras, MNIST] Accuracy on test set: %.2f%%',
            (scores[1] * 100))

        # targeted

        # Generate random target classes
        nb_classes = np.unique(np.argmax(self.y_test_mnist, axis=1)).shape[0]
        targets = np.random.randint(nb_classes, size=self.n_test)
        while (targets == np.argmax(self.y_test_mnist, axis=1)).any():
            targets = np.random.randint(nb_classes, size=self.n_test)

        # Perform attack with DE optimizer
        df = PixelAttack(classifier, th=64, es=0, targeted=True)
        x_test_adv = df.generate(
            self.x_test_mnist, y=to_categorical(
                targets, nb_classes))

        self.assertFalse((self.x_test_mnist == x_test_adv).all())
        self.assertFalse((0.0 == x_test_adv).all())

        y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        self.assertFalse((self.y_test_mnist == y_pred).all())

        accuracy = np.sum(
            np.argmax(
                y_pred,
                axis=1) == np.argmax(
                    self.y_test_mnist,
                    axis=1)) / self.n_test
        logger.info(
            'Accuracy on adversarial examples: %.2f%%',
            (accuracy * 100))

        # Perform attack with CMA-ES optimizer
        df = PixelAttack(classifier, th=64, es=0, targeted=True)
        x_test_adv = df.generate(
            self.x_test_mnist, y=to_categorical(
                targets, nb_classes))

        self.assertFalse((self.x_test_mnist == x_test_adv).all())
        self.assertFalse((0.0 == x_test_adv).all())

        y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        self.assertFalse((self.y_test_mnist == y_pred).all())

        accuracy = np.sum(
            np.argmax(
                y_pred,
                axis=1) == np.argmax(
                    self.y_test_mnist,
                    axis=1)) / self.n_test
        logger.info(
            'Accuracy on adversarial examples: %.2f%%',
            (accuracy * 100))

        # untargeted
        # Perform attack with DE optimizer
        df = PixelAttack(classifier, th=64, es=0, targeted=False)
        x_test_adv = df.generate(self.x_test_mnist)

        self.assertFalse((self.x_test_mnist == x_test_adv).all())
        self.assertFalse((0.0 == x_test_adv).all())

        y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        self.assertFalse((self.y_test_mnist == y_pred).all())

        accuracy = np.sum(
            np.argmax(
                y_pred,
                axis=1) == np.argmax(
                    self.y_test_mnist,
                    axis=1)) / self.n_test
        logger.info(
            'Accuracy on adversarial examples: %.2f%%',
            (accuracy * 100))

        # Perform attack with CMA-ES optimizer
        df = PixelAttack(classifier, th=64, es=1, targeted=False)
        x_test_adv = df.generate(self.x_test_mnist)

        self.assertFalse((self.x_test_mnist == x_test_adv).all())
        self.assertFalse((0.0 == x_test_adv).all())

        y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        self.assertFalse((self.y_test_mnist == y_pred).all())

        accuracy = np.sum(
            np.argmax(
                y_pred,
                axis=1) == np.argmax(
                    self.y_test_mnist,
                    axis=1)) / self.n_test
        logger.info(
            'Accuracy on adversarial examples: %.2f%%',
            (accuracy * 100))

        # Check that x_test has not been modified by attack and classifier
        self.assertAlmostEqual(
            float(
                np.max(
                    np.abs(
                        x_test_original -
                        self.x_test_mnist))),
            0.0,
            delta=0.00001)

    def test_tensorflow_mnist(self):
        """
        TODO: Write Comment
        """
        x_test_original = self.x_test_mnist.copy()

        # Create basic CNN on MNIST using TensorFlow
        classifier, sess = get_classifier_tf()

        scores = get_labels_np_array(classifier.predict(self.x_train_mnist))
        accuracy = np.sum(
            np.argmax(
                scores,
                axis=1) == np.argmax(
                    self.y_train_mnist,
                    axis=1)) / self.n_train
        logger.info(
            '[TF, MNIST] Accuracy on training set: %.2f%%',
            (accuracy * 100))

        scores = get_labels_np_array(classifier.predict(self.x_test_mnist))
        accuracy = np.sum(
            np.argmax(
                scores,
                axis=1) == np.argmax(
                    self.y_test_mnist,
                    axis=1)) / self.n_train
        logger.info(
            '[TF, MNIST] Accuracy on test set: %.2f%%',
            (accuracy * 100))

        # targeted
        # Generate random target classes
        nb_classes = np.unique(np.argmax(self.y_test_mnist, axis=1)).shape[0]
        targets = np.random.randint(nb_classes, size=self.n_test)
        while (targets == np.argmax(self.y_test_mnist, axis=1)).any():
            targets = np.random.randint(nb_classes, size=self.n_test)

        # Perform attack with DE optimizer
        df = PixelAttack(classifier, th=64, es=0, targeted=True)
        x_test_adv = df.generate(
            self.x_test_mnist, y=to_categorical(
                targets, nb_classes))

        self.assertFalse((self.x_test_mnist == x_test_adv).all())
        self.assertFalse((0.0 == x_test_adv).all())

        y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        self.assertFalse((self.y_test_mnist == y_pred).all())

        accuracy = np.sum(
            np.argmax(
                y_pred,
                axis=1) == np.argmax(
                    self.y_test_mnist,
                    axis=1)) / self.n_test
        logger.info(
            'Accuracy on adversarial examples: %.2f%%',
            (accuracy * 100))

        # Perform attack with CMA-ES optimizer
        df = PixelAttack(classifier, th=64, es=0, targeted=True)
        x_test_adv = df.generate(
            self.x_test_mnist, y=to_categorical(
                targets, nb_classes))

        self.assertFalse((self.x_test_mnist == x_test_adv).all())
        self.assertFalse((0.0 == x_test_adv).all())

        y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        self.assertFalse((self.y_test_mnist == y_pred).all())

        accuracy = np.sum(
            np.argmax(
                y_pred,
                axis=1) == np.argmax(
                    self.y_test_mnist,
                    axis=1)) / self.n_test
        logger.info(
            'Accuracy on adversarial examples: %.2f%%',
            (accuracy * 100))

        # untargeted
        # Perform attack with DE optimizer
        df = PixelAttack(classifier, th=64, es=0, targeted=False)
        x_test_adv = df.generate(self.x_test_mnist)

        self.assertFalse((self.x_test_mnist == x_test_adv).all())
        self.assertFalse((0.0 == x_test_adv).all())

        y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        self.assertFalse((self.y_test_mnist == y_pred).all())

        accuracy = np.sum(
            np.argmax(
                y_pred,
                axis=1) == np.argmax(
                    self.y_test_mnist,
                    axis=1)) / self.n_test
        logger.info(
            'Accuracy on adversarial examples: %.2f%%',
            (accuracy * 100))

        # Perform attack with CMA-ES optimizer
        df = PixelAttack(classifier, th=64, es=1, targeted=False)
        x_test_adv = df.generate(self.x_test_mnist)

        self.assertFalse((self.x_test_mnist == x_test_adv).all())
        self.assertFalse((0.0 == x_test_adv).all())

        y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        self.assertFalse((self.y_test_mnist == y_pred).all())

        accuracy = np.sum(
            np.argmax(
                y_pred,
                axis=1) == np.argmax(
                    self.y_test_mnist,
                    axis=1)) / self.n_test
        logger.info(
            'Accuracy on adversarial examples: %.2f%%',
            (accuracy * 100))

        # Check that x_test has not been modified by attack and classifier
        self.assertAlmostEqual(
            float(
                np.max(
                    np.abs(
                        x_test_original -
                        self.x_test_mnist))),
            0.0,
            delta=0.00001)

    def test_pytorch_mnist(self):
        """
        TODO: Write Comment
        """
        x_train_mnist = np.swapaxes(
            self.x_train_mnist, 1, 3).astype(
                np.float32)
        x_test_mnist = np.swapaxes(self.x_test_mnist, 1, 3).astype(np.float32)
        x_test_original = x_test_mnist.copy()

        # Create basic PyTorch model
        classifier = get_classifier_pt()

        scores = get_labels_np_array(classifier.predict(x_train_mnist))
        accuracy = np.sum(
            np.argmax(
                scores,
                axis=1) == np.argmax(
                    self.y_train_mnist,
                    axis=1)) / self.n_train
        logger.info(
            '[PyTorch, MNIST] Accuracy on training set: %.2f%%',
            (accuracy * 100))

        scores = get_labels_np_array(classifier.predict(x_test_mnist))
        accuracy = np.sum(
            np.argmax(
                scores,
                axis=1) == np.argmax(
                    self.y_test_mnist,
                    axis=1)) / self.n_test
        logger.info(
            '\n[PyTorch, MNIST] Accuracy on test set: %.2f%%',
            (accuracy * 100))

        # targeted

        # Generate random target classes
        nb_classes = np.unique(np.argmax(self.y_test_mnist, axis=1)).shape[0]
        targets = np.random.randint(nb_classes, size=self.n_test)
        while (targets == np.argmax(self.y_test_mnist, axis=1)).any():
            targets = np.random.randint(nb_classes, size=self.n_test)

        # Perform attack with DE optimizer
        df = PixelAttack(classifier, th=64, es=0, targeted=True)
        x_test_adv = df.generate(
            self.x_test_mnist, y=to_categorical(
                targets, nb_classes))

        self.assertFalse((self.x_test_mnist == x_test_adv).all())
        self.assertFalse((0.0 == x_test_adv).all())

        y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        self.assertFalse((self.y_test_mnist == y_pred).all())

        accuracy = np.sum(
            np.argmax(
                y_pred,
                axis=1) == np.argmax(
                    self.y_test_mnist,
                    axis=1)) / self.n_test
        logger.info(
            'Accuracy on adversarial examples: %.2f%%',
            (accuracy * 100))

        # Perform attack with CMA-ES optimizer
        df = PixelAttack(classifier, th=64, es=0, targeted=True)
        x_test_adv = df.generate(
            self.x_test_mnist, y=to_categorical(
                targets, nb_classes))

        self.assertFalse((self.x_test_mnist == x_test_adv).all())
        self.assertFalse((0.0 == x_test_adv).all())

        y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        self.assertFalse((self.y_test_mnist == y_pred).all())

        accuracy = np.sum(
            np.argmax(
                y_pred,
                axis=1) == np.argmax(
                    self.y_test_mnist,
                    axis=1)) / self.n_test
        logger.info(
            'Accuracy on adversarial examples: %.2f%%',
            (accuracy * 100))

        # untargeted
        # Perform attack with DE optimizer
        df = PixelAttack(classifier, th=64, es=0, targeted=False)
        x_test_adv = df.generate(self.x_test_mnist)

        self.assertFalse((self.x_test_mnist == x_test_adv).all())
        self.assertFalse((0.0 == x_test_adv).all())

        y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        self.assertFalse((self.y_test_mnist == y_pred).all())

        accuracy = np.sum(
            np.argmax(
                y_pred,
                axis=1) == np.argmax(
                    self.y_test_mnist,
                    axis=1)) / self.n_test
        logger.info(
            'Accuracy on adversarial examples: %.2f%%',
            (accuracy * 100))

        # Perform attack with CMA-ES optimizer
        df = PixelAttack(classifier, th=64, es=1, targeted=False)
        x_test_adv = df.generate(self.x_test_mnist)

        self.assertFalse((self.x_test_mnist == x_test_adv).all())
        self.assertFalse((0.0 == x_test_adv).all())

        y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        self.assertFalse((self.y_test_mnist == y_pred).all())

        accuracy = np.sum(
            np.argmax(
                y_pred,
                axis=1) == np.argmax(
                    self.y_test_mnist,
                    axis=1)) / self.n_test
        logger.info(
            'Accuracy on adversarial examples: %.2f%%',
            (accuracy * 100))

        # Check that x_test has not been modified by attack and classifier
        self.assertAlmostEqual(
            float(
                np.max(
                    np.abs(
                        x_test_original -
                        self.x_test_mnist))),
            0.0,
            delta=0.00001)

    def test_classifier_type_check_fail_classifier(self):
        """
        TODO: Write Comment
        """
        # Use a useless test classifier to test basic classifier properties
        class ClassifierNoAPI:
            """
            TODO: Write Comment
            """
            pass

        classifier = ClassifierNoAPI
        with self.assertRaises(TypeError) as context:
            _ = PixelAttack(classifier=classifier)

        self.assertIn(
            'For `PixelAttack` classifier must be an instance of '
            '`art.classifiers.classifier.Classifier`, the provided classifier is instance of '
            '(<class \'object\'>,).', str(
                context.exception))


if __name__ == '__main__':
    unittest.main()
