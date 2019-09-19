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

from art.attacks import DeepFool
from art.classifiers import KerasClassifier
from art.utils import load_dataset, get_labels_np_array, master_seed
from art.utils_test import get_classifier_tf, get_classifier_kr, get_classifier_pt
from art.utils_test import get_iris_classifier_tf, get_iris_classifier_kr, get_iris_classifier_pt

logger = logging.getLogger('testLogger')

BATCH_SIZE = 10
NB_TRAIN = 100
NB_TEST = 11


@unittest.skipIf(tf.__version__[0] == '2', reason='Skip unittests for TensorFlow v2 until Keras supports TensorFlow'
                                                  ' v2 as backend.')
class TestDeepFool(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (x_train, y_train), (x_test, y_test), _, _ = load_dataset('mnist')

        cls.x_train = x_train[:NB_TRAIN]
        cls.y_train = y_train[:NB_TRAIN]
        cls.x_test = x_test[:NB_TEST]
        cls.y_test = y_test[:NB_TEST]

        # Keras classifier
        cls.classifier_k = get_classifier_kr()

        scores = cls.classifier_k._model.evaluate(x_train, y_train)
        logger.info('[Keras, MNIST] Accuracy on training set: %.2f%%', (scores[1] * 100))
        scores = cls.classifier_k._model.evaluate(cls.x_test, cls.y_test)
        logger.info('[Keras, MNIST] Accuracy on test set: %.2f%%', (scores[1] * 100))

        # Create basic CNN on MNIST using TensorFlow
        cls.classifier_tf, sess = get_classifier_tf()

        scores = get_labels_np_array(cls.classifier_tf.predict(x_train))
        accuracy = np.sum(np.argmax(scores, axis=1) == np.argmax(y_train, axis=1)) / y_train.shape[0]
        logger.info('[TF, MNIST] Accuracy on training set: %.2f%%', (accuracy * 100))

        scores = get_labels_np_array(cls.classifier_tf.predict(cls.x_test))
        accuracy = np.sum(np.argmax(scores, axis=1) == np.argmax(cls.y_test, axis=1)) / cls.y_test.shape[0]
        logger.info('[TF, MNIST] Accuracy on test set: %.2f%%', (accuracy * 100))

        # Create basic PyTorch model
        cls.classifier_py = get_classifier_pt()
        x_train, x_test = np.swapaxes(x_train, 1, 3).astype(np.float32), np.swapaxes(
            cls.x_test, 1, 3).astype(np.float32)

        scores = get_labels_np_array(cls.classifier_py.predict(x_train))
        accuracy = np.sum(np.argmax(scores, axis=1) == np.argmax(cls.y_train, axis=1)) / cls.y_train.shape[0]
        logger.info('[PyTorch, MNIST] Accuracy on training set: %.2f%%', (accuracy * 100))

        scores = get_labels_np_array(cls.classifier_py.predict(x_test))
        accuracy = np.sum(np.argmax(scores, axis=1) == np.argmax(cls.y_test, axis=1)) / cls.y_test.shape[0]
        logger.info('[PyTorch, MNIST] Accuracy on test set: %.2f%%', (accuracy * 100))

    def setUp(self):
        master_seed(1234)

    @classmethod
    def tearDownClass(cls):
        k.clear_session()
        cls.classifier_tf._sess.close()
        tf.reset_default_graph()

    def test_mnist(self):
        # Define all backends to test
        backends = {'keras': self.classifier_k,
                    'tf': self.classifier_tf,
                    'pytorch': self.classifier_py}

        for _, classifier in backends.items():
            if _ == 'pytorch':
                self._swap_axes()
                self.x_train = self.x_train.astype(np.float32)
                self.x_test = self.x_test.astype(np.float32)

            self._test_backend_mnist(classifier)

            if _ == 'pytorch':
                self._swap_axes()
                self.x_train = self.x_train.astype(np.float64)
                self.x_test = self.x_test.astype(np.float64)

    def _swap_axes(self):
        self.x_train = np.swapaxes(self.x_train, 1, 3)
        self.x_test = np.swapaxes(self.x_test, 1, 3)

    def _test_backend_mnist(self, classifier):
        attack = DeepFool(classifier, max_iter=5, batch_size=11)
        x_train_adv = attack.generate(self.x_train)
        x_test_adv = attack.generate(self.x_test)

        self.assertFalse((self.x_train == x_train_adv).all())
        self.assertFalse((self.x_test == x_test_adv).all())

        train_y_pred = get_labels_np_array(classifier.predict(x_train_adv))
        test_y_pred = get_labels_np_array(classifier.predict(x_test_adv))

        self.assertFalse((self.y_train == train_y_pred).all())
        self.assertFalse((self.y_test == test_y_pred).all())

        accuracy = np.sum(np.argmax(train_y_pred, axis=1) == np.argmax(self.y_train, axis=1)) / self.y_train.shape[0]
        logger.info('Accuracy on adversarial train examples: %.2f%%', (accuracy * 100))

        accuracy = np.sum(np.argmax(test_y_pred, axis=1) == np.argmax(self.y_test, axis=1)) / self.y_test.shape[0]
        logger.info('Accuracy on adversarial test examples: %.2f%%', (accuracy * 100))

    def test_partial_grads(self):
        attack = DeepFool(self.classifier_k, max_iter=2, nb_grads=3)
        x_test_adv = attack.generate(self.x_test)
        self.assertFalse((self.x_test == x_test_adv).all())

        test_y_pred = get_labels_np_array(self.classifier_k.predict(x_test_adv))
        self.assertFalse((self.y_test == test_y_pred).all())

        accuracy = np.sum(np.argmax(test_y_pred, axis=1) == np.argmax(self.y_test, axis=1)) / self.y_test.shape[0]
        logger.info('Accuracy on adversarial test examples: %.2f%%', (accuracy * 100))

    def test_classifier_type_check_fail_classifier(self):
        # Use a useless test classifier to test basic classifier properties
        class ClassifierNoAPI:
            pass

        classifier = ClassifierNoAPI
        with self.assertRaises(TypeError) as context:
            _ = DeepFool(classifier=classifier)

        self.assertIn('For `DeepFool` classifier must be an instance of '
                      '`art.classifiers.classifier.Classifier`, the provided classifier is instance of '
                      '(<class \'object\'>,).', str(context.exception))

    def test_classifier_type_check_fail_gradients(self):
        # Use a test classifier not providing gradients required by white-box attack
        from art.classifiers.scikitlearn import ScikitlearnDecisionTreeClassifier
        from sklearn.tree import DecisionTreeClassifier

        classifier = ScikitlearnDecisionTreeClassifier(model=DecisionTreeClassifier())
        with self.assertRaises(TypeError) as context:
            _ = DeepFool(classifier=classifier)

        self.assertIn('For `DeepFool` classifier must be an instance of '
                      '`art.classifiers.classifier.ClassifierNeuralNetwork` and '
                      '`art.classifiers.classifier.ClassifierGradients`, the provided classifier is instance of '
                      '(<class \'art.classifiers.scikitlearn.ScikitlearnClassifier\'>,).', str(context.exception))


class TestDeepFoolVectors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (x_train, y_train), (x_test, y_test), _, _ = load_dataset('iris')

        cls.x_train = x_train
        cls.y_train = y_train
        cls.x_test = x_test
        cls.y_test = y_test

    def setUp(self):
        master_seed(1234)

    @unittest.skipIf(tf.__version__[0] == '2', reason='Skip unittests for TensorFlow v2 until Keras supports TensorFlow'
                                                      ' v2 as backend.')
    def test_iris_k_clipped(self):
        classifier, _ = get_iris_classifier_kr()

        attack = DeepFool(classifier, max_iter=5)
        x_test_adv = attack.generate(self.x_test)
        self.assertFalse((self.x_test == x_test_adv).all())
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)

        predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test, axis=1) == predictions_adv).all())
        accuracy = np.sum(predictions_adv == np.argmax(self.y_test, axis=1)) / self.y_test.shape[0]
        logger.info('Accuracy on Iris with DeepFool adversarial examples: %.2f%%', (accuracy * 100))

    @unittest.skipIf(tf.__version__[0] == '2', reason='Skip unittests for TensorFlow v2 until Keras supports TensorFlow'
                                                      ' v2 as backend.')
    def test_iris_k_unbounded(self):
        classifier, _ = get_iris_classifier_kr()

        # Recreate a classifier without clip values
        classifier = KerasClassifier(model=classifier._model, use_logits=False, channel_index=1)
        attack = DeepFool(classifier, max_iter=5, batch_size=128)
        x_test_adv = attack.generate(self.x_test)
        self.assertFalse((self.x_test == x_test_adv).all())

        predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test, axis=1) == predictions_adv).all())
        accuracy = np.sum(predictions_adv == np.argmax(self.y_test, axis=1)) / self.y_test.shape[0]
        logger.info('Accuracy on Iris with DeepFool adversarial examples: %.2f%%', (accuracy * 100))

    def test_iris_tf(self):
        classifier, _ = get_iris_classifier_tf()

        attack = DeepFool(classifier, max_iter=5, batch_size=128)
        x_test_adv = attack.generate(self.x_test)
        self.assertFalse((self.x_test == x_test_adv).all())
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)

        predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test, axis=1) == predictions_adv).all())
        accuracy = np.sum(predictions_adv == np.argmax(self.y_test, axis=1)) / self.y_test.shape[0]
        logger.info('Accuracy on Iris with DeepFool adversarial examples: %.2f%%', (accuracy * 100))

    def test_iris_pt(self):
        classifier = get_iris_classifier_pt()

        attack = DeepFool(classifier, max_iter=5, batch_size=128)
        x_test_adv = attack.generate(self.x_test)
        self.assertFalse((self.x_test == x_test_adv).all())
        self.assertLessEqual(np.amax(x_test_adv), 1.0)
        self.assertGreaterEqual(np.amin(x_test_adv), 0.0)

        predictions_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(self.y_test, axis=1) == predictions_adv).all())
        accuracy = np.sum(predictions_adv == np.argmax(self.y_test, axis=1)) / self.y_test.shape[0]
        logger.info('Accuracy on Iris with DeepFool adversarial examples: %.2f%%', (accuracy * 100))


if __name__ == '__main__':
    unittest.main()
