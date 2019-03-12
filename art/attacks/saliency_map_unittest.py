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

from art.attacks.saliency_map import SaliencyMapMethod
from art.utils import load_mnist, get_labels_np_array, to_categorical, master_seed
from art.utils import get_classifier_tf, get_classifier_kr, get_classifier_pt

logger = logging.getLogger('testLogger')

# TODO add test with gamma < 1

BATCH_SIZE = 10
NB_TRAIN = 100
NB_TEST = 2


class TestSaliencyMap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        k.set_learning_phase(1)

        # Get MNIST
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        x_train, y_train, x_test, y_test = x_train[:NB_TRAIN], y_train[:NB_TRAIN], x_test[:NB_TEST], y_test[:NB_TEST]
        cls.mnist = (x_train, y_train), (x_test, y_test)

        # Keras classifier
        cls.classifier_k, sess = get_classifier_kr()

        scores = cls.classifier_k._model.evaluate(x_train, y_train)
        logger.info('[Keras, MNIST] Accuracy on training set: %.2f%%', (scores[1] * 100))
        scores = cls.classifier_k._model.evaluate(x_test, y_test)
        logger.info('[Keras, MNIST] Accuracy on test set: %.2f%%', (scores[1] * 100))

        # Create basic CNN on MNIST using TensorFlow
        cls.classifier_tf, sess = get_classifier_tf()

        scores = get_labels_np_array(cls.classifier_tf.predict(x_train))
        acc = np.sum(np.argmax(scores, axis=1) == np.argmax(y_train, axis=1)) / y_train.shape[0]
        logger.info('[TF, MNIST] Accuracy on training set: %.2f%%', (acc * 100))

        scores = get_labels_np_array(cls.classifier_tf.predict(x_test))
        acc = np.sum(np.argmax(scores, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info('[TF, MNIST] Accuracy on test set: %.2f%%', (acc * 100))

        # Create basic PyTorch model
        cls.classifier_py = get_classifier_pt()
        x_train, x_test = np.swapaxes(x_train, 1, 3), np.swapaxes(x_test, 1, 3)

        scores = get_labels_np_array(cls.classifier_py.predict(x_train))
        acc = np.sum(np.argmax(scores, axis=1) == np.argmax(y_train, axis=1)) / y_train.shape[0]
        logger.info('[PyTorch, MNIST] Accuracy on training set: %.2f%%', (acc * 100))

        scores = get_labels_np_array(cls.classifier_py.predict(x_test))
        acc = np.sum(np.argmax(scores, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info('\n[PyTorch, MNIST] Accuracy on test set: %.2f%%', (acc * 100))

    def setUp(self):
        # Set master seed
        master_seed(1234)

    def test_mnist(self):
        # Define all backends to test
        backends = {'keras': self.classifier_k,
                    'tf': self.classifier_tf,
                    'pytorch': self.classifier_py}

        for _, classifier in backends.items():
            if _ == 'pytorch':
                self._swap_axes()
            self._test_mnist_targeted(classifier)
            self._test_mnist_untargeted(classifier)
            if _ == 'pytorch':
                self._swap_axes()

        self.classifier_tf._sess.close()
        tf.reset_default_graph()
        k.clear_session()

    def _swap_axes(self):
        (x_train, y_train), (x_test, y_test) = self.mnist
        x_train = np.swapaxes(x_train, 1, 3)
        x_test = np.swapaxes(x_test, 1, 3)
        self.mnist = (x_train, y_train), (x_test, y_test)

    def _test_mnist_untargeted(self, classifier):
        # Get MNIST
        (_, _), (x_test, y_test) = self.mnist
        x_test, y_test = x_test[:NB_TEST], y_test[:NB_TEST]

        # import time
        df = SaliencyMapMethod(classifier, theta=1)

        # starttime = time.clock()
        # x_test_adv = df.generate(x_test, batch_size=1)
        # endtime = time.clock()
        # print(1, endtime - starttime)
        #
        # starttime = time.clock()
        # x_test_adv = df.generate(x_test, batch_size=10)
        # endtime = time.clock()
        # print(10, endtime - starttime)
        #
        # starttime = time.clock()
        x_test_adv = df.generate(x_test, batch_size=100)
        # endtime = time.clock()
        # print(100, endtime - starttime)

        # starttime = time.clock()
        # x_test_adv = df.generate(x_test, batch_size=1000)
        # endtime = time.clock()
        # print(1000, endtime - starttime)

        self.assertFalse((x_test == x_test_adv).all())
        self.assertFalse((0. == x_test_adv).all())

        y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        self.assertFalse((y_test == y_pred).all())

        acc = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info('Accuracy on adversarial examples: %.2f%%', (acc * 100))

    def _test_mnist_targeted(self, classifier):
        # Get MNIST
        (_, _), (x_test, y_test) = self.mnist
        x_test, y_test = x_test[:NB_TEST], y_test[:NB_TEST]

        # Generate random target classes
        nb_classes = np.unique(np.argmax(y_test, axis=1)).shape[0]
        targets = np.random.randint(nb_classes, size=NB_TEST)
        while (targets == np.argmax(y_test, axis=1)).any():
            targets = np.random.randint(nb_classes, size=NB_TEST)

        # Perform attack
        # import time
        df = SaliencyMapMethod(classifier, theta=1)

        # starttime = time.clock()
        # x_test_adv = df.generate(x_test, y=to_categorical(targets, nb_classes), batch_size=1)
        # endtime = time.clock()
        # print(1, endtime - starttime)
        #
        # starttime = time.clock()
        # x_test_adv = df.generate(x_test, y=to_categorical(targets, nb_classes), batch_size=10)
        # endtime = time.clock()
        # print(10, endtime - starttime)
        #
        # starttime = time.clock()
        x_test_adv = df.generate(x_test, y=to_categorical(targets, nb_classes), batch_size=100)
        # endtime = time.clock()
        # print(100, endtime - starttime)

        # starttime = time.clock()
        # x_test_adv = df.generate(x_test, y=to_categorical(targets, nb_classes), batch_size=1000)
        # endtime = time.clock()
        # print(1000, endtime - starttime)

        self.assertFalse((x_test == x_test_adv).all())
        self.assertFalse((0. == x_test_adv).all())

        y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        self.assertFalse((y_test == y_pred).all())

        acc = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info('Accuracy on adversarial examples: %.2f%%', (acc * 100))


if __name__ == '__main__':
    unittest.main()
