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

import unittest

import keras
import keras.backend as k
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout
import numpy as np
import tensorflow as tf

from art.attacks.saliency_map import SaliencyMapMethod
from art.classifiers.keras import KerasClassifier
from art.classifiers.tensorflow import TFClassifier
from art.utils import load_mnist, get_labels_np_array

# TODO add test with gamma < 1

BATCH_SIZE = 10
NB_TRAIN = 100
NB_TEST = 11


class TestSaliencyMap(unittest.TestCase):

    def setUp(self):
        k.set_learning_phase(1)

        # Get MNIST
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        x_train, y_train, x_test, y_test = x_train[:NB_TRAIN], y_train[:NB_TRAIN], x_test[:NB_TEST], y_test[:NB_TEST]
        self.mnist = ((x_train, y_train), (x_test, y_test))

        # Keras classifier
        self.classifier_k = self._cnn_mnist_k([28, 28, 1])
        self.classifier_k.fit(x_train, y_train, batch_size=BATCH_SIZE, nb_epochs=2)

        scores = self.classifier_k._model.evaluate(x_train, y_train)
        print("\n[Keras, MNIST] Accuracy on training set: %.2f%%" % (scores[1] * 100))
        scores = self.classifier_k._model.evaluate(x_test, y_test)
        print("\n[Keras, MNIST] Accuracy on test set: %.2f%%" % (scores[1] * 100))

        # Create basic CNN on MNIST using TensorFlow
        self.classifier_tf = self._cnn_mnist_tf([28, 28, 1])
        self.classifier_tf.fit(x_train, y_train, nb_epochs=2, batch_size=BATCH_SIZE)

        scores = get_labels_np_array(self.classifier_tf.predict(x_train))
        acc = np.sum(np.argmax(scores, axis=1) == np.argmax(y_train, axis=1)) / y_train.shape[0]
        print('\n[TF, MNIST] Accuracy on training set: %.2f%%' % (acc * 100))

        scores = get_labels_np_array(self.classifier_tf.predict(x_test))
        acc = np.sum(np.argmax(scores, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
        print('\n[TF, MNIST] Accuracy on test set: %.2f%%' % (acc * 100))

    def test_mnist(self):
        # Define all backends to test
        backends = {'keras': self.classifier_k,
                    'tf': self.classifier_tf}

        for _, classifier in backends.items():
            self._test_mnist_targeted(classifier)
            self._test_mnist_untargeted(classifier)

    def _test_mnist_untargeted(self, classifier):
        # Get MNIST
        (_, _), (x_test, y_test) = self.mnist
        x_test, y_test = x_test[:NB_TEST], y_test[:NB_TEST]

        df = SaliencyMapMethod(classifier, theta=1)
        x_test_adv = df.generate(x_test)
        self.assertFalse((x_test == x_test_adv).all())
        self.assertFalse((0. == x_test_adv).all())

        y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        self.assertFalse((y_test == y_pred).all())

        acc = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
        print('\nAccuracy on adversarial examples: %.2f%%' % (acc * 100))

    def _test_mnist_targeted(self, classifier):
        # Get MNIST
        (_, _), (x_test, y_test) = self.mnist
        x_test, y_test = x_test[:NB_TEST], y_test[:NB_TEST]

        # Generate random target classes
        import numpy as np
        nb_classes = np.unique(np.argmax(y_test, axis=1)).shape[0]
        targets = np.random.randint(nb_classes, size=NB_TEST)
        while (targets == np.argmax(y_test, axis=1)).any():
            targets = np.random.randint(nb_classes, size=NB_TEST)

        # Perform attack
        df = SaliencyMapMethod(classifier, theta=1)
        x_test_adv = df.generate(x_test, y=targets)
        self.assertFalse((x_test == x_test_adv).all())
        self.assertFalse((0. == x_test_adv).all())

        y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        self.assertFalse((y_test == y_pred).all())

        acc = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
        print('\nAccuracy on adversarial examples: %.2f%%' % (acc * 100))

    @staticmethod
    def _cnn_mnist_tf(input_shape):
        labels_tf = tf.placeholder(tf.float32, [None, 10])
        inputs_tf = tf.placeholder(tf.float32, [None] + list(input_shape))

        # Define the tensorflow graph
        conv = tf.layers.conv2d(inputs_tf, 4, 5, activation=tf.nn.relu)
        conv = tf.layers.max_pooling2d(conv, 2, 2)
        fc = tf.contrib.layers.flatten(conv)

        # Logits layer
        logits = tf.layers.dense(fc, 10)

        # Train operator
        loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels_tf))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train_tf = optimizer.minimize(loss)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        classifier = TFClassifier((0, 1), inputs_tf, logits, loss=loss, train=train_tf, output_ph=labels_tf, sess=sess)
        return classifier

    @staticmethod
    def _cnn_mnist_k(input_shape):
        # Create simple CNN
        model = Sequential()
        model.add(Conv2D(4, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(10, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.01),
                      metrics=['accuracy'])

        classifier = KerasClassifier((0, 1), model, use_logits=False)
        return classifier


if __name__ == '__main__':
    unittest.main()
