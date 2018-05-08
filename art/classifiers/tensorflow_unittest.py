from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import tensorflow as tf
import numpy as np

from art.classifiers import TFClassifier
from art.utils import load_mnist


NB_TRAIN = 1000
NB_TEST = 20


class TestTFClassifier(unittest.TestCase):
    """
    This class tests the functionalities of the Tensorflow-based classifier.
    """
    def setUp(self):
        # Define input and output placeholders
        self._input_ph = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        self._output_ph = tf.placeholder(tf.int32, shape=[None, 10])

        # Define the tensorflow graph
        conv = tf.layers.conv2d(self._input_ph, 16, 5, activation=tf.nn.relu)
        conv = tf.layers.max_pooling2d(conv, 2, 2)
        fc = tf.contrib.layers.flatten(conv)

        # Logits layer
        self._logits = tf.layers.dense(fc, 10)

        # Train operator
        self._loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=self._logits, onehot_labels=self._output_ph))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        self._train = optimizer.minimize(self._loss)

        # Tensorflow session and initialization
        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())

    def test_fit_predict(self):
        # Get MNIST
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        x_train, y_train = x_train[:NB_TRAIN], y_train[:NB_TRAIN]
        x_test, y_test = x_test[:NB_TEST], y_test[:NB_TEST]

        # Test fit and predict
        tfc = TFClassifier(None, self._input_ph, self._logits, self._output_ph,
                           self._train, self._loss, None, self._sess)
        tfc.fit(x_train, y_train, batch_size=100, nb_epochs=1)
        preds = tfc.predict(x_test)
        preds_class = np.argmax(preds, axis=1)
        trues_class = np.argmax(y_test, axis=1)
        acc = np.sum(preds_class == trues_class) / len(trues_class)

        print("\nAccuracy: %.2f%%" % (acc * 100))
        self.assertGreater(acc, 0.1)

    def test_nb_classes(self):
        # Start to test
        tfc = TFClassifier(None, self._input_ph, self._logits, self._output_ph, self._train, None, None, self._sess)
        self.assertTrue(tfc.nb_classes == 10)

    def test_input_shape(self):
        # Start to test
        tfc = TFClassifier(None, self._input_ph, self._logits, self._output_ph, self._train, None, None, self._sess)
        self.assertTrue(np.array(tfc.input_shape == (28, 28, 1)).all())

    def test_class_gradient(self):
        # Get MNIST
        (_, _), (x_test, y_test), _, _ = load_mnist()
        x_test, y_test = x_test[:NB_TEST], y_test[:NB_TEST]

        # Test gradient
        tfc = TFClassifier(None, self._input_ph, self._logits, None, None, None, None, self._sess)
        grads = tfc.class_gradient(x_test)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 10, 28, 28, 1)).all())
        self.assertTrue(np.sum(grads) != 0)

    def test_loss_gradient(self):
        # Get MNIST
        (_, _), (x_test, y_test), _, _ = load_mnist()
        x_test, y_test = x_test[:NB_TEST], y_test[:NB_TEST]

        # Test gradient
        tfc = TFClassifier(None, self._input_ph, self._logits, self._output_ph, None, self._loss, None, self._sess)
        grads = tfc.loss_gradient(x_test, y_test)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 28, 28, 1)).all())
        self.assertTrue(np.sum(grads) != 0)


if __name__ == '__main__':
    unittest.main()


