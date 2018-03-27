from __future__ import absolute_import, division, print_function

import unittest

import tensorflow as tf
import numpy as np

from src.classifiers.tensorflow import TFClassifier
from src.utils import load_mnist


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
        loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(
            logits=self._logits, onehot_labels=self._output_ph))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        self._train = optimizer.minimize(loss)

        # Tensorflow session and initialization
        self._sess = tf.Session()
        self._sess.run(tf.group(tf.global_variables_initializer()))

    def test_fit_predict(self):
        # Get MNIST
        (X_train, Y_train), (X_test, Y_test), _, _ = load_mnist()
        X_train, Y_train  = X_train[:NB_TRAIN], Y_train[:NB_TRAIN]
        X_test, Y_test = X_test[:NB_TEST], Y_test[:NB_TEST]

        # Test fit and predict
        tfc = TFClassifier(None, self._input_ph, self._logits, True,
                           self._output_ph, self._train, self._sess)

        tfc.fit(X_train, Y_train, batch_size=100, num_epoch=2)
        preds = tfc.predict(X_test)
        preds_class = np.argmax(preds, axis=1)
        trues_class = np.argmax(Y_test, axis=1)
        acc = np.sum(preds_class==trues_class)/len(trues_class)

        print("\nAccuracy: %.2f%%" % (acc * 100))
        self.assertGreater(acc, 0.1)

    def test_nb_classes(self):
        # Start to test
        tfc = TFClassifier(None, self._input_ph, self._logits, True,
                           self._output_ph, self._train, self._sess)

        self.assertTrue(tfc.nb_classes() == 10)

    def test_gradients(self):
        # Get MNIST
        (X_train, Y_train), (X_test, Y_test), _, _ = load_mnist()
        X_train, Y_train  = X_train[:NB_TRAIN], Y_train[:NB_TRAIN]
        X_test, Y_test = X_test[:NB_TEST], Y_test[:NB_TEST]

        # Test gradients
        tfc = TFClassifier(None, self._input_ph, self._logits, True,
                           self._output_ph, self._train, self._sess)
        trues_class = np.argmax(Y_test, axis=1)
        grads = tfc.gradients(X_test, trues_class)

        self.assertTrue(np.array(grads.shape==(10, 20, 28, 28, 1)).all())
        self.assertTrue(np.sum(grads) != 0)


if __name__ == '__main__':
    unittest.main()





