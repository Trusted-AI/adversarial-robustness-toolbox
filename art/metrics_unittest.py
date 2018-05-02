from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import tensorflow as tf
import keras
import keras.backend as k
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import numpy as np

from art.classifiers.tensorflow import TFClassifier
from art.classifiers.keras import KerasClassifier
from art.metrics import empirical_robustness, clever_t, clever_u, loss_sensitivity
from art.utils import load_mnist


BATCH_SIZE = 10
NB_TRAIN = 100
NB_TEST = 100


class TestMetrics(unittest.TestCase):
    def test_emp_robustness_mnist(self):
        # Get MNIST
        (x_train, y_train), (_, _), _, _ = load_mnist()
        x_train, y_train = x_train[:NB_TRAIN], y_train[:NB_TRAIN]

        # Get classifier
        classifier = self._cnn_mnist_k([28, 28, 1])
        classifier.fit(x_train, y_train, batch_size=BATCH_SIZE, nb_epochs=2)

        # Compute minimal perturbations
        params = {"eps_step": 1.1,
                  "clip_min": 0.,
                  "clip_max": 1.}

        emp_robust = empirical_robustness(classifier, x_train, str('fgsm'), params)
        self.assertEqual(emp_robust, 0.)

        params = {"eps_step": 1.,
                  "eps_max": 1.,
                  "clip_min": None,
                  "clip_max": None}
        emp_robust = empirical_robustness(classifier, x_train, str('fgsm'), params)
        self.assertAlmostEqual(emp_robust, 1., 3)

        params = {"eps_step": 0.1,
                  "eps_max": 0.2,
                  "clip_min": None,
                  "clip_max": None}
        emp_robust = empirical_robustness(classifier, x_train, str('fgsm'), params)
        self.assertLessEqual(emp_robust, 0.21)

    def test_loss_sensitivity(self):
        # Get MNIST
        (x_train, y_train), (_, _), _, _ = load_mnist()
        x_train, y_train = x_train[:NB_TRAIN], y_train[:NB_TRAIN]

        # Get classifier
        classifier = self._cnn_mnist_k([28, 28, 1])
        classifier.fit(x_train, y_train, batch_size=BATCH_SIZE, nb_epochs=2)

        l = loss_sensitivity(classifier, x_train)
        self.assertGreaterEqual(l, 0)

    # def testNearestNeighborDist(self):
    #     # Get MNIST
    #     (x_train, y_train), (_, _), _, _ = load_mnist()
    #     x_train, y_train = x_train[:NB_TRAIN], y_train[:NB_TRAIN]
    #
    #     # Get classifier
    #     classifier = self._cnn_mnist_k([28, 28, 1])
    #     classifier.fit(x_train, y_train, batch_size=BATCH_SIZE, nb_epochs=2)
    #
    #     dist = nearest_neighbour_dist(classifier, x_train, x_train, str('fgsm'))
    #     self.assertGreaterEqual(dist, 0)

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

#########################################
# This part is the unit test for Clever.#
#########################################


class TestClever(unittest.TestCase):
    """
    Unittest for Clever metrics.
    """
    @staticmethod
    def _create_tfclassifier():
        """
        To create a simple TFClassifier for testing.
        :return:
        """
        # Define input and output placeholders
        input_ph = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        output_ph = tf.placeholder(tf.int32, shape=[None, 10])

        # Define the tensorflow graph
        conv = tf.layers.conv2d(input_ph, 4, 5, activation=tf.nn.relu)
        conv = tf.layers.max_pooling2d(conv, 2, 2)
        fc = tf.contrib.layers.flatten(conv)

        # Logits layer
        logits = tf.layers.dense(fc, 10)

        # Train operator
        loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=output_ph))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train = optimizer.minimize(loss)

        # Tensorflow session and initialization
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # Create the classifier
        tfc = TFClassifier((0, 1), input_ph, logits, output_ph, train, loss, None, sess)

        return tfc

    @staticmethod
    def _create_krclassifier():
        """
        To create a simple KerasClassifier for testing.
        :return:
        """
        # Initialize a tf session
        session = tf.Session()
        k.set_session(session)

        # Create simple CNN
        model = Sequential()
        model.add(Conv2D(4, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(10, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.01),
                      metrics=['accuracy'])

        # Get the classifier
        krc = KerasClassifier((0, 1), model, use_logits=False)

        return krc

    def test_clever_tf(self):
        """
        Test with tensorflow.
        :return:
        """
        # Get MNIST
        batch_size, nb_train, nb_test = 100, 1000, 10
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        x_train, y_train = x_train[:nb_train], y_train[:nb_train]
        x_test, y_test = x_test[:nb_test], y_test[:nb_test]

        # Get the classifier
        tfc = self._create_tfclassifier()
        tfc.fit(x_train, y_train, batch_size=batch_size, nb_epochs=1)

        # Test targeted clever
        res0 = clever_t(tfc, x_test[-1], 2, 10, 5, 5, norm=1, pool_factor=3)
        res1 = clever_t(tfc, x_test[-1], 2, 10, 5, 5, norm=2, pool_factor=3)
        res2 = clever_t(tfc, x_test[-1], 2, 10, 5, 5, norm=np.inf, pool_factor=3)
        print("Target tf: ", res0, res1, res2)
        self.assertFalse(res0 == res1)
        self.assertFalse(res1 == res2)
        self.assertFalse(res2 == res0)

        # Test untargeted clever
        res0 = clever_u(tfc, x_test[-1], 10, 5, 5, norm=1, pool_factor=3)
        res1 = clever_u(tfc, x_test[-1], 10, 5, 5, norm=2, pool_factor=3)
        res2 = clever_u(tfc, x_test[-1], 10, 5, 5, norm=np.inf, pool_factor=3)
        print("Untarget tf: ", res0, res1, res2)
        self.assertFalse(res0 == res1)
        self.assertFalse(res1 == res2)
        self.assertFalse(res2 == res0)

    def test_clever_kr(self):
        """
        Test with keras.
        :return:
        """
        # Get MNIST
        batch_size, nb_train, nb_test = 100, 1000, 10
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        x_train, y_train = x_train[:nb_train], y_train[:nb_train]
        x_test, y_test = x_test[:nb_test], y_test[:nb_test]

        # Get the classifier
        krc = self._create_krclassifier()
        krc.fit(x_train, y_train, batch_size=batch_size, nb_epochs=1)

        # Test targeted clever
        res0 = clever_t(krc, x_test[-1], 2, 10, 5, 5, norm=1, pool_factor=3)
        res1 = clever_t(krc, x_test[-1], 2, 10, 5, 5, norm=2, pool_factor=3)
        res2 = clever_t(krc, x_test[-1], 2, 10, 5, 5, norm=np.inf, pool_factor=3)
        print("Target kr: ", res0, res1, res2)
        self.assertNotEqual(res0, res1)
        self.assertNotEqual(res1, res2)
        self.assertNotEqual(res2, res0)

        # Test untargeted clever
        res0 = clever_u(krc, x_test[-1], 10, 5, 5, norm=1, pool_factor=3)
        res1 = clever_u(krc, x_test[-1], 10, 5, 5, norm=2, pool_factor=3)
        res2 = clever_u(krc, x_test[-1], 10, 5, 5, norm=np.inf, pool_factor=3)
        print("UnTarget kr: ", res0, res1, res2)
        self.assertNotEqual(res0, res1)
        self.assertNotEqual(res1, res2)
        self.assertNotEqual(res2, res0)


if __name__ == '__main__':
    unittest.main()
