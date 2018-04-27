from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import tensorflow as tf
import keras
import keras.backend as k
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import numpy as np

from art.utils import load_mnist
from art.metrics import clever_t, clever_u
from art.classifiers.tensorflow import TFClassifier
from art.classifiers.keras import KerasClassifier


BATCH_SIZE = 10
NB_TRAIN = 100
NB_TEST = 100


# class TestMinimalPerturbation(unittest.TestCase):
#     # def test_cifar(self):
#     #     session = tf.Session()
#     #     K.set_session(session)
#     #
#     #     # get CIFAR10
#     #     (X_train, Y_train), (X_test, Y_test), _, _ = load_cifar10()
#     #     X_train, Y_train, X_test, Y_test = X_train[:NB_TRAIN], Y_train[:NB_TRAIN], X_test[:NB_TEST], Y_test[:NB_TEST]
#     #     im_shape = X_train[0].shape
#     #
#     #     # Get the classifier
#     #     classifier = CNN(im_shape, act='relu')
#     #     classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     #     classifier.fit(X_train, Y_train, epochs=1, batch_size=BATCH_SIZE)
#     #
#     #     scores = classifier.evaluate(X_test, Y_test)
#     #     print("\naccuracy: %.2f%%" % (scores[1] * 100))
#
#     def test_emp_robustness_mnist(self):
#         session = tf.Session()
#         K.set_session(session)
#
#         comp_params = {"loss": 'categorical_crossentropy',
#                        "optimizer": 'adam',
#                        "metrics": ['accuracy']}
#
#         # get MNIST
#         (X_train, Y_train), (_, _), _, _ = load_mnist()
#         X_train, Y_train = X_train[:NB_TRAIN], Y_train[:NB_TRAIN]
#         im_shape = X_train[0].shape
#
#         # Get classifier
#         classifier = CNN(im_shape, act="relu")
#         classifier.compile(comp_params)
#         classifier.fit(X_train, Y_train, epochs=1, batch_size=BATCH_SIZE)
#
#         # Compute minimal perturbations
#         params = {"eps_step": 1.1,
#                   "clip_min": 0.,
#                   "clip_max": 1.}
#
#         emp_robust = empirical_robustness(X_train, classifier, session, "fgsm", params)
#         self.assertEqual(emp_robust, 0.)
#
#         params = {"eps_step": 1.,
#                   "eps_max": 1.,
#                   "clip_min": None,
#                   "clip_max": None}
#         emp_robust = empirical_robustness(X_train, classifier, session, "fgsm", params)
#         self.assertAlmostEqual(emp_robust, 1., 3)
#
#         params = {"eps_step": 0.1,
#                   "eps_max": 0.2,
#                   "clip_min": None,
#                   "clip_max": None}
#         emp_robust = empirical_robustness(X_train, classifier, session, "fgsm", params)
#         self.assertLessEqual(emp_robust, 0.2)
#
#         # params = {"theta": 1.,
#         #           "gamma": 0.01,
#         #           "clip_min": 0.,
#         #           "clip_max": 1.}
#         # emp_robust_jsma = empirical_robustness(X_train, classifier, session, "jsma", params)
#         # self.assertLessEqual(emp_robust_jsma, 1.)

# class TestMinimalPerturbation(unittest.TestCase):
#     # def test_cifar(self):
#     #     session = tf.Session()
#     #     K.set_session(session)
#     #
#     #     # get CIFAR10
#     #     (X_train, Y_train), (X_test, Y_test), _, _ = load_cifar10()
#     #     X_train, Y_train, X_test, Y_test = X_train[:NB_TRAIN], Y_train[:NB_TRAIN], X_test[:NB_TEST], Y_test[:NB_TEST]
#     #     im_shape = X_train[0].shape
#     #
#     #     # Get the classifier
#     #     classifier = CNN(im_shape, act='relu')
#     #     classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     #     classifier.fit(X_train, Y_train, epochs=1, batch_size=BATCH_SIZE)
#     #
#     #     scores = classifier.evaluate(X_test, Y_test)
#     #     print("\naccuracy: %.2f%%" % (scores[1] * 100))
#
#     def test_emp_robustness_mnist(self):
#         session = tf.Session()
#         k.set_session(session)
#
#         comp_params = {"loss": 'categorical_crossentropy',
#                        "optimizer": 'adam',
#                        "metrics": ['accuracy']}
#
#         # get MNIST
#         (X_train, Y_train), (_, _), _, _ = load_mnist()
#         X_train, Y_train = X_train[:NB_TRAIN], Y_train[:NB_TRAIN]
#         im_shape = X_train[0].shape
#
#         # Get classifier
#         classifier = CNN(im_shape, act="relu")
#         classifier.compile(comp_params)
#         classifier.fit(X_train, Y_train, epochs=1, batch_size=BATCH_SIZE)
#
#         # Compute minimal perturbations
#         params = {"eps_step": 1.1,
#                   "clip_min": 0.,
#                   "clip_max": 1.}
#
#         emp_robust = empirical_robustness(X_train, classifier, session, "fgsm", params)
#         self.assertEqual(emp_robust, 0.)
#
#         params = {"eps_step": 1.,
#                   "eps_max": 1.,
#                   "clip_min": None,
#                   "clip_max": None}
#         emp_robust = empirical_robustness(X_train, classifier, session, "fgsm", params)
#         self.assertAlmostEqual(emp_robust, 1., 3)
#
#         params = {"eps_step": 0.1,
#                   "eps_max": 0.2,
#                   "clip_min": None,
#                   "clip_max": None}
#         emp_robust = empirical_robustness(X_train, classifier, session, "fgsm", params)
#         self.assertLessEqual(emp_robust, 0.2)
#
#         # params = {"theta": 1.,
#         #           "gamma": 0.01,
#         #           "clip_min": 0.,
#         #           "clip_max": 1.}
#         # emp_robust_jsma = empirical_robustness(X_train, classifier, session, "jsma", params)
#         # self.assertLessEqual(emp_robust_jsma, 1.)


#########################################
# This part is the unit test for Clever.#
#########################################
class TestClever(unittest.TestCase):
    """
    Unittest for Clever metrics.
    """
    def _create_tfclassifier(self):
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

    def _create_krclassifier(self):
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
        res0 = clever_t(x_test[-1], tfc, 2, 10, 5, 5, norm=1, pool_factor=3)
        res1 = clever_t(x_test[-1], tfc, 2, 10, 5, 5, norm=2, pool_factor=3)
        res2 = clever_t(x_test[-1], tfc, 2, 10, 5, 5, norm=np.inf, pool_factor=3)
        print("Target tf: ", res0, res1, res2)
        self.assertFalse(res0 == res1)
        self.assertFalse(res1 == res2)
        self.assertFalse(res2 == res0)

        # Test untargeted clever
        res0 = clever_u(x_test[-1], tfc, 10, 5, 5, norm=1, pool_factor=3)
        res1 = clever_u(x_test[-1], tfc, 10, 5, 5, norm=2, pool_factor=3)
        res2 = clever_u(x_test[-1], tfc, 10, 5, 5, norm=np.inf, pool_factor=3)
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
        res0 = clever_t(x_test[-1], krc, 2, 10, 5, 5, norm=1, pool_factor=3)
        res1 = clever_t(x_test[-1], krc, 2, 10, 5, 5, norm=2, pool_factor=3)
        res2 = clever_t(x_test[-1], krc, 2, 10, 5, 5, norm=np.inf, pool_factor=3)
        print("Target kr: ", res0, res1, res2)
        self.assertFalse(res0 == res1)
        self.assertFalse(res1 == res2)
        self.assertFalse(res2 == res0)

        # Test untargeted clever
        res0 = clever_u(x_test[-1], krc, 10, 5, 5, norm=1, pool_factor=3)
        res1 = clever_u(x_test[-1], krc, 10, 5, 5, norm=2, pool_factor=3)
        res2 = clever_u(x_test[-1], krc, 10, 5, 5, norm=np.inf, pool_factor=3)
        print("UnTarget kr: ", res0, res1, res2)
        self.assertFalse(res0 == res1)
        self.assertFalse(res1 == res2)
        self.assertFalse(res2 == res0)


if __name__ == '__main__':
    unittest.main()



