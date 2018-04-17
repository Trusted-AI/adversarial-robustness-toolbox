from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import keras.backend as k
import tensorflow as tf
import numpy as np

from art.classifiers.cnn import CNN
from art.metrics import empirical_robustness
from art.utils import load_mnist, load_cifar10
from art.metrics import clever_t, clever_u
from art.classifiers.classifier import Classifier

BATCH_SIZE = 10
NB_TRAIN = 100
NB_TEST = 100


class TestMinimalPerturbation(unittest.TestCase):
    # def test_cifar(self):
    #     session = tf.Session()
    #     K.set_session(session)
    #
    #     # get CIFAR10
    #     (X_train, Y_train), (X_test, Y_test), _, _ = load_cifar10()
    #     X_train, Y_train, X_test, Y_test = X_train[:NB_TRAIN], Y_train[:NB_TRAIN], X_test[:NB_TEST], Y_test[:NB_TEST]
    #     im_shape = X_train[0].shape
    #
    #     # Get the classifier
    #     classifier = CNN(im_shape, act='relu')
    #     classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #     classifier.fit(X_train, Y_train, epochs=1, batch_size=BATCH_SIZE)
    #
    #     scores = classifier.evaluate(X_test, Y_test)
    #     print("\naccuracy: %.2f%%" % (scores[1] * 100))

    def test_emp_robustness_mnist(self):
        session = tf.Session()
        k.set_session(session)

        comp_params = {"loss": 'categorical_crossentropy',
                       "optimizer": 'adam',
                       "metrics": ['accuracy']}

        # get MNIST
        (X_train, Y_train), (_, _), _, _ = load_mnist()
        X_train, Y_train = X_train[:NB_TRAIN], Y_train[:NB_TRAIN]
        im_shape = X_train[0].shape

        # Get classifier
        classifier = CNN(im_shape, act="relu")
        classifier.compile(comp_params)
        classifier.fit(X_train, Y_train, epochs=1, batch_size=BATCH_SIZE)

        # Compute minimal perturbations
        params = {"eps_step": 1.1,
                  "clip_min": 0.,
                  "clip_max": 1.}

        emp_robust = empirical_robustness(X_train, classifier, session, "fgsm", params)
        self.assertEqual(emp_robust, 0.)

        params = {"eps_step": 1.,
                  "eps_max": 1.,
                  "clip_min": None,
                  "clip_max": None}
        emp_robust = empirical_robustness(X_train, classifier, session, "fgsm", params)
        self.assertAlmostEqual(emp_robust, 1., 3)

        params = {"eps_step": 0.1,
                  "eps_max": 0.2,
                  "clip_min": None,
                  "clip_max": None}
        emp_robust = empirical_robustness(X_train, classifier, session, "fgsm", params)
        self.assertLessEqual(emp_robust, 0.2)

        # params = {"theta": 1.,
        #           "gamma": 0.01,
        #           "clip_min": 0.,
        #           "clip_max": 1.}
        # emp_robust_jsma = empirical_robustness(X_train, classifier, session, "jsma", params)
        # self.assertLessEqual(emp_robust_jsma, 1.)


#########################################
# This part is the unit test for Clever.#
#########################################

class TestClassifier(Classifier):
    def __init__(self, defences=None, preproc=None):
        from keras.models import Sequential
        from keras.layers import Lambda
        model = Sequential(name="TestClassifier")
        model.add(Lambda(lambda x: x + 0, input_shape=(2,)))

        super(TestClassifier, self).__init__(model, defences, preproc)


class TestClever(unittest.TestCase):
    """
    Unittest for Clever metrics.
    """
    def test_clever_t_unit(self):
        """
        Test the targeted version with simplified data.
        :return:
        """
        print("Unit test for the targeted version with simplified data.")
        # Define session & params
        session = tf.Session()
        k.set_session(session)

        # Get classifier
        classifier = TestClassifier()

        # Compute scores
        res = clever_t(np.array([1, 0]), classifier, 1, 20, 10, 1, session)

        # Test
        self.assertAlmostEqual(res[0], 0.9999999999999998, delta=0.00001)
        self.assertAlmostEqual(res[1], 0.7071067811865474, delta=0.00001)
        self.assertAlmostEqual(res[2], 0.4999999999999999, delta=0.00001)

    def test_clever_u_unit(self):
        """
        Test the untargeted version with simplified data.
        :return:
        """
        print("Unit test for the untargeted version with simplified data.")
        # Define session & params
        session = tf.Session()
        k.set_session(session)

        # Get classifier
        classifier = TestClassifier()

        # Compute scores
        res = clever_u(np.array([1, 0]), classifier, 20, 10, 1, session)

        # Test
        self.assertAlmostEqual(res[0], 0.9999999999999998, delta=0.00001)
        self.assertAlmostEqual(res[1], 0.7071067811865474, delta=0.00001)
        self.assertAlmostEqual(res[2], 0.4999999999999999, delta=0.00001)

    def test_clever_t(self):
        """
        Test the targeted version.
        :return:
        """
        print("Test if the targeted version works on a true classifier/data")
        # Define session & params
        session = tf.Session()
        k.set_session(session)

        comp_params = {"loss": 'categorical_crossentropy', "optimizer": 'adam',
                       "metrics": ['accuracy']}

        # Get MNIST
        (X_train, Y_train), (_, _), _, _ = load_mnist()
        X_train, Y_train = X_train[:NB_TRAIN], Y_train[:NB_TRAIN]
        im_shape = X_train[0].shape

        # Get classifier
        classifier = CNN(im_shape, act="relu")
        classifier.compile(comp_params)
        classifier.fit(X_train, Y_train, epochs=1,
                       batch_size=BATCH_SIZE, verbose=0)

        res = clever_t(X_train[-1], classifier, 7, 20, 10, 5, session)
        self.assertGreater(res[0], res[1])
        self.assertGreater(res[1], res[2])

    def test_clever_u(self):
        """
        Test the untargeted version.
        :return:
        """
        print("Test if the untargeted version works on a true classifier/data")
        # Define session & params
        session = tf.Session()
        k.set_session(session)

        comp_params = {"loss": 'categorical_crossentropy', "optimizer": 'adam',
                       "metrics": ['accuracy']}

        # Get MNIST
        (X_train, Y_train), (_, _), _, _ = load_mnist()
        X_train, Y_train = X_train[:NB_TRAIN], Y_train[:NB_TRAIN]
        im_shape = X_train[0].shape

        # Get classifier
        classifier = CNN(im_shape, act="relu")
        classifier.compile(comp_params)
        classifier.fit(X_train, Y_train, epochs=1,
                       batch_size=BATCH_SIZE, verbose=0)

        res = clever_u(X_train[-1], classifier, 2, 10, 5, session)
        self.assertGreater(res[0], res[1])
        self.assertGreater(res[1], res[2])


if __name__ == '__main__':
    unittest.main()
