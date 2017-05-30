from config import config_dict

import unittest

import numpy as np

import keras.backend as K
import tensorflow as tf

from cleverhans.attacks import FastGradientMethod

from src.classifiers import cnn
from src.attackers import perturbations

from src.utils import load_mnist

class TestMinimalPerturbations(unittest.TestCase):

    # def test_cifar(self):
    #
    #     BATCH_SIZE = 10
    #     NB_TRAIN = 1000
    #     NB_TEST = 100
    #
    #     session = tf.Session()
    #     keras.backend.set_session(session)
    #
    #     # get CIFAR10
    #     (X_train, Y_train), (X_test, Y_test) = load_cifar10()
    #     X_train, Y_train, X_test, Y_test = X_train[:NB_TRAIN], Y_train[:NB_TRAIN], X_test[:NB_TEST], Y_test[:NB_TEST]
    #
    #     im_shape = X_train[0].shape
    #
    #     model = cnn.cnn_model(im_shape, act="brelu")
    #
    #     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #
    #     # Fit the model
    #     model.fit(X_train, Y_train, epochs=1, batch_size=BATCH_SIZE)
    #
    #     scores = model.evaluate(X_test, Y_test)
    #
    #     print("\naccuracy: %.2f%%" % (scores[1] * 100))


    def test_mnist(self):

        BATCH_SIZE = 10
        NB_TRAIN = 1000

        session = tf.Session()
        K.set_session(session)

        # get MNIST
        (X_train, Y_train), (_, _) = load_mnist()
        X_train, Y_train = X_train[:NB_TRAIN], Y_train[:NB_TRAIN]

        # get classifier
        im_shape = X_train[0].shape
        model = cnn.cnn_model(im_shape, act="relu")
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Fit the model
        model.fit(X_train, Y_train, epochs=1, batch_size=BATCH_SIZE)

        # Compute minimal perturbations
        emp_robust = perturbations.empirical_robustness(X_train, model, FastGradientMethod, session, eps_step=0.05,
                                                    clip_max=0., clip_min=1.)

        self.assertTrue(isinstance(emp_robust, float))

        emp_robust = perturbations.empirical_robustness(X_train, model, FastGradientMethod, session, eps_step=1.1,
                                                        clip_max=0., clip_min=1.)

        self.assertEqual(emp_robust, 0.)

if __name__ == '__main__':
    unittest.main()