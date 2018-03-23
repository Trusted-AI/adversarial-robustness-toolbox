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

# from config import config_dict
import unittest

import keras.backend as K
import tensorflow as tf
import numpy as np

from src.classifiers.cnn import CNN
from src.metrics import empirical_robustness
from src.utils import load_mnist, load_cifar10
from src.classifiers.classifier import Classifier

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
        K.set_session(session)

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


if __name__ == '__main__':
    unittest.main()
