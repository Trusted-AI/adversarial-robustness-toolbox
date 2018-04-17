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

import keras.backend as k
import tensorflow as tf

from art.attacks.fast_gradient import FastGradientMethod
from art.classifiers.cnn import CNN
from art.utils import load_mnist, get_labels_np_array


class TestFastGradientMethod(unittest.TestCase):
    def test_mnist(self):
        session = tf.Session()
        k.set_session(session)

        comp_params = {"loss": 'categorical_crossentropy',
                       "optimizer": 'adam',
                       "metrics": ['accuracy']}

        # get MNIST
        batch_size, nb_train, nb_test = 100, 1000, 100
        (X_train, Y_train), (X_test, Y_test), _, _ = load_mnist()
        X_train, Y_train = X_train[:nb_train], Y_train[:nb_train]
        X_test, Y_test = X_test[:nb_test], Y_test[:nb_test]
        im_shape = X_train[0].shape

        # get classifier
        classifier = CNN(im_shape, act="relu")
        classifier.compile(comp_params)
        classifier.fit(X_train, Y_train, epochs=1, batch_size=batch_size, verbose=0)
        scores = classifier.evaluate(X_train, Y_train)
        print("\naccuracy on training set: %.2f%%" % (scores[1] * 100))
        scores = classifier.evaluate(X_test, Y_test)
        print("\naccuracy on test set: %.2f%%" % (scores[1] * 100))

        attack_params = {"verbose": 0,
                         "clip_min": 0.,
                         "clip_max": 1.,
                         "eps": 1.}

        attack = FastGradientMethod(classifier, session)
        X_train_adv = attack.generate(X_train, **attack_params)
        X_test_adv = attack.generate(X_test, **attack_params)

        self.assertFalse((X_train == X_train_adv).all())
        self.assertFalse((X_test == X_test_adv).all())

        train_y_pred = get_labels_np_array(classifier.predict(X_train_adv))
        test_y_pred = get_labels_np_array(classifier.predict(X_test_adv))

        self.assertFalse((Y_train == train_y_pred).all())
        self.assertFalse((Y_test == test_y_pred).all())

        scores = classifier.evaluate(X_train_adv, Y_train)
        print('\naccuracy on adversarial train examples: %.2f%%' % (scores[1] * 100))

        scores = classifier.evaluate(X_test_adv, Y_test)
        print('\naccuracy on adversarial test examples: %.2f%%' % (scores[1] * 100))

        # test minimal perturbations
        attack_params = {"verbose": 0,
                         "clip_min": 0.,
                         "clip_max": 1.,
                         "minimal": True,
                         "eps_step": .1,
                         "eps_max": 1.}

        X_train_adv_min = attack.generate(X_train, **attack_params)
        X_test_adv_min = attack.generate(X_test, **attack_params)

        self.assertFalse((X_train_adv_min == X_train_adv).all())
        self.assertFalse((X_test_adv_min == X_test_adv).all())

        self.assertFalse((X_train == X_train_adv_min).all())
        self.assertFalse((X_test == X_test_adv_min).all())

        train_y_pred = get_labels_np_array(classifier.predict(X_train_adv_min))
        test_y_pred = get_labels_np_array(classifier.predict(X_test_adv_min))

        self.assertFalse((Y_train == train_y_pred).all())
        self.assertFalse((Y_test == test_y_pred).all())

        scores = classifier.evaluate(X_train_adv_min, Y_train)
        print('\naccuracy on adversarial train examples with minimal perturbation: %.2f%%' % (scores[1] * 100))

        scores = classifier.evaluate(X_test_adv_min, Y_test)
        print('\naccuracy on adversarial test examples with minimal perturbation: %.2f%%' % (scores[1] * 100))

    def test_with_preprocessing(self):

        session = tf.Session()
        k.set_session(session)

        comp_params = {"loss": 'categorical_crossentropy',
                       "optimizer": 'adam',
                       "metrics": ['accuracy']}

        # get MNIST
        batch_size, nb_train, nb_test = 100, 1000, 100
        (X_train, Y_train), (X_test, Y_test), _, _ = load_mnist()
        X_train, Y_train = X_train[:nb_train], Y_train[:nb_train]
        X_test, Y_test = X_test[:nb_test], Y_test[:nb_test]
        im_shape = X_train[0].shape

        # get classifier
        classifier = CNN(im_shape, act="relu", defences=["featsqueeze1"])
        classifier.compile(comp_params)
        classifier.fit(X_train, Y_train, epochs=1, batch_size=batch_size, verbose=0)
        scores = classifier.evaluate(X_train, Y_train)
        print("\naccuracy on training set: %.2f%%" % (scores[1] * 100))
        scores = classifier.evaluate(X_test, Y_test)
        print("\naccuracy on test set: %.2f%%" % (scores[1] * 100))

        attack_params = {"verbose": 0,
                         "clip_min": 0.,
                         "clip_max": 1.,
                         "eps": 1.}

        attack = FastGradientMethod(classifier, session)
        X_train_adv = attack.generate(X_train, **attack_params)
        X_test_adv = attack.generate(X_test, **attack_params)

        self.assertFalse((X_train == X_train_adv).all())
        self.assertFalse((X_test == X_test_adv).all())

        train_y_pred = get_labels_np_array(classifier.predict(X_train_adv))
        test_y_pred = get_labels_np_array(classifier.predict(X_test_adv))

        self.assertFalse((Y_train == train_y_pred).all())
        self.assertFalse((Y_test == test_y_pred).all())

        scores = classifier.evaluate(X_train_adv, Y_train)
        print('\naccuracy on adversarial train examples: %.2f%%' % (scores[1] * 100))

        scores = classifier.evaluate(X_test_adv, Y_test)
        print('\naccuracy on adversarial test examples: %.2f%%' % (scores[1] * 100))

if __name__ == '__main__':
    unittest.main()
