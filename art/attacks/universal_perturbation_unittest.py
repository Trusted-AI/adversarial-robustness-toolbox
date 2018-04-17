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
from __future__ import absolute_import, division, print_function

import unittest

import keras.backend as k
import tensorflow as tf

from art.attacks.universal_perturbation import UniversalPerturbation
from art.classifiers.cnn import CNN
from art.utils import load_mnist, get_labels_np_array


class TestUniversalPerturbation(unittest.TestCase):
    def test_mnist(self):
        session = tf.Session()
        k.set_session(session)

        comp_params = {"loss": 'categorical_crossentropy',
                       "optimizer": 'adam',
                       "metrics": ['accuracy']}

        # get MNIST
        batch_size, nb_train, nb_test = 10, 10, 10
        (X_train, Y_train), (X_test, Y_test), _, _ = load_mnist()
        X_train, Y_train = X_train[:nb_train], Y_train[:nb_train]
        X_test, Y_test = X_test[:nb_test], Y_test[:nb_test]
        im_shape = X_train[0].shape

        # get classifier
        classifier = CNN(im_shape, act="relu")
        classifier.compile(comp_params)
        classifier.fit(X_train, Y_train, epochs=1, batch_size=batch_size, verbose=0)
        scores = classifier.evaluate(X_test, Y_test)
        print("\naccuracy on test set: %.2f%%" % (scores[1] * 100))

        attack_params = {"verbose": 2,
                         "clip_min": 0.,
                         "clip_max": 1,
                         "attacker": "deepfool"}

        attack = UniversalPerturbation(classifier, session)
        x_train_adv = attack.generate(X_train, **attack_params)
        self.assertTrue((attack.fooling_rate >= 0.2) or not attack.converged)

        x_test_adv = X_test + attack.v
        self.assertFalse((X_test == x_test_adv).all())

        train_y_pred = get_labels_np_array(classifier.predict(x_train_adv))
        test_y_pred = get_labels_np_array(classifier.predict(x_test_adv))

        self.assertFalse((Y_test == test_y_pred).all())
        self.assertFalse((Y_train == train_y_pred).all())

        scores = classifier.evaluate(x_train_adv, Y_train)
        print('\naccuracy on adversarial train examples: %.2f%%' % (scores[1] * 100))

        scores = classifier.evaluate(x_test_adv, Y_test)
        print('\naccuracy on adversarial test examples: %.2f%%' % (scores[1] * 100))

if __name__ == '__main__':
    unittest.main()
