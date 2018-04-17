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

import keras.backend as k
import tensorflow as tf
import unittest

from art.attacks.saliency_map import SaliencyMapMethod
from art.classifiers.cnn import CNN
from art.utils import load_mnist, get_labels_np_array

# TODO add test with gamma < 1


class TestSaliencyMap(unittest.TestCase):
    def test_mnist_untargeted(self):
        session = tf.Session()
        k.set_session(session)

        comp_params = {"loss": 'categorical_crossentropy',
                       "optimizer": 'adam',
                       "metrics": ['accuracy']}

        # Get MNIST
        batch_size, nb_train, nb_test = 100, 1000, 10
        (X_train, Y_train), (X_test, Y_test), _, _ = load_mnist()
        X_train, Y_train = X_train[:nb_train], Y_train[:nb_train]
        X_test, Y_test = X_test[:nb_test], Y_test[:nb_test]
        im_shape = X_train[0].shape

        # Get classifier
        classifier = CNN(im_shape, act="relu")
        classifier.compile(comp_params)
        classifier.fit(X_train, Y_train, epochs=1, batch_size=batch_size, verbose=0)
        scores = classifier.evaluate(X_test, Y_test)
        print("\naccuracy on test set: %.2f%%" % (scores[1] * 100))

        # Perform attack
        df = SaliencyMapMethod(classifier, sess=session)
        df.set_params(clip_min=0, clip_max=1, theta=1)
        x_test_adv = df.generate(X_test)
        self.assertFalse((X_test == x_test_adv).all())

        y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        self.assertFalse((Y_test == y_pred).all())

        scores = classifier.evaluate(x_test_adv, Y_test)
        print('\naccuracy on adversarial examples: %.2f%%' % (scores[1] * 100))

    def test_mnist_targeted(self):
        session = tf.Session()
        k.set_session(session)

        comp_params = {"loss": 'categorical_crossentropy',
                       "optimizer": 'adam',
                       "metrics": ['accuracy']}

        # Get MNIST
        batch_size, nb_train, nb_test = 100, 1000, 10
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        x_train, y_train = x_train[:nb_train], y_train[:nb_train]
        x_test, y_test = x_test[:nb_test], y_test[:nb_test]
        im_shape = x_train[0].shape

        # Get classifier
        classifier = CNN(im_shape, act="relu")
        classifier.compile(comp_params)
        classifier.fit(x_train, y_train, epochs=1, batch_size=batch_size, verbose=0)
        scores = classifier.evaluate(x_test, y_test)
        print("\naccuracy on test set: %.2f%%" % (scores[1] * 100))

        # Generate random target classes
        import numpy as np
        nb_classes = np.unique(np.argmax(y_test, axis=1)).shape[0]
        targets = np.random.randint(nb_classes, size=nb_test)
        while (targets == np.argmax(y_test, axis=1)).any():
            targets = np.random.randint(nb_classes, size=nb_test)

        # Perform attack
        df = SaliencyMapMethod(classifier, sess=session, clip_min=0, clip_max=1, theta=1)
        x_test_adv = df.generate(x_test, y_val=targets)
        self.assertFalse((x_test == x_test_adv).all())

        y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        self.assertFalse((y_test == y_pred).all())

        scores = classifier.evaluate(x_test_adv, y_test)
        print('\naccuracy on adversarial examples: %.2f%%' % (scores[1] * 100))

if __name__ == '__main__':
    unittest.main()
