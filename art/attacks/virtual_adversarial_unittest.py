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

import keras.backend as k
import tensorflow as tf
import unittest

from art.attacks.virtual_adversarial import VirtualAdversarialMethod
from art.classifiers.cnn import CNN
from art.utils import load_mnist, get_labels_np_array


class TestVirtualAdversarial(unittest.TestCase):
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
        scores = classifier.evaluate(X_test, Y_test)
        print("\naccuracy on test set: %.2f%%" % (scores[1] * 100))

        df = VirtualAdversarialMethod(classifier, sess=session, clip_min=0., clip_max=1.)
        x_test_adv = df.generate(X_test, eps=1)
        self.assertFalse((X_test == x_test_adv).all())

        y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        self.assertFalse((Y_test == y_pred).all())

        scores = classifier.evaluate(x_test_adv, Y_test)
        print('\naccuracy on adversarial examples: %.2f%%' % (scores[1] * 100))

if __name__ == '__main__':
    unittest.main()
