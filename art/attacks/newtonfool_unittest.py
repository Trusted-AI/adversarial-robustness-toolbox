from __future__ import absolute_import, division, print_function

import keras.backend as k
import tensorflow as tf
import unittest

from art.attacks.newtonfool import NewtonFool
from art.classifiers.cnn import CNN
from art.utils import load_mnist


class TestNewtonFool(unittest.TestCase):
    def test_mnist(self):
        session = tf.Session()
        k.set_session(session)

        comp_params = {"loss": 'categorical_crossentropy',
                       "optimizer": 'adam',
                       "metrics": ['accuracy']}

        # get MNIST
        batch_size, nb_train, nb_test = 100, 1000, 11
        (X_train, Y_train), (X_test, Y_test), _, _ = load_mnist()
        X_train, Y_train = X_train[:nb_train], Y_train[:nb_train]
        X_test, Y_test = X_test[:nb_test], Y_test[:nb_test]
        im_shape = X_train[0].shape

        # get classifier
        classifier = CNN(im_shape, act="relu")
        classifier.compile(comp_params)
        classifier.fit(X_train, Y_train, epochs=1, batch_size=batch_size, verbose=0)
        classifier.fit(X_train, Y_train, epochs=1, batch_size=batch_size, verbose=0)

        # Attack
        nf = NewtonFool(classifier, sess=session)
        nf.set_params(max_iter=20)
        x_test_adv = nf.generate(X_test)
        self.assertFalse((X_test == x_test_adv).all())

        y_pred = classifier.predict(X_test)
        y_pred_adv = classifier.predict(x_test_adv)
        y_pred_bool = y_pred.max(axis=1, keepdims=1) == y_pred
        y_pred_max = y_pred.max(axis=1)
        y_pred_adv_max = y_pred_adv[y_pred_bool]
        self.assertTrue((y_pred_max >= y_pred_adv_max).all())

        scores1 = classifier.evaluate(X_test, Y_test)
        print("\nAccuracy on test set: %.2f%%" % (scores1[1] * 100))
        scores2 = classifier.evaluate(x_test_adv, Y_test)
        print('\nAccuracy on adversarial examples: %.2f%%' % (scores2[1] * 100))
        self.assertTrue(scores1[1] != scores2[1])


if __name__ == '__main__':
    unittest.main()
