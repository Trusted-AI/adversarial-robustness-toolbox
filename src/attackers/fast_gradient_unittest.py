import unittest

import keras.backend as k
import numpy as np
import tensorflow as tf

from src.attackers.fast_gradient import FastGradientMethod
from src.classifiers import cnn
from src.utils import load_mnist, get_labels_np_array


class TestFastGradientMethod(unittest.TestCase):

    def test_mnist(self):
        session = tf.Session()
        k.set_session(session)

        # get MNIST
        batch_size, nb_train, nb_test = 100, 1000, 100
        (X_train, Y_train), (X_test, Y_test) = load_mnist()
        X_train, Y_train = X_train[:nb_train], Y_train[:nb_train]
        X_test, Y_test = X_test[:nb_test], Y_test[:nb_test]
        im_shape = X_train[0].shape

        # get classifier
        model = cnn.cnn_model(im_shape, act="relu")
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, Y_train, epochs=1, batch_size=batch_size, verbose=0)
        scores = model.evaluate(X_train, Y_train)
        print("\naccuracy on training set: %.2f%%" % (scores[1] * 100))
        scores = model.evaluate(X_test, Y_test)
        print("\naccuracy on test set: %.2f%%" % (scores[1] * 100))

        attack_params = {"verbose": 0,
                         "clip_min": 0.,
                         "clip_max": 1.,
                         "eps": 1.}

        attack = FastGradientMethod(model, session)
        X_train_adv = attack.generate(X_train, **attack_params)
        X_test_adv = attack.generate(X_test, **attack_params)

        self.assertFalse((X_train == X_train_adv).all())
        self.assertFalse((X_test == X_test_adv).all())

        train_y_pred = get_labels_np_array(model.predict(X_train_adv))
        test_y_pred = get_labels_np_array(model.predict(X_test_adv))

        self.assertFalse((Y_train == train_y_pred).all())
        self.assertFalse((Y_test == test_y_pred).all())

        scores = model.evaluate(X_train_adv, Y_train)
        print('\naccuracy on adversarial train examples: %.2f%%' % (scores[1] * 100))

        scores = model.evaluate(X_test_adv, Y_test)
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

        train_y_pred = get_labels_np_array(model.predict(X_train_adv_min))
        test_y_pred = get_labels_np_array(model.predict(X_test_adv_min))

        self.assertFalse((Y_train == train_y_pred).all())
        self.assertFalse((Y_test == test_y_pred).all())

        scores = model.evaluate(X_train_adv_min, Y_train)
        print('\naccuracy on adversarial train examples: %.2f%%' % (scores[1] * 100))

        scores = model.evaluate(X_test_adv_min, Y_test)
        print('\naccuracy on adversarial test examples: %.2f%%' % (scores[1] * 100))


if __name__ == '__main__':
    unittest.main()
