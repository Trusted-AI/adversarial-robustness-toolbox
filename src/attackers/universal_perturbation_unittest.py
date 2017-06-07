import unittest

import keras.backend as k
import tensorflow as tf

from src.attackers.universal_perturbation import UniversalPerturbation
from src.classifiers import cnn
from src.utils import load_mnist, get_label_conf


class TestUniversalPerturbation(unittest.TestCase):

    def test_mnist(self):
        session = tf.Session()
        k.set_session(session)

        # get MNIST
        batch_size, nb_train, nb_test = 100, 100, 10
        (X_train, Y_train), (X_test, Y_test) = load_mnist()
        X_train, Y_train = X_train[:nb_train], Y_train[:nb_train]
        X_test, Y_test = X_test[:nb_test], Y_test[:nb_test]
        im_shape = X_train[0].shape

        # get classifier
        model = cnn.cnn_model(im_shape, act="relu")
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, Y_train, epochs=1, batch_size=batch_size, verbose=0)
        scores = model.evaluate(X_test, Y_test)
        print("\naccuracy on test set: %.2f%%" % (scores[1] * 100))

        attack_params = {"verbose": 0,
                         "clip_min": 0.,
                         "clip_max": 1.}

        attack = UniversalPerturbation(model, session)
        x_train_adv = attack.generate(X_train, "deepfool", attack_params)
        self.assertTrue((attack.fooling_rate >= 0.2) or attack.converged)

        x_test_adv = X_test + attack.v

        self.assertFalse((X_test == x_test_adv).all())

        _, train_y_pred = get_label_conf(model.predict(x_train_adv))
        _, test_y_pred = get_label_conf(model.predict(x_test_adv))

        self.assertFalse((Y_test == test_y_pred).all())
        self.assertFalse((X_train == train_y_pred).all())

        scores = model.evaluate(x_train_adv, Y_train)
        print('\naccuracy on adversarial train examples: %.2f%%' % (scores[1] * 100))

        scores = model.evaluate(x_test_adv, Y_test)
        print('\naccuracy on adversarial test examples: %.2f%%' % (scores[1] * 100))

if __name__ == '__main__':
    unittest.main()
