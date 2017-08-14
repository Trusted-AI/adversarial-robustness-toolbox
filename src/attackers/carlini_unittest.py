import keras.backend as k
import numpy as np
import tensorflow as tf
import unittest

from config import config_dict
from cleverhans.model import CallableModelWrapper

from src.attackers.carlini import CarliniL2Method
from src.classifiers.cnn import CNN
from src.utils import load_mnist, get_labels_np_array, random_targets


class TestCarliniL2(unittest.TestCase):
    def test_mnist(self):
        session = tf.Session()
        k.set_session(session)

        comp_params = {"loss": 'categorical_crossentropy',
                       "optimizer": 'adam',
                       "metrics": ['accuracy']}

        # get MNIST
        batch_size, nb_train, nb_test = 100, 1000, 10
        (X_train, Y_train), (X_test, Y_test) = load_mnist()
        X_train, Y_train = X_train[:nb_train], Y_train[:nb_train]
        X_test, Y_test = X_test[:nb_test], Y_test[:nb_test]
        im_shape = X_train[0].shape

        # get classifier
        classifier = CNN(im_shape, act="relu")
        classifier.compile(comp_params)
        classifier.fit(X_train, Y_train, epochs=1, batch_size=batch_size, verbose=0)
        scores = classifier.evaluate(X_test, Y_test)
        print("\naccuracy on test set: %.2f%%" % (scores[1] * 100))

        df = CarliniL2Method(classifier, sess=session, targeted=True, max_iterations=100, binary_search_steps=2,
                             learning_rate=1e-2, initial_const=1)
        params = {'y_val': random_targets(Y_test, classifier.model.get_output_shape_at(-1)[-1])}
        x_test_adv = df.generate(X_test, **params)
        self.assertFalse((X_test == x_test_adv).all())

        y_pred = get_labels_np_array(classifier.predict(x_test_adv))
        # self.assertFalse((Y_test == y_pred).all())

        scores = classifier.evaluate(x_test_adv, Y_test)
        print('\naccuracy on adversarial examples: %.2f%%' % (scores[1] * 100))

if __name__ == '__main__':
    unittest.main()
