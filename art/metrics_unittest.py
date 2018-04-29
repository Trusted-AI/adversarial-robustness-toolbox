from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import keras
import keras.backend as k
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import numpy as np

from art.classifiers import KerasClassifier
from art.metrics import empirical_robustness, clever_t, clever_u, loss_sensitivity
from art.utils import load_mnist

BATCH_SIZE = 10
NB_TRAIN = 100
NB_TEST = 100


class TestMetrics(unittest.TestCase):
    def test_emp_robustness_mnist(self):
        # Get MNIST
        (x_train, y_train), (_, _), _, _ = load_mnist()
        x_train, y_train = x_train[:NB_TRAIN], y_train[:NB_TRAIN]

        # Get classifier
        classifier = self._cnn_mnist_k([28, 28, 1])
        classifier.fit(x_train, y_train, batch_size=BATCH_SIZE, nb_epochs=2)

        # Compute minimal perturbations
        params = {"eps_step": 1.1,
                  "clip_min": 0.,
                  "clip_max": 1.}

        emp_robust = empirical_robustness(classifier, x_train, str('fgsm'), params)
        self.assertEqual(emp_robust, 0.)

        params = {"eps_step": 1.,
                  "eps_max": 1.,
                  "clip_min": None,
                  "clip_max": None}
        emp_robust = empirical_robustness(classifier, x_train, str('fgsm'), params)
        self.assertAlmostEqual(emp_robust, 1., 3)

        params = {"eps_step": 0.1,
                  "eps_max": 0.2,
                  "clip_min": None,
                  "clip_max": None}
        emp_robust = empirical_robustness(classifier, x_train, str('fgsm'), params)
        self.assertLessEqual(emp_robust, 0.21)

    def test_loss_sensitivity(self):
        # Get MNIST
        (x_train, y_train), (_, _), _, _ = load_mnist()
        x_train, y_train = x_train[:NB_TRAIN], y_train[:NB_TRAIN]

        # Get classifier
        classifier = self._cnn_mnist_k([28, 28, 1])
        classifier.fit(x_train, y_train, batch_size=BATCH_SIZE, nb_epochs=2)

        l = loss_sensitivity(classifier, x_train)
        self.assertGreaterEqual(l, 0)

    # def testNearestNeighborDist(self):
    #     # Get MNIST
    #     (x_train, y_train), (_, _), _, _ = load_mnist()
    #     x_train, y_train = x_train[:NB_TRAIN], y_train[:NB_TRAIN]
    #
    #     # Get classifier
    #     classifier = self._cnn_mnist_k([28, 28, 1])
    #     classifier.fit(x_train, y_train, batch_size=BATCH_SIZE, nb_epochs=2)
    #
    #     dist = nearest_neighbour_dist(classifier, x_train, x_train, str('fgsm'))
    #     self.assertGreaterEqual(dist, 0)

    @staticmethod
    def _cnn_mnist_k(input_shape):
        # Create simple CNN
        model = Sequential()
        model.add(Conv2D(4, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(10, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.01),
                      metrics=['accuracy'])

        classifier = KerasClassifier((0, 1), model, use_logits=False)
        return classifier


#########################################
# This part is the unit test for Clever.#
#########################################

class TestClassifier(Classifier):
    def __init__(self, defences=None, preproc=None):
        from keras.layers import Lambda
        model = Sequential(name="TestClassifier")
        model.add(Lambda(lambda x: x + 0, input_shape=(2,)))

        super(TestClassifier, self).__init__(model, defences, preproc)


class TestClever(unittest.TestCase):
    """
    Unittest for Clever metrics.
    """
    def test_clever_t_unit(self):
        """
        Test the targeted version with simplified data.
        :return:
        """
        print("Unit test for the targeted version with simplified data.")
        # Define session & params
        session = tf.Session()
        k.set_session(session)

        # Get classifier
        classifier = TestClassifier()

        # Compute scores
        res = clever_t(np.array([1, 0]), classifier, 1, 20, 10, 1, session)

        # Test
        self.assertAlmostEqual(res[0], 0.9999999999999998, delta=0.00001)
        self.assertAlmostEqual(res[1], 0.7071067811865474, delta=0.00001)
        self.assertAlmostEqual(res[2], 0.4999999999999999, delta=0.00001)

    def test_clever_u_unit(self):
        """
        Test the untargeted version with simplified data.
        :return:
        """
        print("Unit test for the untargeted version with simplified data.")
        # Define session & params
        session = tf.Session()
        k.set_session(session)

        # Get classifier
        classifier = TestClassifier()

        # Compute scores
        res = clever_u(np.array([1, 0]), classifier, 20, 10, 1, session)

        # Test
        self.assertAlmostEqual(res[0], 0.9999999999999998, delta=0.00001)
        self.assertAlmostEqual(res[1], 0.7071067811865474, delta=0.00001)
        self.assertAlmostEqual(res[2], 0.4999999999999999, delta=0.00001)

    def test_clever_t(self):
        """
        Test the targeted version.
        :return:
        """
        print("Test if the targeted version works on a true classifier/data")
        # Define session & params
        session = tf.Session()
        k.set_session(session)

        comp_params = {"loss": 'categorical_crossentropy', "optimizer": 'adam',
                       "metrics": ['accuracy']}

        # Get MNIST
        (X_train, Y_train), (_, _), _, _ = load_mnist()
        X_train, Y_train = X_train[:NB_TRAIN], Y_train[:NB_TRAIN]
        im_shape = X_train[0].shape

        # Get classifier
        classifier = CNN(im_shape, act="relu")
        classifier.compile(comp_params)
        classifier.fit(X_train, Y_train, epochs=1,
                       batch_size=BATCH_SIZE, verbose=0)

        res = clever_t(X_train[-1], classifier, 7, 20, 10, 5, session)
        self.assertGreater(res[0], res[1])
        self.assertGreater(res[1], res[2])

    def test_clever_u(self):
        """
        Test the untargeted version.
        :return:
        """
        print("Test if the untargeted version works on a true classifier/data")
        # Define session & params
        session = tf.Session()
        k.set_session(session)

        comp_params = {"loss": 'categorical_crossentropy', "optimizer": 'adam',
                       "metrics": ['accuracy']}

        # Get MNIST
        (X_train, Y_train), (_, _), _, _ = load_mnist()
        X_train, Y_train = X_train[:NB_TRAIN], Y_train[:NB_TRAIN]
        im_shape = X_train[0].shape

        # Get classifier
        classifier = CNN(im_shape, act="relu")
        classifier.compile(comp_params)
        classifier.fit(X_train, Y_train, epochs=1,
                       batch_size=BATCH_SIZE, verbose=0)

        res = clever_u(X_train[-1], classifier, 2, 10, 5, session)
        self.assertGreater(res[0], res[1])
        self.assertGreater(res[1], res[2])


if __name__ == '__main__':
    unittest.main()
