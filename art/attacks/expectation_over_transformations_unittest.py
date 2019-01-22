from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

import keras
import keras.backend as k
import tensorflow as tf
import numpy as np
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential

from art.classifiers import KerasClassifier
from art.attacks import FastGradientMethod, ExpectationOverTransformations
from art.utils import load_mnist, random_targets, master_seed

logger = logging.getLogger('testLogger')

BATCH_SIZE, NB_TRAIN, NB_TEST = 100, 5000, 10


class TestExpectationOverTransformations(unittest.TestCase):
    """
    A unittest class for testing the Expectation over Transformations in attacks.
    """
    @classmethod
    def setUpClass(cls):
        # Get MNIST
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        x_train, y_train = x_train[:NB_TRAIN], y_train[:NB_TRAIN]
        x_test, y_test = x_test[:NB_TEST], y_test[:NB_TEST]
        cls.mnist = (x_train, y_train), (x_test, y_test)

    def setUp(self):
        # Set master seed
        master_seed(1234)

    def test_krclassifier(self):
        """
        Test with a KerasClassifier.
        :return:
        """
        # Initialize a tf session
        session = tf.Session()
        k.set_session(session)

        # Get MNIST
        (x_train, y_train), (x_test, y_test) = self.mnist

        # Create simple CNN
        model = Sequential()
        model.add(Conv2D(4, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(10, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.01),
                      metrics=['accuracy'])

        # Get classifier
        krc = KerasClassifier((0, 1), model, use_logits=False)
        krc.fit(x_train, y_train, batch_size=BATCH_SIZE, nb_epochs=10)

        # First attack (without EoT):
        fgsm = FastGradientMethod(classifier=krc, targeted=True)
        params = {'y': random_targets(y_test, krc.nb_classes)}
        x_test_adv = fgsm.generate(x_test, **params)

        # Second attack (with EoT):
        def t(x):
            return x

        def transformation():
            while True:
                yield t

        eot = ExpectationOverTransformations(sample_size=1, transformation=transformation)

        fgsm_with_eot = FastGradientMethod(classifier=krc,
                                           expectation=eot,
                                           targeted=True)
        self.assertFalse(fgsm_with_eot.expectation is None)
        x_test_adv_with_eot = fgsm_with_eot.generate(x_test, **params)

        self.assertTrue((np.abs(x_test_adv - x_test_adv_with_eot) < 0.001).all())


if __name__ == '__main__':
    unittest.main()
