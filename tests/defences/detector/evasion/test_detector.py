# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
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

import logging
import unittest

import keras
import keras.backend as k
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
import numpy as np

from art.attacks.evasion.fast_gradient import FastGradientMethod
from art.estimators.classification.keras import KerasClassifier
from art.defences.detector.evasion import BinaryInputDetector, BinaryActivationDetector
from art.utils import load_mnist

from tests.utils import master_seed, get_image_classifier_kr

logger = logging.getLogger(__name__)

BATCH_SIZE, NB_TRAIN, NB_TEST = 100, 1000, 10


class TestBinaryInputDetector(unittest.TestCase):
    """
    A unittest class for testing the binary input detector.
    """

    def setUp(self):
        master_seed(seed=1234)

    def tearDown(self):
        k.clear_session()

    def test_binary_input_detector(self):
        """
        Test the binary input detector end-to-end.
        :return:
        """
        # Get MNIST
        nb_train, nb_test = 1000, 10
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        x_train, y_train = x_train[:NB_TRAIN], y_train[:NB_TRAIN]
        x_test, y_test = x_test[:NB_TEST], y_test[:NB_TEST]

        # Keras classifier
        classifier = get_image_classifier_kr()

        # Generate adversarial samples:
        attacker = FastGradientMethod(classifier, eps=0.1)
        x_train_adv = attacker.generate(x_train[:nb_train])
        x_test_adv = attacker.generate(x_test[:nb_test])

        # Compile training data for detector:
        x_train_detector = np.concatenate((x_train[:nb_train], x_train_adv), axis=0)
        y_train_detector = np.concatenate((np.array([[1, 0]] * nb_train), np.array([[0, 1]] * nb_train)), axis=0)

        # Create a simple CNN for the detector
        input_shape = x_train.shape[1:]
        try:
            from keras.optimizers import Adam

            optimizer = Adam(lr=0.01)
        except ImportError:
            from keras.optimizers import adam_v2

            optimizer = adam_v2.Adam(lr=0.01)
        model = Sequential()
        model.add(Conv2D(4, kernel_size=(5, 5), activation="relu", input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(2, activation="softmax"))
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=["accuracy"])

        # Create detector and train it:
        detector = BinaryInputDetector(KerasClassifier(model=model, clip_values=(0, 1), use_logits=False))
        detector.fit(x_train_detector, y_train_detector, nb_epochs=2, batch_size=128)

        # Apply detector on clean and adversarial test data:
        test_detection = np.argmax(detector.predict(x_test), axis=1)
        test_adv_detection = np.argmax(detector.predict(x_test_adv), axis=1)

        # Assert there is at least one true positive and negative:
        nb_true_positives = len(np.where(test_adv_detection == 1)[0])
        nb_true_negatives = len(np.where(test_detection == 0)[0])
        logger.debug("Number of true positives detected: %i", nb_true_positives)
        logger.debug("Number of true negatives detected: %i", nb_true_negatives)
        self.assertGreater(nb_true_positives, 0)
        self.assertGreater(nb_true_negatives, 0)


class TestBinaryActivationDetector(unittest.TestCase):
    """
    A unittest class for testing the binary activation detector.
    """

    def setUp(self):
        master_seed(seed=1234)

    def tearDown(self):
        k.clear_session()

    def test_binary_activation_detector(self):
        """
        Test the binary activation detector end-to-end.
        :return:
        """
        # Get MNIST
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        x_train, y_train = x_train[:NB_TRAIN], y_train[:NB_TRAIN]
        x_test, y_test = x_test[:NB_TEST], y_test[:NB_TEST]

        # Keras classifier
        classifier = get_image_classifier_kr()

        # Generate adversarial samples:
        attacker = FastGradientMethod(classifier, eps=0.1)
        x_train_adv = attacker.generate(x_train[:NB_TRAIN])
        x_test_adv = attacker.generate(x_test[:NB_TRAIN])

        # Compile training data for detector:
        x_train_detector = np.concatenate((x_train[:NB_TRAIN], x_train_adv), axis=0)
        y_train_detector = np.concatenate((np.array([[1, 0]] * NB_TRAIN), np.array([[0, 1]] * NB_TRAIN)), axis=0)

        # Create a simple CNN for the detector
        activation_shape = classifier.get_activations(x_test[:1], 0, batch_size=128).shape[1:]
        number_outputs = 2
        try:
            from keras.optimizers import Adam

            optimizer = Adam(lr=0.01)
        except ImportError:
            from keras.optimizers import adam_v2

            optimizer = adam_v2.Adam(lr=0.01)
        model = Sequential()
        model.add(MaxPooling2D(pool_size=(2, 2), input_shape=activation_shape))
        model.add(Flatten())
        model.add(Dense(number_outputs, activation="softmax"))
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=["accuracy"])

        # Create detector and train it.
        # Detector consider activations at layer=0:
        detector = BinaryActivationDetector(
            classifier=classifier, detector=KerasClassifier(model=model, clip_values=(0, 1), use_logits=False), layer=0
        )
        detector.fit(x_train_detector, y_train_detector, nb_epochs=2, batch_size=128)

        # Apply detector on clean and adversarial test data:
        test_detection = np.argmax(detector.predict(x_test), axis=1)
        test_adv_detection = np.argmax(detector.predict(x_test_adv), axis=1)

        # Assert there is at least one true positive and negative
        nb_true_positives = len(np.where(test_adv_detection == 1)[0])
        nb_true_negatives = len(np.where(test_detection == 0)[0])
        logger.debug("Number of true positives detected: %i", nb_true_positives)
        logger.debug("Number of true negatives detected: %i", nb_true_negatives)
        self.assertGreater(nb_true_positives, 0)
        self.assertGreater(nb_true_negatives, 0)


if __name__ == "__main__":
    unittest.main()
