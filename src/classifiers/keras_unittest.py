from __future__ import absolute_import, division, print_function, unicode_literals

import keras
import keras.backend as k
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout
import numpy as np
import unittest
import shutil

from src.classifiers.keras import KerasClassifier
from src.utils import load_cifar10, load_mnist, make_directory

BATCH_SIZE = 10
NB_TRAIN = 1000
NB_TEST = 100


class TestKerasClassifier(unittest.TestCase):

    def setUp(self):
        k.set_learning_phase(1)

        make_directory("./tests/")

        # Get MNIST
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        x_train, y_train, x_test, y_test = x_train[:NB_TRAIN], y_train[:NB_TRAIN], x_test[:NB_TEST], y_test[:NB_TEST]
        self.mnist = ((x_train, y_train), (x_test, y_test))
        im_shape = x_train[0].shape

        # Create basic CNN on MNIST; architecture from Keras examples
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=im_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

        model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=1)
        self.model_mnist = model

    def tearDown(self):
        shutil.rmtree("./tests/")

    # def test_logits(self):
    #     classifier = KerasClassifier((0, 1), self.model_mnist, use_logits=True)

    # def test_probabilities(self):
    #     classifier = KerasClassifier((0, 1), self.model_mnist, use_logits=False)

    def test_fit(self):
        labels = np.argmax(self.mnist[1][1], axis=1)
        classifier = KerasClassifier((0, 1), self.model_mnist, use_logits=False)
        scores = np.sum(np.argmax(classifier.predict(self.mnist[1][0]), axis=1) == labels) / NB_TEST
        # print(np.argmax(classifier.predict(self.mnist[1][0]), axis=1))
        print("\naccuracy: %.2f%%" % (scores * 100))

        classifier.fit(self.mnist[0][0], self.mnist[0][1], batch_size=BATCH_SIZE, nb_epochs=1)
        scores2 = np.sum(np.argmax(classifier.predict(self.mnist[1][0]), axis=1) == labels) / NB_TEST
        # print(np.argmax(classifier.predict(self.mnist[1][0]), axis=1))
        print("\naccuracy: %.2f%%" % (scores2 * 100))

        self.assertTrue(scores2 >= scores)
