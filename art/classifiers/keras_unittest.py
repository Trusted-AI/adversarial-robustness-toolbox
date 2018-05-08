from __future__ import absolute_import, division, print_function, unicode_literals

import keras
import keras.backend as k
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout
import numpy as np
import unittest

from art.classifiers import KerasClassifier
from art.utils import load_mnist

BATCH_SIZE = 10
NB_TRAIN = 500
NB_TEST = 100


class TestKerasClassifier(unittest.TestCase):

    def setUp(self):
        k.set_learning_phase(1)

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

    # def test_logits(self):
    #     classifier = KerasClassifier((0, 1), self.model_mnist, use_logits=True)

    # def test_probabilities(self):
    #     classifier = KerasClassifier((0, 1), self.model_mnist, use_logits=False)

    def test_fit(self):
        labels = np.argmax(self.mnist[1][1], axis=1)
        classifier = KerasClassifier((0, 1), self.model_mnist, use_logits=False)
        acc = np.sum(np.argmax(classifier.predict(self.mnist[1][0]), axis=1) == labels) / NB_TEST
        print("\nAccuracy: %.2f%%" % (acc * 100))

        classifier.fit(self.mnist[0][0], self.mnist[0][1], batch_size=BATCH_SIZE, nb_epochs=1)
        acc2 = np.sum(np.argmax(classifier.predict(self.mnist[1][0]), axis=1) == labels) / NB_TEST
        print("\nAccuracy: %.2f%%" % (acc2 * 100))

        self.assertTrue(acc2 >= acc)

    def test_shapes(self):
        x_test, y_test = self.mnist[1]
        classifier = KerasClassifier((0, 1), self.model_mnist)

        preds = classifier.predict(self.mnist[1][0])
        self.assertTrue(preds.shape == y_test.shape)

        self.assertTrue(classifier.nb_classes == 10)

        class_grads = classifier.class_gradient(x_test[:11])
        self.assertTrue(class_grads.shape == tuple([11, 10] + list(x_test[1].shape)))

        loss_grads = classifier.loss_gradient(x_test[:11], y_test[:11])
        self.assertTrue(loss_grads.shape == x_test[:11].shape)
