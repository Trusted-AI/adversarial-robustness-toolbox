from __future__ import absolute_import, division, print_function, unicode_literals

import unittest
import numpy as np
import os
import pickle

import keras.backend as k
import tensorflow as tf
import keras
# import torch.nn as nn
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

from art.classifiers.classifier import Classifier
from art.classifiers.keras import KerasClassifier
from art.metrics import clever_u, clever_t, clever

from tensorflow.examples.tutorials.mnist import input_data
from art.utils import load_mnist
session = tf.Session()
k.set_session(session)

mnist = input_data.read_data_sets('mnist_data', one_hot=True)

def get_trained_mnist(nb_epochs=2):
    """
    Get a simple trained MNIST model to run tests on
    """
    # Initialize a tf session
    session = tf.Session()
    k.set_session(session)

    # Get MNIST
    batch_size, nb_train, nb_test = 100, 500, 5
    (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
    x_train, y_train = x_train[:nb_train], y_train[:nb_train]
    x_test, y_test = x_test[:nb_test], y_test[:nb_test]

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
    krc.fit(x_train, y_train, batch_size=batch_size, nb_epochs=nb_epochs)
    return krc

class TestCleverL2(unittest.TestCase):
    # def __init__(self, *args, **kwargs):
    #     (x_train, y_train), (self.x_test, self.y_test), _, _ = load_mnist()
    #     self.model = get_trained_mnist()
    def test_mnist_building(self):
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        model = get_trained_mnist(2)
        y_pred = model.predict(x_test)

        accuracy = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
        print("Testing mnist accuracy",accuracy)
    def test_clever_l2_no_target(self):
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        model = get_trained_mnist()
        scores = clever(model, x_test[0], 5, 5, 3, 2, target=None, c_init=1, pool_factor=10)
        print("Clever Scores for n-1 classes",scores,scores.shape)
        assert(scores.shape == (model.nb_classes-1,))
    def test_clever_l2_no_target_sorted(self):
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        model = get_trained_mnist()
        scores = clever(model, x_test[0], 5, 5, 3, 2, target=None, target_sort=True, c_init=1, pool_factor=10)
        print("Clever Scores for n-1 classes",scores,scores.shape)
        # Should approx. be in decreasing value
        assert(scores.shape == (model.nb_classes-1,))
    def test_clever_l2_same_target(self):
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        model = get_trained_mnist()
        scores = clever(model, x_test[0], 5, 5, 3, 2, target=np.argmax(model.predict(x_test[:1])), c_init=1, pool_factor=10)
        print("Clever Scores for the predicted class should be None",scores)
        assert(scores[0] is None)
    def test_clever_t_l2(self):
        # clever_t(classifier, x, target_class, n_b, n_s, r, norm, c_init=1, pool_factor=10):
        pass
    def test_clever_u_l2(self):
        #  clever_u(classifier, x, n_b, n_s, r, norm, c_init=1, pool_factor=10):
        pass

if __name__ == '__main__':
    unittest.main()