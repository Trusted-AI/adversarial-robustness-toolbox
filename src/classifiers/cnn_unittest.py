import unittest

import keras

import tensorflow as tf

from src.classifiers import cnn
from src.utils import load_cifar10,load_mnist

class TestCNNModel(unittest.TestCase):

    def test_cifar(self):

        BATCH_SIZE = 10
        NB_CLASSES = 10
        NB_TRAIN = 1000
        NB_TEST = 100

        session = tf.Session()
        keras.backend.set_session(session)

        # get CIFAR10
        (X_train, Y_train), (X_test, Y_test) = load_cifar10()
        X_train, Y_train, X_test, Y_test = X_train[:NB_TRAIN], Y_train[:NB_TRAIN], X_test[:NB_TEST], Y_test[:NB_TEST]

        im_shape = X_train[0].shape

        # x = tf.placeholder(tf.float32, shape=(None,im_shape[0],im_shape[1],im_shape[2]))
        # y = tf.placeholder(tf.float32, shape=(None,NB_CLASSES))

        model = cnn.cnn_model(im_shape,act="brelu")

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Fit the model
        model.fit(X_train, Y_train, epochs=1, batch_size=BATCH_SIZE)

        scores = model.evaluate(X_test, Y_test)

        print("\naccuracy: %.2f%%" % (scores[1] * 100))


    def test_mnist(self):

        BATCH_SIZE = 10
        NB_TRAIN = 1000
        NB_TEST = 100

        session = tf.Session()
        keras.backend.set_session(session)

        # get MNIST
        (X_train, Y_train), (X_test, Y_test) = load_mnist()
        X_train, Y_train, X_test, Y_test = X_train[:NB_TRAIN], Y_train[:NB_TRAIN], X_test[:NB_TEST], Y_test[:NB_TEST]

        im_shape = X_train[0].shape

        # x = tf.placeholder(tf.float32, shape=(None,im_shape[0],im_shape[1],im_shape[2]))
        # y = tf.placeholder(tf.float32, shape=(None,NB_CLASSES))

        model = cnn.cnn_model(im_shape,act="relu")

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Fit the model
        model.fit(X_train,Y_train,epochs=1,batch_size=BATCH_SIZE)

        scores = model.evaluate(X_test,Y_test)

        print("\naccuracy: %.2f%%" % (scores[1] * 100))

if __name__ == '__main__':
    unittest.main()
