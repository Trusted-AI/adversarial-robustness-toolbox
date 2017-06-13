from config import config_dict

import unittest

import os.path
import shutil

import keras
from keras.optimizers import RMSprop

import tensorflow as tf

from src.classifiers import siamese
from src.utils import load_cifar10, load_mnist, create_class_pairs

class TestResNetModel(unittest.TestCase):

    # def test_cifar(self):
    #
    #     BATCH_SIZE = 10
    #     NB_TRAIN = 1000
    #     NB_TEST = 100
    #
    #     session = tf.Session()
    #     keras.backend.set_session(session)
    #
    #     # get CIFAR10
    #     (X_train, Y_train), (X_test, Y_test) = load_cifar10()
    #     X_train, Y_train, X_test, Y_test = X_train[:NB_TRAIN], Y_train[:NB_TRAIN], X_test[:NB_TEST], Y_test[:NB_TEST]
    #
    #     im_shape = X_train[0].shape
    #
    #     model = resnet.resnet_model(input_shape=im_shape)
    #
    #     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #
    #     # Fit the model
    #     model.fit(X_train, Y_train, epochs=2, batch_size=BATCH_SIZE)
    #
    #     scores = model.evaluate(X_test, Y_test)
    #
    #     print("\naccuracy: %.2f%%" % (scores[1] * 100))


    def test_mnist(self):

        BATCH_SIZE = 10
        NB_TRAIN = 100
        NB_TEST = 100

        session = tf.Session()
        keras.backend.set_session(session)

        # get MNIST
        (X_train, Y_train), (X_test, Y_test) = load_mnist()
        X_train, Y_train, X_test, Y_test = X_train[:NB_TRAIN], Y_train[:NB_TRAIN], X_test[:NB_TEST], Y_test[:NB_TEST]

        # get learning pairs
        tr_pairs, tr_y = create_class_pairs(X_train, Y_train)
        te_pairs, te_y = create_class_pairs(X_test, Y_test)

        im_shape = X_train[0].shape

        model = siamese.siamese_model(input_shape=im_shape)

        # train
        rms = RMSprop()
        model.compile(loss=siamese.contrastive_loss, optimizer=rms, metrics=['accuracy'])
        model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y, batch_size=BATCH_SIZE, epochs=2)

        scores = model.evaluate([te_pairs[:, 0], te_pairs[:, 1]], te_y)
        print("\n", scores, type(scores))
        print("\naccuracy: %.2f%%" % (scores[1] * 100))

if __name__ == '__main__':
    unittest.main()
