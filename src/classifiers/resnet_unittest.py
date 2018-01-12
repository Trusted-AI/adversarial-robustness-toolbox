from config import config_dict
import unittest

import keras
import os.path
import tensorflow as tf
import shutil

from src.classifiers.resnet import ResNet
from src.classifiers.utils import save_classifier, load_classifier
from src.utils import load_cifar10, load_mnist, make_directory

BATCH_SIZE = 10
NB_TRAIN = 1000
NB_TEST = 100


class TestResNetModel(unittest.TestCase):

    def setUp(self):
        make_directory("./tests/")

    def tearDown(self):
        shutil.rmtree("./tests/")

    def test_cifar(self):
        session = tf.Session()
        keras.backend.set_session(session)

        comp_params = {"loss": 'categorical_crossentropy',
                       "optimizer": 'adam',
                       "metrics": ['accuracy']}

        # get CIFAR10
        (X_train, Y_train), (X_test, Y_test), _, _ = load_cifar10()
        X_train, Y_train, X_test, Y_test = X_train[:NB_TRAIN], Y_train[:NB_TRAIN], X_test[:NB_TEST], Y_test[:NB_TEST]

        im_shape = X_train[0].shape

        classifier = ResNet(input_shape=im_shape)

        classifier.compile(comp_params)

        # Fit the classifier
        classifier.fit(X_train, Y_train, epochs=2, batch_size=BATCH_SIZE)

        scores = classifier.evaluate(X_test, Y_test)

        print("\naccuracy: %.2f%%" % (scores[1] * 100))

    def test_mnist(self):
        session = tf.Session()
        keras.backend.set_session(session)

        comp_params = {"loss": 'categorical_crossentropy',
                       "optimizer": 'adam',
                       "metrics": ['accuracy']}

        # get MNIST
        (X_train, Y_train), (X_test, Y_test), _, _ = load_mnist()
        X_train, Y_train, X_test, Y_test = X_train[:NB_TRAIN], Y_train[:NB_TRAIN], X_test[:NB_TEST], Y_test[:NB_TEST]
        im_shape = X_train[0].shape

        # Fit the classifier
        classifier = ResNet(input_shape=im_shape)
        classifier.compile(comp_params)
        classifier.fit(X_train, Y_train, epochs=2, batch_size=BATCH_SIZE)
        scores = classifier.evaluate(X_test, Y_test)
        print("\naccuracy: %.2f%%" % (scores[1] * 100))

    def test_save_load_model(self):
        NB_TRAIN = 100
        NB_TEST = 10

        comp_params = {"loss": "categorical_crossentropy",
                       "optimizer": "adam",
                       "metrics": ["accuracy"]}
        session = tf.Session()
        keras.backend.set_session(session)

        # get MNIST
        (X_train, Y_train), (X_test, Y_test), _, _ = load_mnist()
        X_train, Y_train, X_test, Y_test = X_train[:NB_TRAIN], Y_train[:NB_TRAIN], X_test[:NB_TEST], Y_test[:NB_TEST]
        im_shape = X_train[0].shape

        # Fit the classifier
        classifier = ResNet(input_shape=im_shape)
        classifier.compile(comp_params)
        classifier.fit(X_train, Y_train, epochs=1, batch_size=BATCH_SIZE)

        # Test saving
        path = "./tests/save/resnet/"
        save_classifier(classifier, path)

        self.assertTrue(os.path.isfile(path + "model.json"))
        self.assertTrue(os.path.getsize(path + "model.json") > 0)
        self.assertTrue(os.path.isfile(path + "weights.h5"))
        self.assertTrue(os.path.getsize(path + "weights.h5") > 0)

        # Test loading
        loaded_classifier = load_classifier(path)
        scores = classifier.evaluate(X_test, Y_test)
        scores_loaded = loaded_classifier.evaluate(X_test, Y_test)
        self.assertAlmostEqual(scores, scores_loaded)

if __name__ == '__main__':
    unittest.main()
