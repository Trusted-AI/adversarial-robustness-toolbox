from __future__ import absolute_import, division, print_function

from config import config_dict
import unittest
import os.path
import shutil

import keras.backend as k
import tensorflow as tf

from src.classifiers.bnn import BNN
from src.classifiers.utils import save_classifier, load_classifier
from src.utils import load_cifar10, load_mnist, make_directory

BATCH_SIZE = 10
NB_TRAIN = 1000
NB_TEST = 10


class TestBNNModel(unittest.TestCase):
    def setUp(self):
        make_directory("./tests/")

    def tearDown(self):
        shutil.rmtree("./tests/")
    
    def test_mc_preds_cifar(self):
        session = tf.Session()
        k.set_session(session)

        # get CIFAR10
        (X_train, Y_train), (X_test, Y_test), _, _ = load_cifar10()
        X_train, Y_train, X_test, Y_test = X_train[:NB_TRAIN], Y_train[:NB_TRAIN], X_test[:NB_TEST], Y_test[:NB_TEST]
        im_shape = X_train[0].shape

        classifier = BNN(im_shape, act="relu", dataset="cifar10")
        classifier.compile({'loss': 'categorical_crossentropy', 'optimizer': 'adam', 'metrics': ['accuracy']})

        # Fit the classifier
        classifier.fit(X_train, Y_train, epochs=1, batch_size=BATCH_SIZE)

        # monte carlo predictions with dropouts turned on
        samples = classifier._mc_preds(X_test)
        print("\nSamples shape", samples.shape)
    
    def test_load_from_bnn(self):
        nb_mc_samples = 5

        comp_params = {'loss': 'categorical_crossentropy',
                       'optimizer': 'adam',
                       'metrics': ['accuracy']}

        session = tf.Session()
        k.set_session(session)

        # get MNIST
        (X_train, Y_train), (X_test, Y_test), _, _ = load_mnist()
        X_train, Y_train, X_test, Y_test = X_train[:NB_TRAIN], Y_train[:NB_TRAIN], X_test[:NB_TEST], Y_test[:NB_TEST]
        im_shape = X_train[0].shape

        # Fit the classifier
        classifier = BNN(im_shape, act="relu")
        classifier.compile(comp_params)
        classifier.fit(X_train, Y_train, epochs=1, batch_size=BATCH_SIZE)

        # Test saving
        path = "./tests/save/bnn/"
        save_classifier(classifier, path)

        self.assertTrue(os.path.isfile(path + "model.json"))
        self.assertTrue(os.path.getsize(path + "model.json") > 0)
        self.assertTrue(os.path.isfile(path + "weights.h5"))
        self.assertTrue(os.path.getsize(path + "weights.h5") > 0)

        # Test loading
        loaded_classifier = BNN._load_from_bnn(file_path=path, nb_mc_samples=nb_mc_samples)
        # Monte Carlo predictions with dropouts turned on
        samples = loaded_classifier._mc_preds(X_test)
        print("\nSamples shape", samples.shape)


if __name__ == '__main__':
    unittest.main()
