# MIT License
#
# Copyright (C) IBM Corporation 2018
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
from __future__ import absolute_import, division, print_function

from config import config_dict

import unittest
import os.path
import shutil
import keras
import tensorflow as tf

from art.classifiers.cnn import CNN
from art.classifiers.utils import save_classifier, load_classifier
from art.utils import load_cifar10, load_mnist, make_directory

BATCH_SIZE = 10
NB_TRAIN = 1000
NB_TEST = 100


class TestCNNModel(unittest.TestCase):

    def setUp(self):
        make_directory("./tests/")

    def tearDown(self):
        shutil.rmtree("./tests/")

    def test_cifar(self):
        session = tf.Session()
        keras.backend.set_session(session)

        # get CIFAR10
        (X_train, Y_train), (X_test, Y_test), _, _ = load_cifar10()
        X_train, Y_train, X_test, Y_test = X_train[:NB_TRAIN], Y_train[:NB_TRAIN], X_test[:NB_TEST], Y_test[:NB_TEST]
        im_shape = X_train[0].shape

        classifier = CNN(im_shape, act="brelu", dataset="cifar10")
        classifier.compile({'loss': 'categorical_crossentropy', 'optimizer': 'adam', 'metrics': ['accuracy']})

        # Fit the classifier
        classifier.fit(X_train, Y_train, epochs=1, batch_size=BATCH_SIZE)
        scores = classifier.evaluate(X_test, Y_test)
        print("\naccuracy: %.2f%%" % (scores[1] * 100))

    def test_mnist(self):
        session = tf.Session()
        keras.backend.set_session(session)

        # get MNIST
        (X_train, Y_train), (X_test, Y_test), _, _ = load_mnist()
        X_train, Y_train, X_test, Y_test = X_train[:NB_TRAIN], Y_train[:NB_TRAIN], X_test[:NB_TEST], Y_test[:NB_TEST]
        im_shape = X_train[0].shape

        classifier = CNN(im_shape, act="relu")
        classifier.compile({'loss': 'categorical_crossentropy', 'optimizer': 'adam', 'metrics': ['accuracy']})

        # Fit the classifier
        classifier.fit(X_train, Y_train, epochs=1, batch_size=BATCH_SIZE)
        scores = classifier.evaluate(X_test, Y_test)
        print("\naccuracy: %.2f%%" % (scores[1] * 100))

    def test_cnn_brelu(self):
        session = tf.Session()
        keras.backend.set_session(session)

        # get MNIST
        (X_train, Y_train), (X_test, Y_test), _, _ = load_mnist()
        X_train, Y_train, X_test, Y_test = X_train[:NB_TRAIN], Y_train[:NB_TRAIN], X_test[:NB_TEST], Y_test[:NB_TEST]
        im_shape = X_train[0].shape

        classifier = CNN(im_shape, act="brelu", act_params={"alpha": 1, "max_value": 2})
        classifier.compile({'loss': 'categorical_crossentropy', 'optimizer': 'adam', 'metrics': ['accuracy']})

        # Fit the classifier
        classifier.fit(X_train, Y_train, epochs=1, batch_size=BATCH_SIZE)
        act_config = classifier.model.layers[1].get_config()
        self.assertEqual(act_config["alpha"], 1)
        self.assertEqual(act_config["max_value"], 2)

    def test_cnn_batchnorm(self):
        session = tf.Session()
        keras.backend.set_session(session)

        # get MNIST
        (X_train, Y_train), (X_test, Y_test), _, _ = load_mnist()
        X_train, Y_train, X_test, Y_test = X_train[:NB_TRAIN], Y_train[:NB_TRAIN], X_test[:NB_TEST], Y_test[:NB_TEST]
        im_shape = X_train[0].shape

        classifier = CNN(im_shape, act="relu", bnorm=True)
        classifier.compile({'loss': 'categorical_crossentropy', 'optimizer': 'adam', 'metrics': ['accuracy']})

        # Fit the classifier
        classifier.fit(X_train, Y_train, epochs=1, batch_size=BATCH_SIZE)
        bnorm_layer = classifier.model.layers[2]
        self.assertIsInstance(bnorm_layer, keras.layers.normalization.BatchNormalization)

    def test_save_load_cnn(self):
        NB_TRAIN = 100
        NB_TEST = 10

        comp_params = {'loss': 'categorical_crossentropy',
                       'optimizer': 'adam',
                       'metrics': ['accuracy']}

        session = tf.Session()
        keras.backend.set_session(session)

        # get MNIST
        (X_train, Y_train), (X_test, Y_test), _, _ = load_mnist()
        X_train, Y_train, X_test, Y_test = X_train[:NB_TRAIN], Y_train[:NB_TRAIN], X_test[:NB_TEST], Y_test[:NB_TEST]
        im_shape = X_train[0].shape

        classifier = CNN(im_shape, act="brelu")
        classifier.compile(comp_params)

        # Fit the classifier
        classifier.fit(X_train, Y_train, epochs=1, batch_size=BATCH_SIZE)
        path = "./tests/save/cnn/"
        # Test saving
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

    def test_feat_squeeze(self):
        session = tf.Session()
        keras.backend.set_session(session)

        # get MNIST
        (X_train, Y_train), (X_test, Y_test), _, _ = load_mnist()
        X_train, Y_train, X_test, Y_test = X_train[:NB_TRAIN], Y_train[:NB_TRAIN], X_test[:NB_TEST], Y_test[:NB_TEST]
        im_shape = X_train[0].shape

        classifier = CNN(im_shape, act="relu", defences=["featsqueeze1"])
        classifier.compile({'loss': 'categorical_crossentropy', 'optimizer': 'adam', 'metrics': ['accuracy']})

        # Fit the classifier
        classifier.fit(X_train, Y_train, epochs=1, batch_size=BATCH_SIZE)
        scores = classifier.evaluate(X_test, Y_test)
        print("\naccuracy: %.2f%%" % (scores[1] * 100))

    def test_label_smooth(self):

        session = tf.Session()
        keras.backend.set_session(session)

        # get MNIST
        (X_train, Y_train), (X_test, Y_test), _, _ = load_mnist()
        X_train, Y_train, X_test, Y_test = X_train[:NB_TRAIN], Y_train[:NB_TRAIN], X_test[:NB_TEST], Y_test[:NB_TEST]
        im_shape = X_train[0].shape

        classifier = CNN(im_shape, act="relu", defences=["labsmooth"])
        classifier.compile({'loss': 'categorical_crossentropy', 'optimizer': 'adam', 'metrics': ['accuracy']})

        # Fit the classifier
        classifier.fit(X_train, Y_train, epochs=1, batch_size=BATCH_SIZE)
        scores = classifier.evaluate(X_test, Y_test)
        print("\naccuracy: %.2f%%" % (scores[1] * 100))

if __name__ == '__main__':
    unittest.main()
