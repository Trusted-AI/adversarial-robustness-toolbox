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
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

import keras
import keras.backend as k
import numpy as np
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential

from art.classifiers import EnsembleClassifier, KerasClassifier
from art.utils import load_mnist, master_seed

logger = logging.getLogger('testLogger')

BATCH_SIZE = 10
NB_TRAIN = 500
NB_TEST = 100


class TestEnsembleClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        k.clear_session()
        k.set_learning_phase(1)

        # Get MNIST
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        x_train, y_train, x_test, y_test = x_train[:NB_TRAIN], y_train[:NB_TRAIN], x_test[:NB_TEST], y_test[:NB_TEST]
        cls.mnist = ((x_train, y_train), (x_test, y_test))

        model_1 = KerasClassifier((0, 1), cls._get_model(epochs=2))
        model_2 = KerasClassifier((0, 1), cls._get_model(epochs=2))
        cls.ensemble = EnsembleClassifier((0, 1), [model_1, model_2])

    @classmethod
    def tearDownClass(cls):
        k.clear_session()

    def setUp(self):
        master_seed(1234)

    def test_fit(self):
        with self.assertRaises(NotImplementedError):
            self.ensemble.fit(self.mnist[0][0], self.mnist[0][1])

    def test_fit_generator(self):
        with self.assertRaises(NotImplementedError):
            self.ensemble.fit_generator(None)

    def test_layers(self):
        with self.assertRaises(NotImplementedError):
            self.ensemble.get_activations(self.mnist[1][0], layer=2)

    @classmethod
    def _get_model(cls, epochs=1):
        im_shape = cls.mnist[0][0][0].shape

        # Create basic CNN on MNIST; architecture from Keras examples
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=im_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(10, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

        model.fit(cls.mnist[0][0], cls.mnist[0][1], batch_size=BATCH_SIZE, epochs=epochs)
        return model

    def test_predict(self):
        preds = self.ensemble.predict(self.mnist[1][0])
        preds_logits = self.ensemble.predict(self.mnist[1][0], logits=True)
        self.assertTrue(preds.shape == preds_logits.shape)
        self.assertTrue(np.array(preds.shape == (NB_TEST, 10)).all())
        self.assertFalse((preds == preds_logits).all())

        preds_raw = self.ensemble.predict(self.mnist[1][0], raw=True)
        preds_raw_logits = self.ensemble.predict(self.mnist[1][0], raw=True, logits=True)
        self.assertTrue(preds_raw.shape == preds_raw_logits.shape)
        self.assertTrue(preds_raw.shape == (2, NB_TEST, 10))
        self.assertFalse((preds_raw == preds_raw_logits).all())

        self.assertFalse((preds == preds_raw[0]).all())
        self.assertFalse((preds_logits == preds_raw_logits[0]).all())

    def test_loss_gradient(self):
        grad = self.ensemble.loss_gradient(self.mnist[1][0], self.mnist[1][1])
        self.assertTrue(np.array(grad.shape == (NB_TEST, 28, 28, 1)).all())

        grad2 = self.ensemble.loss_gradient(self.mnist[1][0], self.mnist[1][1], raw=True)
        self.assertTrue(np.array(grad2.shape == (2, NB_TEST, 28, 28, 1)).all())

        self.assertFalse((grad2[0] == grad).all())

    def test_class_gradient(self):
        grad = self.ensemble.class_gradient(self.mnist[1][0])
        self.assertTrue(np.array(grad.shape == (NB_TEST, 10, 28, 28, 1)).all())

        grad2 = self.ensemble.class_gradient(self.mnist[1][0], raw=True)
        self.assertTrue(np.array(grad2.shape == (2, NB_TEST, 10, 28, 28, 1)).all())

        self.assertFalse((grad2[0] == grad).all())

    def test_repr(self):
        repr_ = repr(self.ensemble)
        self.assertTrue('art.classifiers.ensemble.EnsembleClassifier' in repr_)
        self.assertTrue('clip_values=(0, 1)' in repr_)
        self.assertTrue('classifier_weights=array([0.5, 0.5])' in repr_)
        self.assertTrue('channel_index=3, defences=None, preprocessing=(0, 1)' in repr_)
