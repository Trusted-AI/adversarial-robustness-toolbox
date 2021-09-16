# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2019
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
import os
import unittest

import numpy as np

from art.defences.transformer.poisoning import NeuralCleanse
from art.utils import load_dataset

from tests.utils import master_seed

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
logger = logging.getLogger(__name__)

BATCH_SIZE = 100
NB_TRAIN = 5000
NB_TEST = 1000


class TestNeuralCleanse(unittest.TestCase):
    """
    A unittest class for testing Randomized Smoothing as a post-processing step for classifiers.
    """

    @classmethod
    def setUpClass(cls):
        # Get MNIST
        (x_train, y_train), (x_test, y_test), _, _ = load_dataset("mnist")
        x_train, y_train = x_train[:NB_TRAIN], y_train[:NB_TRAIN]
        x_test, y_test = x_test[:NB_TEST], y_test[:NB_TEST]
        cls.mnist = (x_train, y_train), (x_test, y_test)

    def setUp(self):
        master_seed(seed=1234)

    def test_keras(self):
        """
        Test with a KerasClassifier.
        :return:
        """
        # Build KerasClassifier
        import tensorflow as tf

        tf.compat.v1.disable_eager_execution()
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Flatten, Conv2D
        from tensorflow.keras.losses import CategoricalCrossentropy
        from tensorflow.keras.optimizers import Adam

        model = Sequential()
        model.add(Conv2D(filters=4, kernel_size=(5, 5), strides=1, activation="relu", input_shape=(28, 28, 1)))
        model.add(Flatten())
        model.add(Dense(100, activation="relu"))
        model.add(Dense(100, activation="relu"))
        model.add(Dense(10, activation="linear"))

        model.compile(
            loss=CategoricalCrossentropy(from_logits=True), optimizer=Adam(learning_rate=0.01), metrics=["accuracy"]
        )

        from art.estimators.classification import KerasClassifier

        krc = KerasClassifier(model=model, clip_values=(0, 1))

        # Get MNIST
        (x_train, y_train), (x_test, y_test) = self.mnist

        for i in range(2500):
            if np.argmax(y_train[[i]], axis=1) == 0:
                y_train[i, :] = 0
                y_train[i, 1] = 1
                x_train[i, 0:5, 0:5, :] = 1.0

            if np.argmax(y_train[[i]], axis=1) == 9:
                y_train[i, :] = 0
                y_train[i, 9] = 1
                x_train[i, 0:5, 0:5, :] = 1.0

        for i in range(500):
            if np.argmax(y_test[[i]], axis=1) == 0:
                y_test[i, :] = 0
                y_test[i, 1] = 1
                x_test[i, 0:5, 0:5, :] = 1.0

            if np.argmax(y_test[[i]], axis=1) == 9:
                y_test[i, :] = 0
                y_test[i, 9] = 1
                x_test[i, 0:5, 0:5, :] = 1.0

        krc.fit(x_train, y_train, nb_epochs=3)

        cleanse = NeuralCleanse(krc)
        defense_cleanse = cleanse(krc, steps=1, patience=1)
        defense_cleanse.mitigate(x_test, y_test, mitigation_types=["filtering", "pruning", "unlearning"])

        # is_fitted
        assert cleanse._is_fitted == cleanse.is_fitted

        # get_classifier
        assert cleanse.get_classifier

        # set_params
        cleanse.set_params(**{"batch_size": 1})
        assert cleanse.batch_size == 1


if __name__ == "__main__":
    unittest.main()
