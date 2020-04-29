# MIT License
#
# Copyright (C) Two Six Labs 2020
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

import numpy as np

from art.poison_detection import SpectralSignatureDefense
from art.utils import load_mnist

from tests.utils import master_seed

logger = logging.getLogger(__name__)

NB_TRAIN, NB_TEST, BATCH_SIZE, EPS_MULTIPLIER, UB_PCT_POISON = 30000, 10, 128, 1.5, 0.2


class TestSpectralSignatureDefense(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        (x_train, y_train), (x_test, y_test), min_, max_ = load_mnist()
        x_train, y_train = x_train[:NB_TRAIN], y_train[:NB_TRAIN]
        cls.mnist = (x_train, y_train), (x_test, y_test), (min_, max_)
        # Create basic keras model
        import keras.backend as k
        from keras.models import Sequential
        from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

        k.set_learning_phase(1)
        model = Sequential()
        model.add(
            Conv2D(
                filters=4,
                kernel_size=(5, 5),
                strides=1,
                activation="relu",
                input_shape=(28, 28, 1),
            )
        )
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(
            Conv2D(
                filters=10,
                kernel_size=(5, 5),
                strides=1,
                activation="relu",
                input_shape=(23, 23, 4),
            )
        )
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(100, activation="relu"))
        model.add(Dense(10, activation="softmax"))

        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )

        from art.estimators.classification.keras import KerasClassifier

        cls.classifier = KerasClassifier(model=model, clip_values=(min_, max_))

        cls.classifier.fit(x_train, y_train, nb_epochs=1, batch_size=128)

        cls.defence = SpectralSignatureDefense(
            cls.classifier,
            x_train,
            y_train,
            batch_size=BATCH_SIZE,
            eps_multiplier=EPS_MULTIPLIER,
            ub_pct_poison=UB_PCT_POISON,
            nb_classes=10,
        )

    def setUp(self):
        # Set master seed
        master_seed(1234)

    @unittest.expectedFailure
    def test_wrong_parameters_1(self):
        self.defence.set_params(batch_size=-1)

    @unittest.expectedFailure
    def test_wrong_parameters_2(self):
        self.defence.set_params(eps_multiplier=-1.0)

    @unittest.expectedFailure
    def test_wrong_parameters_2(self):
        self.defence.set_params(ub_pct_poison=2.0)

    def test_detect_poison(self):
        # Get MNIST
        (x_train, _), (_, _), (_, _) = self.mnist

        _, is_clean_lst = self.defence.detect_poison()

        # Check number of items in is_clean
        self.assertEqual(len(x_train), len(is_clean_lst))

    def test_evaluate_defense(self):
        # Get MNIST
        (x_train, _), (_, _), (_, _) = self.mnist

        is_clean = np.zeros(len(x_train))
        self.defence.evaluate_defence(is_clean)


if __name__ == "__main__":
    unittest.main()
