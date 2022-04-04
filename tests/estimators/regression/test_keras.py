# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
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

import keras
import tensorflow as tf

from art.estimators.regression.keras import KerasRegressor

from tests.utils import TestBase, master_seed

logger = logging.getLogger(__name__)


class TestScikitlearnDecisionTreeRegressor(TestBase):
    @classmethod
    def setUpClass(cls):
        master_seed(seed=1234)
        super().setUpClass()

        tf.compat.v1.disable_eager_execution()

        model = keras.models.Sequential()
        # model.add(keras.Input(shape=(10,)))
        model.add(keras.layers.Dense(10, activation="relu"))
        model.add(keras.layers.Dense(100, activation="relu"))
        model.add(keras.layers.Dense(10, activation="relu"))
        model.add(keras.layers.Dense(1))

        model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.Adam(learning_rate=0.01),
                      metrics=["accuracy"])

        cls.keras_model = model
        cls.art_model = KerasRegressor(model=cls.keras_model)
        cls.art_model.fit(x=cls.x_train_diabetes, y=cls.y_train_diabetes)

    def test_type(self):
        self.assertIsInstance(self.art_model, type(KerasRegressor(model=self.keras_model)))
        with self.assertRaises(TypeError):
            KerasRegressor(model="model")

    def test_predict(self):
        y_predicted = self.art_model.predict(self.x_test_diabetes[:4])
        y_expected = np.asarray([69.0, 81.0, 68.0, 68.0])
        # np.testing.assert_array_almost_equal(y_predicted, y_expected, decimal=1)

    def test_save(self):
        self.art_model.save(filename="test.file", path=None)
        self.art_model.save(filename="test.file", path="./")

    def test_clone_for_refitting(self):
        _ = self.art_model.clone_for_refitting()


if __name__ == "__main__":
    unittest.main()
