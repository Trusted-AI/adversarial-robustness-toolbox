# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2022
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

from art.estimators.regression.keras import KerasRegressor

from tests.utils import TestBase, master_seed, get_tabular_regressor_kr

logger = logging.getLogger(__name__)


class TestKerasRegressor(TestBase):
    @classmethod
    def setUpClass(cls):
        master_seed(seed=1234, set_tensorflow=True)
        super().setUpClass()

        import tensorflow as tf

        tf.compat.v1.disable_eager_execution()

        cls.art_model = get_tabular_regressor_kr()

    def test_type(self):
        with self.assertRaises(TypeError):
            KerasRegressor(model="model")

    def test_predict(self):
        y_predicted = self.art_model.predict(self.x_test_diabetes[:4])
        y_expected = np.asarray([[24.9], [52.7], [30.4], [68.1]])
        np.testing.assert_array_almost_equal(y_predicted, y_expected, decimal=1)

    def test_save(self):
        self.art_model.save(filename="test.file", path=None)
        self.art_model.save(filename="test.file", path="./")

    def test_input_shape(self):
        np.testing.assert_equal(self.art_model.input_shape, (10,))

    def test_input_layer(self):
        np.testing.assert_equal(isinstance(self.art_model.input_layer, int), True)

    def test_output_layer(self):
        np.testing.assert_equal(isinstance(self.art_model.output_layer, int), True)

    def test_compute_loss(self):
        test_loss = self.art_model.compute_loss(self.x_test_diabetes[:4], self.y_test_diabetes[:4])
        loss_expected = [6089.8, 2746.3, 5306.8, 1554.9]
        np.testing.assert_array_almost_equal(test_loss, loss_expected, decimal=1)

    def test_loss_gradient(self):
        grad = self.art_model.loss_gradient(self.x_test_diabetes[:4], self.y_test_diabetes[:4])
        grad_expected = [-333.9, 586.4, -1190.9, -123.9, -1206.2, -883.7, 295.9, -830.5, -1333.1, -553.8]
        np.testing.assert_array_almost_equal(grad[0], grad_expected, decimal=1)

    def test_get_activations(self):
        act = self.art_model.get_activations(self.x_test_diabetes[:4], 1)
        act_expected = [0, 0, 0, 7.8, 8.5, 0, 5.6, 0, 6.6, 5.8]
        np.testing.assert_array_almost_equal(act[0], act_expected, decimal=1)


class TestKerasRegressorClass(TestBase):
    @classmethod
    def setUpClass(cls):
        master_seed(seed=1234, set_tensorflow=True)
        super().setUpClass()

        import tensorflow as tf
        import tensorflow.keras as keras

        tf.compat.v1.disable_eager_execution()

        class TestModel(tf.keras.Model):
            def __init__(self):
                super().__init__()
                self.dense1 = keras.layers.Dense(10, activation=tf.nn.relu)
                self.dense2 = keras.layers.Dense(100, activation=tf.nn.relu)
                self.dense3 = keras.layers.Dense(10, activation=tf.nn.relu)
                self.dense4 = keras.layers.Dense(1)

            def call(self, inputs):
                x = self.dense1(inputs)
                return self.dense4(self.dense3(self.dense2(x)))

        cls.keras_model = TestModel()
        cls.keras_model.compile(
            loss=keras.losses.CosineSimilarity(axis=-1, reduction="auto", name="cosine_similarity"),
            optimizer=keras.optimizers.Adam(learning_rate=0.01),
            metrics=["accuracy"],
        )
        cls.keras_model.fit(cls.x_train_diabetes, cls.y_train_diabetes)

        cls.art_model = KerasRegressor(model=cls.keras_model)

    def test_type(self):
        with self.assertRaises(TypeError):
            KerasRegressor(model="model")

    def test_predict(self):
        y_predicted = self.art_model.predict(self.x_test_diabetes[:4])
        np.testing.assert_equal(len(np.unique(y_predicted)), 4)

    def test_save(self):
        self.art_model.save(filename="test.file", path=None)
        self.art_model.save(filename="test.file", path="./")

    def test_input_shape(self):
        np.testing.assert_equal(self.art_model.input_shape, (10,))

    def test_input_layer(self):
        np.testing.assert_equal(isinstance(self.art_model.input_layer, int), True)

    def test_output_layer(self):
        np.testing.assert_equal(isinstance(self.art_model.output_layer, int), True)

    def test_compute_loss(self):
        test_loss = self.art_model.compute_loss(self.x_test_diabetes[:4], self.y_test_diabetes[:4].astype(np.float32))
        # cosine similarity works on vectors, so it returns the same value for each sample
        np.testing.assert_equal(len(np.unique(test_loss)), 1)

    def test_loss_gradient(self):
        grad = self.art_model.loss_gradient(self.x_test_diabetes[:4], self.y_test_diabetes[:4])
        for row in grad:
            if not np.all((row == 0)):
                np.testing.assert_equal(len(np.unique(row)), 10)


class TestKerasRegressorFunctional(TestBase):
    @classmethod
    def setUpClass(cls):
        master_seed(seed=1234, set_tensorflow=True)
        super().setUpClass()

        import tensorflow as tf
        import keras
        from keras.models import Model

        tf.compat.v1.disable_eager_execution()

        def functional():
            in_layer = keras.layers.Input(shape=(10,))
            layer = keras.layers.Dense(100, activation=tf.nn.relu)(in_layer)
            layer = keras.layers.Dense(10, activation=tf.nn.relu)(layer)
            out_layer = keras.layers.Dense(1)(layer)

            model = Model(inputs=[in_layer], outputs=[out_layer])

            model.compile(
                loss=keras.losses.MeanAbsoluteError(),
                optimizer=keras.optimizers.Adam(learning_rate=0.01),
                metrics=["accuracy"],
            )

            return model

        cls.keras_model = functional()
        cls.keras_model.fit(cls.x_train_diabetes, cls.y_train_diabetes)

        cls.art_model = KerasRegressor(model=cls.keras_model)

    def test_type(self):
        with self.assertRaises(TypeError):
            KerasRegressor(model="model")

    def test_predict(self):
        y_predicted = self.art_model.predict(self.x_test_diabetes[:4])
        np.testing.assert_equal(len(np.unique(y_predicted)), 4)

    def test_save(self):
        self.art_model.save(filename="test.file", path=None)
        self.art_model.save(filename="test.file", path="./")

    def test_input_shape(self):
        np.testing.assert_equal(self.art_model.input_shape, (10,))

    def test_input_layer(self):
        np.testing.assert_equal(isinstance(self.art_model.input_layer, int), True)

    def test_output_layer(self):
        np.testing.assert_equal(isinstance(self.art_model.output_layer, int), True)

    def test_compute_loss(self):
        test_loss = self.art_model.compute_loss(self.x_test_diabetes[:4], self.y_test_diabetes[:4].astype(np.float32))
        np.testing.assert_equal(len(np.unique(test_loss)), 4)

    def test_loss_gradient(self):
        grad = self.art_model.loss_gradient(self.x_test_diabetes[:4], self.y_test_diabetes[:4])
        np.testing.assert_equal(len(np.unique(grad[0])), 10)


if __name__ == "__main__":
    unittest.main()
