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

from torch import nn, optim

from art.estimators.regression.pytorch import PyTorchRegressor

from tests.utils import TestBase, master_seed

logger = logging.getLogger(__name__)


class TestPytorchRegressor(TestBase):
    @classmethod
    def setUpClass(cls):
        master_seed(seed=1234, set_torch=True)
        super().setUpClass()

        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()

                self.features = nn.Sequential(
                    nn.Linear(10, 100),
                    nn.ReLU(),
                    nn.Linear(100, 10),
                    nn.ReLU(),
                )

                self.output = nn.Linear(10, 1)

            def forward(self, x):
                return self.output(self.features(x))

        cls.pytorch_model = TestModel()

        cls.art_model = PyTorchRegressor(
            model=cls.pytorch_model,
            loss=nn.modules.loss.MSELoss(),
            input_shape=(10,),
            optimizer=optim.Adam(cls.pytorch_model.parameters(), lr=0.01),
        )
        cls.art_model.fit(cls.x_train_diabetes.astype(np.float32), cls.y_train_diabetes.astype(np.float32))

    def test_type(self):
        self.assertIsInstance(
            self.art_model,
            type(
                PyTorchRegressor(
                    model=self.pytorch_model,
                    loss=nn.modules.loss.MSELoss(),
                    input_shape=(10,),
                    optimizer=optim.Adam(self.pytorch_model.parameters(), lr=0.01),
                )
            ),
        )
        with self.assertRaises(TypeError):
            PyTorchRegressor(
                model="model",
                loss=nn.modules.loss.MSELoss,
                input_shape=(10,),
                optimizer=optim.Adam(self.pytorch_model.parameters(), lr=0.01),
            )

    def test_predict(self):
        y_predicted = self.art_model.predict(self.x_test_diabetes[:4].astype(np.float32))
        y_expected = np.array([[19.2], [31.8], [13.8], [42.1]])
        np.testing.assert_array_almost_equal(y_predicted, y_expected, decimal=1)

    def test_save(self):
        self.art_model.save(filename="test.file", path=None)
        self.art_model.save(filename="test.file", path="./")

    def test_input_shape(self):
        np.testing.assert_equal(self.art_model.input_shape, (10,))

    def test_compute_loss(self):
        test_loss = self.art_model.compute_loss(
            self.x_test_diabetes[:4].astype(np.float32), self.y_test_diabetes[:4].astype(np.float32)
        )
        loss_expected = [3461.6, 5214.4, 3994.9, 9003.6]
        np.testing.assert_array_almost_equal(test_loss, loss_expected, decimal=1)

    def test_loss_gradient(self):
        grad = self.art_model.loss_gradient(
            self.x_test_diabetes[:4].astype(np.float32), self.y_test_diabetes[:4].astype(np.float32)
        )
        # grads are same shape as x (i.e., (10,4)), we are looking only at the first row
        grad_expected = [-49.4, 129.9, -170.1, -116.6, -225.2, -171.9, 174.6, -166.8, -223.9, -154.4]
        np.testing.assert_array_almost_equal(grad[0], grad_expected, decimal=1)

    def test_get_activations(self):
        act = self.art_model.get_activations(self.x_test_diabetes[:4].astype(np.float32), 1)
        act_expected = np.array([[19.2], [31.8], [13.8], [42.1]])
        np.testing.assert_array_almost_equal(act, act_expected, decimal=1)


if __name__ == "__main__":
    unittest.main()
