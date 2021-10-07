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

from sklearn.tree import DecisionTreeRegressor

from art.estimators.regression.scikitlearn import ScikitlearnDecisionTreeRegressor
from art.estimators.regression.scikitlearn import ScikitlearnRegressor

from tests.utils import TestBase, master_seed

logger = logging.getLogger(__name__)


class TestScikitlearnDecisionTreeRegressor(TestBase):
    @classmethod
    def setUpClass(cls):
        master_seed(seed=1234)
        super().setUpClass()

        cls.sklearn_model = DecisionTreeRegressor()
        cls.classifier = ScikitlearnDecisionTreeRegressor(model=cls.sklearn_model)
        cls.classifier.fit(x=cls.x_train_iris, y=cls.y_train_iris)

    def test_type(self):
        self.assertIsInstance(self.classifier, type(ScikitlearnRegressor(model=self.sklearn_model)))
        with self.assertRaises(TypeError):
            ScikitlearnDecisionTreeRegressor(model="sklearn_model")

    def test_predict(self):
        y_predicted = self.classifier.predict(self.x_test_iris[0:1])
        y_expected = np.asarray([2.0])
        np.testing.assert_array_almost_equal(y_predicted, y_expected, decimal=4)

    def test_save(self):
        self.classifier.save(filename="test.file", path=None)
        self.classifier.save(filename="test.file", path="./")

    def test_clone_for_refitting(self):
        _ = self.classifier.clone_for_refitting()


if __name__ == "__main__":
    unittest.main()
