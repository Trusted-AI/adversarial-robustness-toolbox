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

import xgboost as xgb
import numpy as np

from art.estimators.classification.xgboost import XGBoostClassifier

from tests.utils import TestBase, master_seed

logger = logging.getLogger(__name__)


class TestXGBoostClassifierBoosterSoftprob(TestBase):
    @classmethod
    def setUpClass(cls):
        master_seed(seed=1234)
        super().setUpClass()

        num_round = 10
        param = {"objective": "multi:softprob", "metric": "multi_logloss", "num_class": 3}
        train_data = xgb.DMatrix(cls.x_train_iris, label=np.argmax(cls.y_train_iris, axis=1))
        eval_list = [(train_data, "train")]
        model = xgb.train(param, train_data, num_round, eval_list)

        cls.classifier = XGBoostClassifier(model=model, nb_classes=3)

    def test_predict(self):
        y_predicted = self.classifier.predict(self.x_test_iris[0:1])
        y_expected = np.asarray([[0.0236, 0.0268, 0.9496]])
        np.testing.assert_array_almost_equal(y_predicted, y_expected, decimal=4)


class TestXGBoostClassifierBoosterSoftmax(TestBase):
    @classmethod
    def setUpClass(cls):
        master_seed(seed=1234)
        super().setUpClass()

        num_round = 10
        param = {"objective": "multi:softmax", "metric": "multi_logloss", "num_class": 3}
        train_data = xgb.DMatrix(cls.x_train_iris, label=np.argmax(cls.y_train_iris, axis=1))
        eval_list = [(train_data, "train")]
        model = xgb.train(param, train_data, num_round, eval_list)

        cls.classifier = XGBoostClassifier(model=model, nb_classes=3)

    def test_predict(self):
        y_predicted = self.classifier.predict(self.x_test_iris[0:1])
        y_expected = np.asarray([[0.0, 0.0, 1.0]])
        np.testing.assert_array_almost_equal(y_predicted, y_expected, decimal=4)


class TestXGBoostClassifierPythonAPI(TestBase):
    @classmethod
    def setUpClass(cls):
        master_seed(seed=1234)
        super().setUpClass()

        model = xgb.XGBClassifier(n_estimators=30, max_depth=5)
        model.fit(cls.x_train_iris, np.argmax(cls.y_train_iris, axis=1))

        cls.classifier = XGBoostClassifier(model=model)

    def test_predict(self):
        y_predicted = self.classifier.predict(self.x_test_iris[0:1])
        y_expected = np.asarray([[0.00280161, 0.00359648, 0.99360186]])
        np.testing.assert_array_almost_equal(y_predicted, y_expected, decimal=4)


if __name__ == "__main__":
    unittest.main()
