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
import logging
import unittest

from catboost import CatBoostClassifier
import numpy as np

from art.estimators.classification.catboost import CatBoostARTClassifier

from tests.utils import TestBase, master_seed

logger = logging.getLogger(__name__)


class TestCatBoostClassifier(TestBase):
    @classmethod
    def setUpClass(cls):
        master_seed(seed=1234)
        super().setUpClass()

        cls.y_train_iris_index = np.argmax(cls.y_train_iris, axis=1)
        cls.y_test_iris_index = np.argmax(cls.y_test_iris, axis=1)

        model = CatBoostClassifier(custom_loss=["Accuracy"], random_seed=42, logging_level="Silent")
        cls.classifier = CatBoostARTClassifier(model=model)
        cls.classifier.fit(cls.x_train_iris, cls.y_train_iris_index, cat_features=[], eval_set=[])

    def test_predict(self):
        y_predicted = self.classifier.predict(self.x_test_iris[0:1])
        y_expected = np.asarray([[0.00106307, 0.00118028, 0.99775665]])
        np.testing.assert_array_almost_equal(y_predicted, y_expected, decimal=4)


if __name__ == "__main__":
    unittest.main()
