from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

from catboost import CatBoostClassifier
import numpy as np

from art.classifiers import CatBoostARTClassifier
from art.utils import load_dataset

logger = logging.getLogger('testLogger')
np.random.seed(seed=1234)


class TestXGBoostClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        (x_train, y_train), (x_test, y_test), _, _ = load_dataset('iris')

        cls.x_train = x_train
        cls.y_train = np.argmax(y_train, axis=1)
        cls.x_test = x_test
        cls.y_test = np.argmax(y_test, axis=1)

        model = CatBoostClassifier(custom_loss=['Accuracy'], random_seed=42, logging_level='Silent')
        model.fit(cls.x_train, cls.y_train, cat_features=None, eval_set=(cls.x_train, cls.y_train))

        cls.classifier = CatBoostARTClassifier(model=model)

    def test_predict(self):
        y_predicted = (self.classifier.predict(self.x_test[0:1]))
        y_expected = [0.00289702, 0.00442229, 0.99268069]
        for i in range(3):
            self.assertAlmostEqual(y_predicted[0, i], y_expected[i], 4)
