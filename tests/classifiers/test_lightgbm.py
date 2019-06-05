from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

import lightgbm as lgb
import numpy as np

from art.classifiers import LightGBMClassifier
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
        cls.y_test = y_test

        num_round = 10
        param = {'objective': 'multiclass', 'metric': 'multi_logloss', 'num_class': 3}
        train_data = lgb.Dataset(cls.x_train, label=cls.y_train)
        model = lgb.train(param, train_data, num_round, valid_sets=[train_data])

        cls.classifier = LightGBMClassifier(model=model)

    def test_predict(self):
        y_predicted = (self.classifier.predict(self.x_test[0:1]))
        y_expected = [0.14644158, 0.16454982, 0.68900861]
        for i in range(3):
            self.assertAlmostEqual(y_predicted[0, i], y_expected[i], 4)
