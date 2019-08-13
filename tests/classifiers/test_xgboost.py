from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

import xgboost as xgb
import numpy as np

from art.classifiers import XGBoostClassifier
from art.utils import load_dataset

logger = logging.getLogger('testLogger')


class TestXGBoostClassifierBoosterSoftprob(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(seed=1234)
        (x_train, y_train), (x_test, y_test), _, _ = load_dataset('iris')

        cls.x_train = x_train
        cls.y_train = y_train
        cls.x_test = x_test
        cls.y_test = np.argmax(y_test, axis=1)

        num_round = 10
        param = {'objective': 'multi:softprob', 'metric': 'multi_logloss', 'num_class': 3}
        train_data = xgb.DMatrix(cls.x_train, label=cls.y_train)
        evallist = [(train_data, 'train')]
        model = xgb.train(param, train_data, num_round, evallist)

        cls.classifier = XGBoostClassifier(model=model)

    def test_predict(self):
        y_predicted = self.classifier.predict(self.x_test[0:1])
        y_expected = [0.85742694, 0.10067055, 0.0419025]
        for i in range(3):
            self.assertAlmostEqual(y_predicted[0, i], y_expected[i], 4)


class TestXGBoostClassifierBoosterSoftmax(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(seed=1234)
        (x_train, y_train), (x_test, y_test), _, _ = load_dataset('iris')

        cls.x_train = x_train
        cls.y_train = y_train
        cls.x_test = x_test
        cls.y_test = np.argmax(y_test, axis=1)

        num_round = 10
        param = {'objective': 'multi:softmax', 'metric': 'multi_logloss', 'num_class': 3}
        train_data = xgb.DMatrix(cls.x_train, label=cls.y_train)
        evallist = [(train_data, 'train')]
        model = xgb.train(param, train_data, num_round, evallist)

        cls.classifier = XGBoostClassifier(model=model, nb_classes=3)

    def test_predict(self):
        y_predicted = self.classifier.predict(self.x_test[0:1])
        y_expected = [1.0, 0.0, 0.0]
        for i in range(3):
            self.assertAlmostEqual(y_predicted[0, i], y_expected[i], 4)


class TestXGBoostClassifierPythonAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(seed=1234)
        (x_train, y_train), (x_test, y_test), _, _ = load_dataset('iris')

        cls.x_train = x_train
        cls.y_train = y_train
        cls.x_test = x_test
        cls.y_test = np.argmax(y_test, axis=1)

        model = xgb.XGBClassifier(n_estimators=30, max_depth=5)
        model.fit(x_train, np.argmax(y_train, axis=1))

        cls.classifier = XGBoostClassifier(model=model)

    def test_predict(self):
        y_predicted = self.classifier.predict(self.x_test[0:1])
        y_expected = [0.02563512, 0.02925956, 0.94510525]
        for i in range(3):
            self.assertAlmostEqual(y_predicted[0, i], y_expected[i], 4)
