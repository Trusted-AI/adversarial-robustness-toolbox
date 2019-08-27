# MIT License
#
# Copyright (C) IBM Corporation 2018
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

import lightgbm as lgb
import numpy as np

from art.classifiers import LightGBMClassifier
from art.utils import load_dataset

logger = logging.getLogger('testLogger')
np.random.seed(seed=1234)


class TestLightGBMClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        (x_train, y_train), (x_test, y_test), _, _ = load_dataset('iris')

        cls.x_train = x_train
        cls.y_train = np.argmax(y_train, axis=1)
        cls.x_test = x_test
        cls.y_test = np.argmax(y_test, axis=1)

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
