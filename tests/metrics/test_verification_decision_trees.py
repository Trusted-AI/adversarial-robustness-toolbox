# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2019
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

from xgboost import XGBClassifier
import lightgbm
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier

from art.estimators.classification.xgboost import XGBoostClassifier
from art.estimators.classification.lightgbm import LightGBMClassifier
from art.estimators.classification.scikitlearn import SklearnClassifier
from art.utils import load_dataset
from art.metrics.verification_decisions_trees import RobustnessVerificationTreeModelsCliqueMethod

from tests.utils import master_seed

logger = logging.getLogger(__name__)

NB_TRAIN = 100
NB_TEST = 100


class TestMetricsTrees(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (x_train, y_train), (x_test, y_test), _, _ = load_dataset("mnist")

        cls.n_classes = 10
        cls.n_features = 28 * 28
        n_train = x_train.shape[0]
        n_test = x_test.shape[0]
        x_train = x_train.reshape((n_train, cls.n_features))
        x_test = x_test.reshape((n_test, cls.n_features))

        cls.x_train = x_train[:NB_TRAIN]
        cls.y_train = y_train[:NB_TRAIN]
        cls.x_test = x_test[:NB_TEST]
        cls.y_test = y_test[:NB_TEST]

    @classmethod
    def setUp(cls):
        master_seed(seed=42)

    def test_XGBoost(self):
        model = XGBClassifier(n_estimators=4, max_depth=6, objective="multi:softprob", eval_metric="merror")
        model.fit(self.x_train, np.argmax(self.y_train, axis=1))

        classifier = XGBoostClassifier(model=model, nb_features=self.n_features, nb_classes=self.n_classes)

        rt = RobustnessVerificationTreeModelsCliqueMethod(classifier=classifier, verbose=False)
        average_bound, verified_error = rt.verify(
            x=self.x_test, y=self.y_test, eps_init=0.3, nb_search_steps=10, max_clique=2, max_level=2
        )

        self.assertEqual(average_bound, 0.0011425781249999997)
        self.assertEqual(verified_error, 1.0)

    def test_LightGBM(self):
        train_data = lightgbm.Dataset(self.x_train, label=np.argmax(self.y_train, axis=1))
        test_data = lightgbm.Dataset(self.x_test, label=np.argmax(self.y_test, axis=1))

        parameters = {
            "objective": "multiclass",
            "num_class": self.n_classes,
            "metric": "multi_logloss",
            "is_unbalance": "true",
            "boosting": "gbdt",
            "num_leaves": 5,
            "feature_fraction": 0.5,
            "bagging_fraction": 0.5,
            "bagging_freq": 0,
            "learning_rate": 0.05,
            "verbose": 0,
        }

        model = lightgbm.train(parameters, train_data, valid_sets=test_data, num_boost_round=2, early_stopping_rounds=1)

        classifier = LightGBMClassifier(model=model)

        rt = RobustnessVerificationTreeModelsCliqueMethod(classifier=classifier, verbose=False)
        average_bound, verified_error = rt.verify(
            x=self.x_test, y=self.y_test, eps_init=0.3, nb_search_steps=10, max_clique=2, max_level=2
        )

        self.assertEqual(average_bound, 0.047742187500000005)
        self.assertEqual(verified_error, 0.94)

    def test_GradientBoosting(self):
        model = GradientBoostingClassifier(n_estimators=4, max_depth=6)
        model.fit(self.x_train, np.argmax(self.y_train, axis=1))

        classifier = SklearnClassifier(model=model)

        rt = RobustnessVerificationTreeModelsCliqueMethod(classifier=classifier, verbose=False)
        average_bound, verified_error = rt.verify(
            x=self.x_test, y=self.y_test, eps_init=0.3, nb_search_steps=10, max_clique=2, max_level=2
        )

        self.assertAlmostEqual(average_bound, 0.009, delta=0.0002)
        self.assertEqual(verified_error, 1.0)

    def test_RandomForest(self):
        model = RandomForestClassifier(n_estimators=4, max_depth=6)
        model.fit(self.x_train, np.argmax(self.y_train, axis=1))

        classifier = SklearnClassifier(model=model)

        rt = RobustnessVerificationTreeModelsCliqueMethod(classifier=classifier, verbose=False)
        average_bound, verified_error = rt.verify(
            x=self.x_test, y=self.y_test, eps_init=0.3, nb_search_steps=10, max_clique=2, max_level=2
        )

        self.assertEqual(average_bound, 0.016482421874999993)
        self.assertEqual(verified_error, 1.0)

    def test_ExtraTrees(self):
        model = ExtraTreesClassifier(n_estimators=4, max_depth=6)
        model.fit(self.x_train, np.argmax(self.y_train, axis=1))

        classifier = SklearnClassifier(model=model)

        rt = RobustnessVerificationTreeModelsCliqueMethod(classifier=classifier, verbose=False)
        average_bound, verified_error = rt.verify(
            x=self.x_test, y=self.y_test, eps_init=0.3, nb_search_steps=10, max_clique=2, max_level=2
        )

        self.assertEqual(average_bound, 0.05406445312499999)
        self.assertEqual(verified_error, 0.96)


if __name__ == "__main__":
    unittest.main()
