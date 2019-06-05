from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

import numpy as np

from sklearn.linear_model import LogisticRegression

from art.classifiers import ScikitlearnLogisticRegression
from art.utils import load_dataset

logger = logging.getLogger('testLogger')
np.random.seed(seed=1234)

NB_TRAIN = 20


class TestScikitlearnLogisticRegression(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        (x_train, y_train), (x_test, y_test), _, _ = load_dataset('iris')

        cls.x_train = x_train
        cls.y_train = y_train
        cls.x_test = x_test
        cls.y_test = y_test

        sklearn_model = LogisticRegression(verbose=0, C=1, solver='newton-cg', dual=False, fit_intercept=True)
        cls.classifier = ScikitlearnLogisticRegression(model=sklearn_model)
        cls.classifier.fit(x=cls.x_train, y=cls.y_train)

    def test_predict(self):
        y_predicted = self.classifier.predict(self.x_test[0:1])
        y_expected = [0.07809449, 0.36258262, 0.55932295]

        for i in range(3):
            self.assertAlmostEqual(y_predicted[0, i], y_expected[i], places=4)

    def test_class_gradient(self):
        grad_predicted = self.classifier.class_gradient(self.x_test[0:1], label=np.asarray([[1, 0, 0]]))
        grad_expected = [-1.9793415, 1.3634679, -6.2971964, -2.613862]

        for i in range(4):
            self.assertAlmostEqual(grad_predicted[0, i], grad_expected[i], 4)
