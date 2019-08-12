from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

import numpy as np

from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, \
    RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC

from art.classifiers import GPyGaussianProcessClassifier
from art.utils import load_dataset

logger = logging.getLogger('testLogger')
np.random.seed(seed=1234)


class TestScikitlearnDecisionTreeClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(seed=1234)
        # make iris a two class problem for GP
        (x_train, y_train), (x_test, y_test), _, _ = load_dataset('iris')
        # change iris to binary problem, so it is learnable for GPC
        cls.iris = (x_train, y_train[:, 1]), (x_test, y_test[:, 1])
        (X, y), (x_test, y_test) = cls.iris
        # set up GPclassifier
        gpkern = GPy.kern.RBF(np.shape(X)[1])
        m = GPy.models.GPClassification(X, y.reshape(-1, 1), kernel=gpkern)
        m.inference_method = GPy.inference.latent_function_inference.laplace.Laplace()
        m.optimize(messages=True, optimizer='lbfgs')
        # get ART classifier + clean accuracy
        cls.classifier = GPyGaussianProcessClassifier(m)

    def test_predict(self):
        (_, _), (x_test, y_test) = cls.iris
        # predictions should be correct
        self.assertTrue(
            np.mean((m_art.predict(x_test[:3])[:, 0] > 0.5) == y_test[:3]) > 0.6)
        outlier = np.ones(np.shape(x_test[:3]))*10.0
        # output for random points should be 0.5 (don't know)
        self.assertTrue(np.sum(m_art.predict(outlier).flatten()) == 3.0)

    def test_predict_unc(self):
        (_, _), (x_test, y_test) = cls.iris
        outlier = np.ones(np.shape(x_test[:3]))*(np.max(x_test.flatten())*10.0)
        # uncertainty should increase as we go deeper into data
        self.assertTrue(m_art.predict_uncertainty(
            outlier), m_art.predict(outlier))
