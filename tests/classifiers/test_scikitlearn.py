from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

import numpy as np

from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, \
    RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC

from art.classifiers import ScikitlearnDecisionTreeClassifier, ScikitlearnExtraTreeClassifier, \
    ScikitlearnAdaBoostClassifier, ScikitlearnBaggingClassifier, ScikitlearnExtraTreesClassifier, \
    ScikitlearnGradientBoostingClassifier, ScikitlearnRandomForestClassifier, ScikitlearnLogisticRegression, \
    ScikitlearnSVC
from art.utils import load_dataset

logger = logging.getLogger('testLogger')
np.random.seed(seed=1234)

(x_train, y_train), (x_test, y_test), _, _ = load_dataset('iris')


class TestScikitlearnDecisionTreeClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(seed=1234)

        sklearn_model = DecisionTreeClassifier()
        cls.classifier = ScikitlearnDecisionTreeClassifier(model=sklearn_model)
        cls.classifier.fit(x=x_train, y=y_train)

    def test_predict(self):
        y_predicted = self.classifier.predict(x_test[0:1])
        y_expected = [0.0, 0.0, 1.0]

        for i in range(3):
            self.assertAlmostEqual(y_predicted[0, i], y_expected[i], places=4)


class TestScikitlearnExtraTreeClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(seed=1234)

        sklearn_model = ExtraTreeClassifier()
        cls.classifier = ScikitlearnExtraTreeClassifier(model=sklearn_model)
        cls.classifier.fit(x=x_train, y=y_train)

    def test_predict(self):
        y_predicted = self.classifier.predict(x_test[0:1])
        y_expected = [0.0, 0.0, 1.0]

        for i in range(3):
            self.assertAlmostEqual(y_predicted[0, i], y_expected[i], places=4)


class TestScikitlearnAdaBoostClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(seed=1234)

        sklearn_model = AdaBoostClassifier()
        cls.classifier = ScikitlearnAdaBoostClassifier(model=sklearn_model)
        cls.classifier.fit(x=x_train, y=y_train)

    def test_predict(self):
        y_predicted = self.classifier.predict(x_test[0:1])
        y_expected = [3.07686594e-16, 2.23540978e-02, 9.77645902e-01]

        for i in range(3):
            self.assertAlmostEqual(y_predicted[0, i], y_expected[i], places=4)


class TestScikitlearnBaggingClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(seed=1234)

        sklearn_model = BaggingClassifier()
        cls.classifier = ScikitlearnBaggingClassifier(model=sklearn_model)
        cls.classifier.fit(x=x_train, y=y_train)

    def test_predict(self):
        y_predicted = self.classifier.predict(x_test[0:1])
        y_expected = [0.0, 0.0, 1.0]

        for i in range(3):
            self.assertAlmostEqual(y_predicted[0, i], y_expected[i], places=4)


class TestScikitlearnExtraTreesClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(seed=1234)

        sklearn_model = ExtraTreesClassifier()
        cls.classifier = ScikitlearnExtraTreesClassifier(model=sklearn_model)
        cls.classifier.fit(x=x_train, y=y_train)

    def test_predict(self):
        y_predicted = self.classifier.predict(x_test[0:1])
        y_expected = [0.0, 0.0, 1.0]

        for i in range(3):
            self.assertAlmostEqual(y_predicted[0, i], y_expected[i], places=4)


class TestScikitlearnGradientBoostingClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(seed=1234)

        sklearn_model = GradientBoostingClassifier()
        cls.classifier = ScikitlearnGradientBoostingClassifier(model=sklearn_model)
        cls.classifier.fit(x=x_train, y=y_train)

    def test_predict(self):
        y_predicted = self.classifier.predict(x_test[0:1])
        y_expected = [1.00105813e-05, 2.07276221e-05, 9.99969262e-01]

        for i in range(3):
            self.assertAlmostEqual(y_predicted[0, i], y_expected[i], places=4)


class TestScikitlearnRandomForestClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(seed=1234)

        sklearn_model = RandomForestClassifier()
        cls.classifier = ScikitlearnRandomForestClassifier(model=sklearn_model)
        cls.classifier.fit(x=x_train, y=y_train)

    def test_predict(self):
        y_predicted = self.classifier.predict(x_test[11:12])
        y_expected = [1.0, 0.0, 0.0]

        for i in range(3):
            self.assertAlmostEqual(y_predicted[0, i], y_expected[i], places=4)


class TestScikitlearnLogisticRegression(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(seed=1234)

        sklearn_model = LogisticRegression(verbose=0, C=1, solver='newton-cg', dual=False, fit_intercept=True)
        cls.classifier = ScikitlearnLogisticRegression(model=sklearn_model)
        cls.classifier.fit(x=x_train, y=y_train)

    def test_predict(self):
        y_predicted = self.classifier.predict(x_test[0:1])
        y_expected = [0.07809449, 0.36258262, 0.55932295]

        for i in range(3):
            self.assertAlmostEqual(y_predicted[0, i], y_expected[i], places=4)

    def test_class_gradient(self):
        grad_predicted = self.classifier.class_gradient(x_test[0:1], label=np.asarray([[1, 0, 0]]))
        grad_expected = [-1.9793415, 1.3634679, -6.2971964, -2.613862]

        for i in range(4):
            self.assertAlmostEqual(grad_predicted[0, i], grad_expected[i], 4)

    def test_class_gradient_2(self):
        grad_predicted = self.classifier.class_gradient(x_test[0:1], y_test[0:1])
        grad_expected = [0.6454806, 0.27053964, 1.5315429, 0.8058077]

        for i in range(4):
            self.assertAlmostEqual(grad_predicted[0, i], grad_expected[i], 4)

    def test_loss_gradient(self):
        grad_predicted = self.classifier.loss_gradient(x_test[0:1], y_test[0:1])
        grad_expected = [-2.5487468, 0.6524621, -7.3034525, -3.2939239]

        for i in range(4):
            self.assertAlmostEqual(grad_predicted[0, i], grad_expected[i], 4)


class TestScikitlearnSVCSVC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(seed=1234)

        sklearn_model = SVC()
        cls.classifier = ScikitlearnSVC(model=sklearn_model)
        cls.classifier.fit(x=x_train, y=y_train)

    def test_predict(self):
        y_predicted = self.classifier.predict(x_test[0:1])
        y_expected = [0.0, 0.0, 1.0]

        for i in range(3):
            self.assertAlmostEqual(y_predicted[0, i], y_expected[i], places=4)

    def test_loss_gradient(self):
        grad_predicted = self.classifier.loss_gradient(x_test[0:1], y_test[0:1])
        grad_expected = [-2.7088013, 0.31372938, -7.4563603, -3.5995052]

        for i in range(4):
            self.assertAlmostEqual(grad_predicted[0, i], grad_expected[i], 4)


class TestScikitlearnSVCLinearSVC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(seed=1234)

        sklearn_model = LinearSVC()
        cls.classifier = ScikitlearnSVC(model=sklearn_model)
        cls.classifier.fit(x=x_train, y=y_train)

    def test_predict(self):
        y_predicted = self.classifier.predict(x_test[0:1])
        y_expected = [0.0, 0.0, 1.0]

        for i in range(3):
            self.assertAlmostEqual(y_predicted[0, i], y_expected[i], places=4)

    def test_loss_gradient(self):
        grad_predicted = self.classifier.loss_gradient(x_test[0:1], y_test[0:1])
        grad_expected = [0.38537693, 0.5659405, -3.600912, -2.338979]

        for i in range(4):
            self.assertAlmostEqual(grad_predicted[0, i], grad_expected[i], 4)
