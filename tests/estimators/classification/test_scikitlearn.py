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

import numpy as np

from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from art.estimators.classification.scikitlearn import (
    ScikitlearnDecisionTreeClassifier,
    ScikitlearnExtraTreeClassifier,
    ScikitlearnAdaBoostClassifier,
    ScikitlearnBaggingClassifier,
    ScikitlearnExtraTreesClassifier,
    ScikitlearnGradientBoostingClassifier,
    ScikitlearnRandomForestClassifier,
    ScikitlearnLogisticRegression,
    ScikitlearnSVC,
)
from art.estimators.classification.scikitlearn import SklearnClassifier

from tests.utils import TestBase, master_seed

logger = logging.getLogger(__name__)


class TestScikitlearnDecisionTreeClassifier(TestBase):
    @classmethod
    def setUpClass(cls):
        master_seed(seed=1234)
        super().setUpClass()

        cls.sklearn_model = DecisionTreeClassifier()
        cls.classifier = ScikitlearnDecisionTreeClassifier(model=cls.sklearn_model)
        cls.classifier.fit(x=cls.x_train_iris, y=cls.y_train_iris)

    def test_type(self):
        self.assertIsInstance(self.classifier, type(SklearnClassifier(model=self.sklearn_model)))

    def test_predict(self):
        y_predicted = self.classifier.predict(self.x_test_iris[0:1])
        y_expected = np.asarray([[0.0, 0.0, 1.0]])
        np.testing.assert_array_almost_equal(y_predicted, y_expected, decimal=4)


class TestScikitlearnExtraTreeClassifier(TestBase):
    @classmethod
    def setUpClass(cls):
        master_seed(seed=1234)
        super().setUpClass()

        cls.sklearn_model = ExtraTreeClassifier()
        cls.classifier = ScikitlearnExtraTreeClassifier(model=cls.sklearn_model)
        cls.classifier.fit(x=cls.x_train_iris, y=cls.y_train_iris)

    def test_type(self):
        self.assertIsInstance(self.classifier, type(SklearnClassifier(model=self.sklearn_model)))

    def test_predict(self):
        y_predicted = self.classifier.predict(self.x_test_iris[0:1])
        y_expected = np.asarray([[0.0, 0.0, 1.0]])
        np.testing.assert_array_almost_equal(y_predicted, y_expected, decimal=4)


class TestScikitlearnAdaBoostClassifier(TestBase):
    @classmethod
    def setUpClass(cls):
        master_seed(seed=1234)
        super().setUpClass()

        cls.sklearn_model = AdaBoostClassifier()
        cls.classifier = ScikitlearnAdaBoostClassifier(model=cls.sklearn_model)
        cls.classifier.fit(x=cls.x_train_iris, y=cls.y_train_iris)

    def test_type(self):
        self.assertIsInstance(self.classifier, type(SklearnClassifier(model=self.sklearn_model)))

    def test_predict(self):
        y_predicted = self.classifier.predict(self.x_test_iris[0:1])
        y_expected = np.asarray([[3.07686594e-16, 2.23540978e-02, 9.77645902e-01]])
        np.testing.assert_array_almost_equal(y_predicted, y_expected, decimal=4)


class TestScikitlearnBaggingClassifier(TestBase):
    @classmethod
    def setUpClass(cls):
        master_seed(seed=1234)
        super().setUpClass()

        cls.sklearn_model = BaggingClassifier()
        cls.classifier = ScikitlearnBaggingClassifier(model=cls.sklearn_model)
        cls.classifier.fit(x=cls.x_train_iris, y=cls.y_train_iris)

    def test_type(self):
        self.assertIsInstance(self.classifier, type(SklearnClassifier(model=self.sklearn_model)))

    def test_predict(self):
        y_predicted = self.classifier.predict(self.x_test_iris[0:1])
        y_expected = np.asarray([[0.0, 0.0, 1.0]])
        np.testing.assert_array_almost_equal(y_predicted, y_expected, decimal=4)


class TestScikitlearnExtraTreesClassifier(TestBase):
    @classmethod
    def setUpClass(cls):
        master_seed(seed=1234)
        super().setUpClass()

        cls.sklearn_model = ExtraTreesClassifier(n_estimators=10)
        cls.classifier = ScikitlearnExtraTreesClassifier(model=cls.sklearn_model)
        cls.classifier.fit(x=cls.x_train_iris, y=cls.y_train_iris)

    def test_type(self):
        self.assertIsInstance(self.classifier, type(SklearnClassifier(model=self.sklearn_model)))

    def test_predict(self):
        y_predicted = self.classifier.predict(self.x_test_iris[0:1])
        y_expected = np.asarray([[0.0, 0.0, 1.0]])
        np.testing.assert_array_almost_equal(y_predicted, y_expected, decimal=4)


class TestScikitlearnGradientBoostingClassifier(TestBase):
    @classmethod
    def setUpClass(cls):
        master_seed(seed=1234)
        super().setUpClass()

        cls.sklearn_model = GradientBoostingClassifier(n_estimators=100)
        cls.classifier = ScikitlearnGradientBoostingClassifier(model=cls.sklearn_model)
        cls.classifier.fit(x=cls.x_train_iris, y=cls.y_train_iris)

    def test_type(self):
        self.assertIsInstance(self.classifier, type(SklearnClassifier(model=self.sklearn_model)))

    def test_predict(self):
        y_predicted = self.classifier.predict(self.x_test_iris[0:1])
        y_expected = np.asarray([[1.00105813e-05, 2.07276221e-05, 9.99969262e-01]])
        np.testing.assert_array_almost_equal(y_predicted, y_expected, decimal=4)


class TestScikitlearnRandomForestClassifier(TestBase):
    @classmethod
    def setUpClass(cls):
        master_seed(seed=1234)
        super().setUpClass()

        cls.sklearn_model = RandomForestClassifier(n_estimators=10)
        cls.classifier = ScikitlearnRandomForestClassifier(model=cls.sklearn_model)
        cls.classifier.fit(x=cls.x_train_iris, y=cls.y_train_iris)

    def test_type(self):
        self.assertIsInstance(self.classifier, type(SklearnClassifier(model=self.sklearn_model)))

    def test_predict(self):
        y_predicted = self.classifier.predict(self.x_test_iris[11:12])
        y_expected = np.asarray([[0.9, 0.1, 0.0]])
        np.testing.assert_array_almost_equal(y_predicted, y_expected, decimal=4)


class TestScikitlearnLogisticRegression(TestBase):
    @classmethod
    def setUpClass(cls):
        master_seed(seed=1234)
        super().setUpClass()

        cls.sklearn_model = LogisticRegression(
            verbose=0, C=1, solver="newton-cg", dual=False, fit_intercept=True, multi_class="ovr"
        )
        cls.classifier = ScikitlearnLogisticRegression(model=cls.sklearn_model)
        cls.classifier.fit(x=cls.x_train_iris, y=cls.y_train_iris)

    def test_type(self):
        self.assertIsInstance(self.classifier, type(SklearnClassifier(model=self.sklearn_model)))

    def test_predict(self):
        y_predicted = self.classifier.predict(self.x_test_iris[0:1])
        y_expected = np.asarray([[0.07809449, 0.36258262, 0.55932295]])
        np.testing.assert_array_almost_equal(y_predicted, y_expected, decimal=4)

    def test_class_gradient_none_1(self):
        grad_predicted = self.classifier.class_gradient(self.x_test_iris[0:1], label=None)
        grad_expected = [
            [
                [-1.97934151, 1.36346793, -6.29719639, -2.61386204],
                [-0.56940532, -0.71100581, -1.00625587, -0.68006182],
                [0.64548057, 0.27053964, 1.5315429, 0.80580771],
            ]
        ]
        np.testing.assert_array_almost_equal(grad_predicted, grad_expected, decimal=4)

    def test_class_gradient_none_2(self):
        grad_predicted = self.classifier.class_gradient(self.x_test_iris[0:2], label=None)
        grad_expected = [
            [
                [-1.97934151, 1.36346793, -6.29719639, -2.61386204],
                [-0.56940532, -0.71100581, -1.00625587, -0.68006182],
                [0.64548057, 0.27053964, 1.5315429, 0.80580771],
            ],
            [
                [-1.92147708, 1.3512013, -6.13324356, -2.53924561],
                [-0.51154077, -0.72327244, -0.84230322, -0.60544527],
                [0.70334512, 0.25827295, 1.69549561, 0.88042426],
            ],
        ]
        np.testing.assert_array_almost_equal(grad_predicted, grad_expected, decimal=4)

    def test_class_gradient_int_1(self):
        grad_predicted = self.classifier.class_gradient(self.x_test_iris[0:1], label=1)
        grad_expected = [[[-0.56940532, -0.71100581, -1.00625587, -0.68006182]]]

        for i_shape in range(4):
            self.assertAlmostEqual(grad_predicted[0, 0, i_shape], grad_expected[0][0][i_shape], 3)

    def test_class_gradient_int_2(self):
        grad_predicted = self.classifier.class_gradient(self.x_test_iris[0:2], label=1)
        grad_expected = [
            [[-0.56940532, -0.71100581, -1.00625587, -0.68006182]],
            [[-0.51154077, -0.72327244, -0.84230322, -0.60544527]],
        ]
        np.testing.assert_array_almost_equal(grad_predicted, grad_expected, decimal=4)

    def test_class_gradient_list_1(self):
        grad_predicted = self.classifier.class_gradient(self.x_test_iris[0:1], label=[1])
        grad_expected = [[[-0.56940532, -0.71100581, -1.00625587, -0.68006182]]]

        for i_shape in range(4):
            self.assertAlmostEqual(grad_predicted[0, 0, i_shape], grad_expected[0][0][i_shape], 3)

    def test_class_gradient_list_2(self):
        grad_predicted = self.classifier.class_gradient(self.x_test_iris[0:2], label=[1, 2])
        grad_expected = [
            [[-0.56940532, -0.71100581, -1.00625587, -0.68006182]],
            [[0.70334512, 0.25827295, 1.69549561, 0.88042426]],
        ]
        np.testing.assert_array_almost_equal(grad_predicted, grad_expected, decimal=4)

    def test_class_gradient_label_wrong_type(self):

        with self.assertRaises(TypeError) as context:
            _ = self.classifier.class_gradient(self.x_test_iris[0:2], label=np.asarray([0, 1, 0]))

        self.assertIn(
            "Unrecognized type for argument `label` with type <class 'numpy.ndarray'>", str(context.exception)
        )

    def test_loss_gradient(self):
        grad_predicted = self.classifier.loss_gradient(self.x_test_iris[0:1], self.y_test_iris[0:1])
        grad_expected = np.asarray([[-2.5487468, 0.6524621, -7.3034525, -3.2939239]])
        np.testing.assert_array_almost_equal(grad_predicted, grad_expected, decimal=4)


class TestScikitlearnBinaryLogisticRegression(TestBase):
    @classmethod
    def setUpClass(cls):
        master_seed(seed=1234)
        super().setUpClass()

        binary_class_index = np.argmax(cls.y_train_iris, axis=1) < 2
        x_train_binary = cls.x_train_iris[
            binary_class_index,
        ]
        y_train_binary = cls.y_train_iris[binary_class_index,][:, [0, 1]]

        cls.sklearn_model = LogisticRegression(
            verbose=0, C=1, solver="newton-cg", dual=False, fit_intercept=True, multi_class="ovr"
        )
        cls.classifier = ScikitlearnLogisticRegression(model=cls.sklearn_model)
        cls.classifier.fit(x=x_train_binary, y=y_train_binary)

    def test_type(self):
        self.assertIsInstance(self.classifier, type(SklearnClassifier(model=self.sklearn_model)))

    def test_class_gradient(self):
        grad_predicted = self.classifier.class_gradient(self.x_test_iris[0:1], label=None)
        grad_expected = np.asarray(
            [[[-0.1428355, 0.12111039, -0.45059183, -0.17579888], [0.1428355, -0.12111039, 0.45059183, 0.17579888]]]
        )
        np.testing.assert_array_almost_equal(grad_predicted, grad_expected, decimal=3)

    def test_loss_gradient(self):
        binary_class_index = np.argmax(self.y_test_iris, axis=1) < 2
        x_test_binary = self.x_test_iris[
            binary_class_index,
        ]
        y_test_binary = self.y_test_iris[binary_class_index,][:, [0, 1]]

        grad_predicted = self.classifier.loss_gradient(x_test_binary[0:1], y_test_binary[0:1])
        grad_expected = np.asarray([[-0.25267282, 0.21424159, -0.79708695, -0.31098431]])
        np.testing.assert_array_almost_equal(grad_predicted, grad_expected, decimal=4)


class TestScikitlearnSVCSVC(TestBase):
    @classmethod
    def setUpClass(cls):
        master_seed(seed=1234)
        super().setUpClass()

        cls.sklearn_model = SVC(gamma="auto")
        cls.classifier = ScikitlearnSVC(model=cls.sklearn_model)
        cls.classifier.fit(x=cls.x_train_iris, y=cls.y_train_iris)

    def test_type(self):
        self.assertIsInstance(self.classifier, type(SklearnClassifier(model=self.sklearn_model)))

    def test_predict(self):
        y_predicted = self.classifier.predict(self.x_test_iris[0:1])
        y_expected = np.asarray([[0.0, 0.0, 1.0]])
        np.testing.assert_array_almost_equal(y_predicted, y_expected, decimal=4)

    def test_loss_gradient(self):
        grad_predicted = self.classifier.loss_gradient(self.x_test_iris[0:1], self.y_test_iris[0:1])
        grad_expected = np.asarray([[-2.7088013, 0.31372938, -7.4563603, -3.5995052]])
        np.testing.assert_array_almost_equal(grad_predicted, grad_expected, decimal=4)

    def test_class_gradient_none_1(self):
        grad_predicted = self.classifier.class_gradient(self.x_test_iris[0:1], label=None)
        grad_expected = [
            [
                [-1.423344, 1.61497281, -5.69580521, -2.29865516],
                [-0.41941481, -1.301, 0.36379309, -0.14524699],
                [1.84275881, -0.3139728, 5.33201212, 2.44390215],
            ]
        ]
        np.testing.assert_array_almost_equal(grad_predicted, grad_expected, decimal=4)

    def test_class_gradient_none_2(self):
        grad_predicted = self.classifier.class_gradient(self.x_test_iris[0:2], label=None)
        grad_expected = [
            [
                [-1.423344, 1.61497281, -5.69580521, -2.29865516],
                [-0.41941481, -1.301, 0.36379309, -0.14524699],
                [1.84275881, -0.3139728, 5.33201212, 2.44390215],
            ],
            [
                [-1.400397, 1.58932854, -5.82124657, -2.40741955],
                [-0.43525245, -1.30608313, 0.38701557, -0.12901405],
                [1.83564945, -0.28324542, 5.434231, 2.5364336],
            ],
        ]
        np.testing.assert_array_almost_equal(grad_predicted, grad_expected, decimal=4)

    def test_class_gradient_int_1(self):
        grad_predicted = self.classifier.class_gradient(self.x_test_iris[0:1], label=1)
        grad_expected = [[[-0.41941481, -1.301, 0.36379309, -0.14524699]]]
        np.testing.assert_array_almost_equal(grad_predicted, grad_expected, decimal=4)

    def test_class_gradient_int_2(self):
        grad_predicted = self.classifier.class_gradient(self.x_test_iris[0:2], label=1)
        grad_expected = [
            [[-0.41941481, -1.301, 0.36379309, -0.14524699]],
            [[-0.43525245, -1.30608313, 0.38701557, -0.12901405]],
        ]
        np.testing.assert_array_almost_equal(grad_predicted, grad_expected, decimal=4)

    def test_class_gradient_list_1(self):
        grad_predicted = self.classifier.class_gradient(self.x_test_iris[0:1], label=[1])
        grad_expected = [[[-0.41941481, -1.301, 0.36379309, -0.14524699]]]
        np.testing.assert_array_almost_equal(grad_predicted, grad_expected, decimal=4)

    def test_class_gradient_list_2(self):
        grad_predicted = self.classifier.class_gradient(self.x_test_iris[0:2], label=[1, 2])
        grad_expected = [
            [[-0.41941481, -1.301, 0.36379309, -0.14524699]],
            [[1.83564945, -0.28324542, 5.434231, 2.5364336]],
        ]
        np.testing.assert_array_almost_equal(grad_predicted, grad_expected, decimal=4)

    def test_class_gradient_label_wrong_type(self):
        with self.assertRaises(TypeError) as context:
            _ = self.classifier.class_gradient(self.x_test_iris[0:2], label=np.asarray([0, 1, 0]))

        self.assertIn(
            "Unrecognized type for argument `label` with type <class 'numpy.ndarray'>", str(context.exception)
        )


class TestScikitlearnSVCLinearSVC(TestBase):
    @classmethod
    def setUpClass(cls):
        master_seed(seed=1234)
        super().setUpClass()

        cls.sklearn_model = LinearSVC()
        cls.classifier = ScikitlearnSVC(model=cls.sklearn_model)
        cls.classifier.fit(x=cls.x_train_iris, y=cls.y_train_iris)

    def test_type(self):
        self.assertIsInstance(self.classifier, type(SklearnClassifier(model=self.sklearn_model)))

    def test_predict(self):
        y_predicted = self.classifier.predict(self.x_test_iris[0:1])
        y_expected = np.asarray([[0.0, 0.0, 1.0]])
        np.testing.assert_array_almost_equal(y_predicted, y_expected, decimal=4)

    def test_loss_gradient(self):
        grad_predicted = self.classifier.loss_gradient(self.x_test_iris[0:1], self.y_test_iris[0:1])
        grad_expected = np.asarray([[0.38537693, 0.5659405, -3.600912, -2.338979]])
        np.testing.assert_array_almost_equal(grad_predicted, grad_expected, decimal=4)

    def test_class_gradient(self):
        grad_predicted = self.classifier.class_gradient(self.x_test_iris[0:1], label=None)
        grad_expected = [
            [
                [-0.34997019, 1.61489704, -3.49002061, -1.46298544],
                [-0.11249995, -2.52947052, 0.7052329, -0.44872424],
                [-0.3853818, -0.5659519, 3.60090744, 2.33898192],
            ]
        ]
        np.testing.assert_array_almost_equal(grad_predicted, grad_expected, decimal=4)


class TestScikitlearnPipeline(TestBase):
    @classmethod
    def setUpClass(cls):
        master_seed(seed=1234)
        super().setUpClass()

        svc = SVC(C=1.0, kernel="rbf", gamma="auto")
        pca = PCA()
        sklearn_model = Pipeline(steps=[("pca", pca), ("svc", svc)])
        cls.classifier = SklearnClassifier(model=sklearn_model)
        cls.classifier.fit(x=cls.x_train_iris, y=cls.y_train_iris)

    def test_predict(self):
        y_predicted = self.classifier.predict(self.x_test_iris[0:1])
        y_expected = np.asarray([[0.0, 0.0, 1.0]])
        np.testing.assert_array_almost_equal(y_predicted, y_expected, decimal=4)

    def test_input_shape(self):
        self.assertEqual(self.classifier.input_shape, (4,))


if __name__ == "__main__":
    unittest.main()
