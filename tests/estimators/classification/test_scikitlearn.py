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
        with self.assertRaises(TypeError):
            ScikitlearnDecisionTreeClassifier(model="sklearn_model")

    def test_predict(self):
        y_predicted = self.classifier.predict(self.x_test_iris[0:1])
        y_expected = np.asarray([[0.0, 0.0, 1.0]])
        np.testing.assert_array_almost_equal(y_predicted, y_expected, decimal=4)

    def test_save(self):
        self.classifier.save(filename="test.file", path=None)
        self.classifier.save(filename="test.file", path="./")

    def test_clone_for_refitting(self):
        _ = self.classifier.clone_for_refitting()


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
        with self.assertRaises(TypeError):
            ScikitlearnExtraTreeClassifier(model="sklearn_model")

    def test_predict(self):
        y_predicted = self.classifier.predict(self.x_test_iris[0:1])
        y_expected = np.asarray([[0.0, 0.0, 1.0]])
        np.testing.assert_array_almost_equal(y_predicted, y_expected, decimal=4)

    def test_save(self):
        self.classifier.save(filename="test.file", path=None)
        self.classifier.save(filename="test.file", path="./")

    def test_clone_for_refitting(self):
        _ = self.classifier.clone_for_refitting()


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
        with self.assertRaises(TypeError):
            ScikitlearnAdaBoostClassifier(model="sklearn_model")

    def test_predict(self):
        y_predicted = self.classifier.predict(self.x_test_iris[0:1])
        y_expected = np.asarray([[3.07686594e-16, 2.23540978e-02, 9.77645902e-01]])
        np.testing.assert_array_almost_equal(y_predicted, y_expected, decimal=4)

    def test_save(self):
        self.classifier.save(filename="test.file", path=None)
        self.classifier.save(filename="test.file", path="./")

    def test_clone_for_refitting(self):
        _ = self.classifier.clone_for_refitting()


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
        with self.assertRaises(TypeError):
            ScikitlearnBaggingClassifier(model="sklearn_model")

    def test_predict(self):
        y_predicted = self.classifier.predict(self.x_test_iris[0:1])
        y_expected = np.asarray([[0.0, 0.0, 1.0]])
        np.testing.assert_array_almost_equal(y_predicted, y_expected, decimal=4)

    def test_save(self):
        self.classifier.save(filename="test.file", path=None)
        self.classifier.save(filename="test.file", path="./")

    def test_clone_for_refitting(self):
        _ = self.classifier.clone_for_refitting()


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
        with self.assertRaises(TypeError):
            ScikitlearnExtraTreesClassifier(model="sklearn_model")

    def test_predict(self):
        y_predicted = self.classifier.predict(self.x_test_iris[0:1])
        y_expected = np.asarray([[0.0, 0.0, 1.0]])
        np.testing.assert_array_almost_equal(y_predicted, y_expected, decimal=4)

    def test_save(self):
        self.classifier.save(filename="test.file", path=None)
        self.classifier.save(filename="test.file", path="./")

    def test_clone_for_refitting(self):
        _ = self.classifier.clone_for_refitting()


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
        with self.assertRaises(TypeError):
            ScikitlearnGradientBoostingClassifier(model="sklearn_model")

    def test_predict(self):
        y_predicted = self.classifier.predict(self.x_test_iris[0:1])
        y_expected = np.asarray([[1.00105813e-05, 2.07276221e-05, 9.99969262e-01]])
        np.testing.assert_array_almost_equal(y_predicted, y_expected, decimal=4)

    def test_save(self):
        self.classifier.save(filename="test.file", path=None)
        self.classifier.save(filename="test.file", path="./")

    def test_clone_for_refitting(self):
        _ = self.classifier.clone_for_refitting()


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
        with self.assertRaises(TypeError):
            ScikitlearnRandomForestClassifier(model="sklearn_model")

    def test_predict(self):
        y_predicted = self.classifier.predict(self.x_test_iris[11:12])
        y_expected = np.asarray([[0.9, 0.1, 0.0]])
        np.testing.assert_array_almost_equal(y_predicted, y_expected, decimal=1)

    def test_save(self):
        self.classifier.save(filename="test.file", path=None)
        self.classifier.save(filename="test.file", path="./")

    def test_clone_for_refitting(self):
        _ = self.classifier.clone_for_refitting()


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
        with self.assertRaises(TypeError):
            ScikitlearnLogisticRegression(model="sklearn_model")

    def test_predict(self):
        y_predicted = self.classifier.predict(self.x_test_iris[0:1])
        y_expected = np.asarray([[0.07997696, 0.36272544, 0.5572976]])
        np.testing.assert_array_almost_equal(y_predicted, y_expected, decimal=3)

    def test_class_gradient_none_1(self):
        grad_predicted = self.classifier.class_gradient(self.x_test_iris[0:1], label=None)
        grad_expected = [
            [
                [-1.98021257, 1.35627806, -6.28222561, -2.60076308],
                [-0.56322294, -0.70427608, -0.98874801, -0.67053026],
                [0.6508137, 0.26377308, 1.54522324, 0.80972391],
            ]
        ]
        np.testing.assert_array_almost_equal(grad_predicted, grad_expected, decimal=3)

    def test_class_gradient_none_2(self):
        grad_predicted = self.classifier.class_gradient(self.x_test_iris[0:2], label=None)
        grad_expected = [
            [
                [-1.98021257, 1.35627806, -6.28222561, -2.60076308],
                [-0.56322294, -0.70427608, -0.98874801, -0.67053026],
                [0.6508137, 0.26377308, 1.54522324, 0.80972391],
            ],
            [
                [-1.92227495, 1.34355378, -6.11815596, -2.52638054],
                [-0.50528532, -0.71700042, -0.82467848, -0.59614766],
                [0.70875132, 0.25104877, 1.70929277, 0.88410652],
            ],
        ]
        np.testing.assert_array_almost_equal(grad_predicted, grad_expected, decimal=3)

    def test_class_gradient_int_1(self):
        grad_predicted = self.classifier.class_gradient(self.x_test_iris[0:1], label=1)
        grad_expected = [[[-0.56317311, -0.70493763, -0.98908609, -0.67106276]]]

        for i_shape in range(4):
            self.assertAlmostEqual(grad_predicted[0, 0, i_shape], grad_expected[0][0][i_shape], 3)

    def test_class_gradient_int_2(self):
        grad_predicted = self.classifier.class_gradient(self.x_test_iris[0:2], label=1)
        grad_expected = [
            [[-0.56317306, -0.70493776, -0.98908573, -0.67106259]],
            [[-0.50522697, -0.71762568, -0.82497531, -0.5966416]],
        ]
        np.testing.assert_array_almost_equal(grad_predicted, grad_expected, decimal=4)

    def test_class_gradient_list_1(self):
        grad_predicted = self.classifier.class_gradient(self.x_test_iris[0:1], label=[1])
        grad_expected = [[[-0.56317311, -0.70493763, -0.98874801, -0.67106276]]]

        for i_shape in range(4):
            self.assertAlmostEqual(grad_predicted[0, 0, i_shape], grad_expected[0][0][i_shape], 3)

    def test_class_gradient_list_2(self):
        grad_predicted = self.classifier.class_gradient(self.x_test_iris[0:2], label=[1, 2])
        grad_expected = [
            [[-0.56317306, -0.70493776, -0.98908573, -0.67106259]],
            [[0.70866591, 0.25158876, 1.70947325, 0.88450021]],
        ]
        np.testing.assert_array_almost_equal(grad_predicted, grad_expected, decimal=3)

    def test_class_gradient_label_wrong_type(self):

        with self.assertRaises(TypeError) as context:
            _ = self.classifier.class_gradient(self.x_test_iris[0:2], label=np.asarray([0, 1, 0]))

        self.assertIn(
            "Unrecognized type for argument `label` with type <class 'numpy.ndarray'>", str(context.exception)
        )

    def test_loss_gradient(self):
        grad_predicted = self.classifier.loss_gradient(self.x_test_iris[0:1], self.y_test_iris[0:1])
        grad_expected = np.asarray([[-0.21690657, -0.08809226, -0.51512082, -0.27002635]])
        np.testing.assert_array_almost_equal(grad_predicted, grad_expected, decimal=4)

    def test_save(self):
        self.classifier.save(filename="test.file", path=None)
        self.classifier.save(filename="test.file", path="./")

    def test_clone_for_refitting(self):
        _ = self.classifier.clone_for_refitting()


class TestScikitlearnBinaryLogisticRegression(TestBase):
    @classmethod
    def setUpClass(cls):
        master_seed(seed=1234)
        super().setUpClass()

        binary_class_index = np.argmax(cls.y_train_iris, axis=1) < 2
        x_train_binary = cls.x_train_iris[
            binary_class_index,
        ]
        y_train_binary = cls.y_train_iris[
            binary_class_index,
        ][:, [0, 1]]

        cls.sklearn_model = LogisticRegression(
            verbose=0, C=1, solver="newton-cg", dual=False, fit_intercept=True, multi_class="ovr"
        )
        cls.classifier = ScikitlearnLogisticRegression(model=cls.sklearn_model)
        cls.classifier.fit(x=x_train_binary, y=y_train_binary)

    def test_type(self):
        self.assertIsInstance(self.classifier, type(SklearnClassifier(model=self.sklearn_model)))
        with self.assertRaises(TypeError):
            ScikitlearnLogisticRegression(model="sklearn_model")

    def test_class_gradient(self):
        grad_predicted = self.classifier.class_gradient(self.x_test_iris[0:1], label=None)
        grad_expected = np.asarray(
            [[[-0.14551339, 0.12298754, -0.45839342, -0.17835225], [0.14551339, -0.12298754, 0.45839342, 0.17835225]]]
        )
        np.testing.assert_array_almost_equal(grad_predicted, grad_expected, decimal=3)

    def test_loss_gradient(self):
        binary_class_index = np.argmax(self.y_test_iris, axis=1) < 2
        x_test_binary = self.x_test_iris[
            binary_class_index,
        ]
        y_test_binary = self.y_test_iris[
            binary_class_index,
        ][:, [0, 1]]

        grad_predicted = self.classifier.loss_gradient(x_test_binary[0:1], y_test_binary[0:1])
        grad_expected = np.asarray([[-0.37703343, 0.31890249, -1.18813638, -0.46208951]])
        np.testing.assert_array_almost_equal(grad_predicted, grad_expected, decimal=4)

    def test_save(self):
        self.classifier.save(filename="test.file", path=None)
        self.classifier.save(filename="test.file", path="./")

    def test_clone_for_refitting(self):
        _ = self.classifier.clone_for_refitting()


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
        with self.assertRaises(TypeError):
            ScikitlearnSVC(model="sklearn_model")

    def test_predict(self):
        y_predicted = self.classifier.predict(self.x_test_iris[0:1])
        y_expected = np.asarray([[0.0, 0.0, 1.0]])
        np.testing.assert_array_almost_equal(y_predicted, y_expected, decimal=4)

    def test_loss_gradient(self):
        grad_predicted = self.classifier.loss_gradient(self.x_test_iris[0:1], self.y_test_iris[0:1])
        grad_expected = np.asarray([[-2.9100819, 0.3048792, -7.935282, -3.840562]])
        np.testing.assert_array_almost_equal(grad_predicted, grad_expected, decimal=4)

    def test_class_gradient_none_1(self):
        grad_predicted = self.classifier.class_gradient(self.x_test_iris[0:1], label=None)
        grad_expected = [
            [
                [-1.5939425, 1.67301144, -6.15095666, -2.4862934],
                [-0.40469415, -1.37572607, 0.46867108, -0.13317975],
                [1.99863665, -0.29728537, 5.68228559, 2.61947315],
            ]
        ]
        np.testing.assert_array_almost_equal(grad_predicted, grad_expected, decimal=4)

    def test_class_gradient_none_2(self):
        grad_predicted = self.classifier.class_gradient(self.x_test_iris[0:2], label=None)
        grad_expected = [
            [
                [-1.5939425, 1.67301144, -6.15095666, -2.4862934],
                [-0.40469415, -1.37572607, 0.46867108, -0.13317975],
                [1.99863665, -0.29728537, 5.68228559, 2.61947315],
            ],
            [
                [-1.5962279, 1.64964639, -6.30453897, -2.60572715],
                [-0.40788449, -1.37232544, 0.53680777, -0.08368929],
                [2.00411239, -0.27732096, 5.7677312, 2.68941644],
            ],
        ]
        np.testing.assert_array_almost_equal(grad_predicted, grad_expected, decimal=4)

    def test_class_gradient_int_1(self):
        grad_predicted = self.classifier.class_gradient(self.x_test_iris[0:1], label=1)
        grad_expected = [[[-0.40469415, -1.37572607, 0.46867108, -0.13317975]]]
        np.testing.assert_array_almost_equal(grad_predicted, grad_expected, decimal=4)

    def test_class_gradient_int_2(self):
        grad_predicted = self.classifier.class_gradient(self.x_test_iris[0:2], label=1)
        grad_expected = [
            [[-0.40469415, -1.37572607, 0.46867108, -0.13317975]],
            [[-0.40788449, -1.37232544, 0.53680777, -0.08368929]],
        ]
        np.testing.assert_array_almost_equal(grad_predicted, grad_expected, decimal=4)

    def test_class_gradient_list_1(self):
        grad_predicted = self.classifier.class_gradient(self.x_test_iris[0:1], label=[1])
        grad_expected = [[[-0.40469415, -1.37572607, 0.46867108, -0.13317975]]]
        np.testing.assert_array_almost_equal(grad_predicted, grad_expected, decimal=4)

    def test_class_gradient_list_2(self):
        grad_predicted = self.classifier.class_gradient(self.x_test_iris[0:2], label=[1, 2])
        grad_expected = [
            [[-0.40469415, -1.37572607, 0.46867108, -0.13317975]],
            [[2.00411239, -0.27732096, 5.7677312, 2.68941644]],
        ]
        np.testing.assert_array_almost_equal(grad_predicted, grad_expected, decimal=4)

    def test_class_gradient_label_wrong_type(self):
        with self.assertRaises(TypeError) as context:
            _ = self.classifier.class_gradient(self.x_test_iris[0:2], label=np.asarray([0, 1, 0]))

        self.assertIn(
            "Unrecognized type for argument `label` with type <class 'numpy.ndarray'>", str(context.exception)
        )

    def test_save(self):
        self.classifier.save(filename="test.file", path=None)
        self.classifier.save(filename="test.file", path="./")

    def test_clone_for_refitting(self):
        _ = self.classifier.clone_for_refitting()


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
        with self.assertRaises(TypeError):
            ScikitlearnSVC(model="sklearn_model")

    def test_predict(self):
        y_predicted = self.classifier.predict(self.x_test_iris[0:1])
        y_expected = np.asarray([[0.0, 0.0, 1.0]])
        np.testing.assert_array_almost_equal(y_predicted, y_expected, decimal=4)

    def test_loss_gradient(self):
        grad_predicted = self.classifier.loss_gradient(self.x_test_iris[0:1], self.y_test_iris[0:1])
        grad_expected = np.asarray([[0.38021886, 0.57562107, -3.599666, -2.3177252]])
        np.testing.assert_array_almost_equal(grad_predicted, grad_expected, decimal=4)

    def test_class_gradient_none(self):
        grad_predicted = self.classifier.class_gradient(self.x_test_iris[0:1], label=None)
        grad_expected = [
            [
                [-0.34648966, 1.63777444, -3.51845999, -1.4609451],
                [-0.11198028, -2.51565392, 0.71538245, -0.44830889],
                [-0.38021886, -0.57562105, 3.59966607, 2.31772514],
            ]
        ]
        np.testing.assert_array_almost_equal(grad_predicted, grad_expected, decimal=4)

    def test_class_gradient_int_1(self):
        grad_predicted = self.classifier.class_gradient(self.x_test_iris[0:1], label=1)
        grad_expected = [[[-0.11198028, -2.51565392, 0.71538245, -0.44830889]]]
        np.testing.assert_array_almost_equal(grad_predicted, grad_expected, decimal=4)

    def test_class_gradient_int_2(self):
        grad_predicted = self.classifier.class_gradient(self.x_test_iris[0:2], label=1)
        grad_expected = [
            [[-0.11198028, -2.51565392, 0.71538245, -0.44830889]],
            [[-0.11198028, -2.51565392, 0.71538245, -0.44830889]],
        ]
        np.testing.assert_array_almost_equal(grad_predicted, grad_expected, decimal=4)

    def test_class_gradient_list_1(self):
        grad_predicted = self.classifier.class_gradient(self.x_test_iris[0:1], label=[1])
        grad_expected = [[[-0.11198028, -2.51565392, 0.71538245, -0.44830889]]]
        np.testing.assert_array_almost_equal(grad_predicted, grad_expected, decimal=4)

    def test_class_gradient_list_2(self):
        grad_predicted = self.classifier.class_gradient(self.x_test_iris[0:2], label=[1, 2])
        grad_expected = [
            [[-0.11198028, -2.51565392, 0.71538245, -0.44830889]],
            [[-0.38021886, -0.57562105, 3.59966607, 2.31772514]],
        ]
        np.testing.assert_array_almost_equal(grad_predicted, grad_expected, decimal=4)

    def test_save(self):
        self.classifier.save(filename="test.file", path=None)
        self.classifier.save(filename="test.file", path="./")

    def test_clone_for_refitting(self):
        _ = self.classifier.clone_for_refitting()


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
        with self.assertRaises(TypeError):
            TestScikitlearnPipeline(model="sklearn_model")

    def test_input_shape(self):
        self.assertEqual(self.classifier.input_shape, (4,))

    def test_save(self):
        self.classifier.save(filename="test.file", path=None)
        self.classifier.save(filename="test.file", path="./")

    def test_clone_for_refitting(self):
        _ = self.classifier.clone_for_refitting()


if __name__ == "__main__":
    unittest.main()
