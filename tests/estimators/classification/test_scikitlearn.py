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
from art.utils import check_and_transform_label_format

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

    def test_multi_label(self):
        x_train = self.x_train_iris
        y_train = self.y_train_iris
        x_test = self.x_test_iris
        y_test = self.y_test_iris

        # make multi-label binary
        y_train = np.column_stack((y_train, y_train, y_train))
        y_train[y_train > 1] = 1
        y_test = np.column_stack((y_test, y_test, y_test))
        y_test[y_test > 1] = 1

        underlying_model = DecisionTreeClassifier()
        underlying_model.fit(x_train, y_train)
        model = ScikitlearnDecisionTreeClassifier(model=underlying_model)

        pred = model.predict(x_test)
        assert pred[0].shape[0] == x_test.shape[0]
        assert isinstance(model.nb_classes, np.ndarray)
        with self.assertRaises(TypeError):
            check_and_transform_label_format(y_train, nb_classes=model.nb_classes)


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
        y_expected = np.asarray([[0.25544427, 0.35099888, 0.39355685]])
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

        cls.sklearn_model = LogisticRegression(verbose=0, C=1, solver="newton-cg", dual=False, fit_intercept=True)
        cls.classifier = ScikitlearnLogisticRegression(model=cls.sklearn_model)
        cls.classifier.fit(x=cls.x_train_iris, y=cls.y_train_iris)

    def test_type(self):
        self.assertIsInstance(self.classifier, type(SklearnClassifier(model=self.sklearn_model)))
        with self.assertRaises(TypeError):
            ScikitlearnLogisticRegression(model="sklearn_model")

    def test_predict(self):
        y_predicted = self.classifier.predict(self.x_test_iris[0:1])
        y_expected = np.asarray([[0.05651609, 0.3351358, 0.60834813]])
        np.testing.assert_array_almost_equal(y_predicted, y_expected, decimal=3)

    def test_class_gradient_none_1(self):
        grad_predicted = self.classifier.class_gradient(self.x_test_iris[0:1], label=None)
        grad_expected = [
            [
                [-1.73085618, 1.11711752, -5.45074177, -2.27298689],
                [-0.52926081, -0.60170209, -1.15163267, -0.75595492],
                [0.45236516, 0.22769327, 1.14080715, 0.62761414],
            ]
        ]
        np.testing.assert_array_almost_equal(grad_predicted, grad_expected, decimal=3)

    def test_class_gradient_none_2(self):
        grad_predicted = self.classifier.class_gradient(self.x_test_iris[0:2], label=None)
        grad_expected = [
            [
                [
                    -1.73085618,
                    1.11711752,
                    -5.45074129,
                    -2.27298665,
                ],
                [
                    -0.52926081,
                    -0.60170197,
                    -1.15163255,
                    -0.75595486,
                ],
                [
                    0.45236516,
                    0.22769333,
                    1.14080727,
                    0.6276142,
                ],
            ],
            [
                [-1.66938925, 1.12607121, -5.28372431, -2.18912697],
                [-0.46779382, -0.5927484, -0.98461545, -0.67209512],
                [0.51383215, 0.23664697, 1.30782437, 0.71147394],
            ],
        ]
        np.testing.assert_array_almost_equal(grad_predicted, grad_expected, decimal=3)

    def test_class_gradient_int_1(self):
        grad_predicted = self.classifier.class_gradient(self.x_test_iris[0:1], label=1)
        grad_expected = [[[-0.52926081, -0.60170209, -1.15163267, -0.75595492]]]

        for i_shape in range(4):
            self.assertAlmostEqual(grad_predicted[0, 0, i_shape], grad_expected[0][0][i_shape], 3)

    def test_class_gradient_int_2(self):
        grad_predicted = self.classifier.class_gradient(self.x_test_iris[0:2], label=1)
        grad_expected = [
            [[-0.52926081, -0.60170197, -1.15163255, -0.75595486]],
            [[-0.46779382, -0.5927484, -0.98461545, -0.67209512]],
        ]
        np.testing.assert_array_almost_equal(grad_predicted, grad_expected, decimal=4)

    def test_class_gradient_list_1(self):
        grad_predicted = self.classifier.class_gradient(self.x_test_iris[0:1], label=[1])
        grad_expected = [[[-0.52926081, -0.60170209, -1.15163267, -0.75595492]]]

        for i_shape in range(4):
            self.assertAlmostEqual(grad_predicted[0, 0, i_shape], grad_expected[0][0][i_shape], 3)

    def test_class_gradient_list_2(self):
        grad_predicted = self.classifier.class_gradient(self.x_test_iris[0:2], label=[1, 2])
        grad_expected = [
            [[-0.52926081, -0.60170197, -1.15163255, -0.75595486]],
            [[0.51383215, 0.23664697, 1.30782437, 0.71147394]],
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
        grad_expected = np.asarray([[-0.15078837, -0.07589776, -0.38026905, -0.2092047]])
        np.testing.assert_array_almost_equal(grad_predicted, grad_expected, decimal=4)

    def test_loss_gradient_index_labels(self):
        y_one_hot = self.y_test_iris[0:3]
        y_index = np.argmax(y_one_hot, axis=1)

        grad_one_hot = self.classifier.loss_gradient(self.x_test_iris[0:3], y_one_hot)
        grad_index = self.classifier.loss_gradient(self.x_test_iris[0:3], y_index)
        grad_index_column = self.classifier.loss_gradient(self.x_test_iris[0:3], np.expand_dims(y_index, axis=1))

        np.testing.assert_array_almost_equal(grad_index, grad_one_hot, decimal=4)
        np.testing.assert_array_almost_equal(grad_index_column, grad_one_hot, decimal=4)

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
        x_train_binary = cls.x_train_iris[binary_class_index,]
        y_train_binary = cls.y_train_iris[binary_class_index,][:, [0, 1]]

        cls.sklearn_model = LogisticRegression(verbose=0, C=1, solver="newton-cg", dual=False, fit_intercept=True)
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
        x_test_binary = self.x_test_iris[binary_class_index,]
        y_test_binary = self.y_test_iris[binary_class_index,][:, [0, 1]]

        grad_predicted = self.classifier.loss_gradient(x_test_binary[0:1], y_test_binary[0:1])
        grad_expected = np.asarray([[-0.37703343, 0.31890249, -1.18813638, -0.46208951]])
        np.testing.assert_array_almost_equal(grad_predicted, grad_expected, decimal=4)

    def test_loss_gradient_index_labels(self):
        binary_class_index = np.argmax(self.y_test_iris, axis=1) < 2
        x_test_binary = self.x_test_iris[binary_class_index,]
        y_test_binary = self.y_test_iris[binary_class_index,][:, [0, 1]]
        y_index = np.argmax(y_test_binary[0:3], axis=1)

        grad_one_hot = self.classifier.loss_gradient(x_test_binary[0:3], y_test_binary[0:3])
        grad_index = self.classifier.loss_gradient(x_test_binary[0:3], y_index)

        np.testing.assert_array_almost_equal(grad_index, grad_one_hot, decimal=4)

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

    def test_loss_gradient_index_labels(self):
        y_one_hot = self.y_test_iris[0:3]
        y_index = np.argmax(y_one_hot, axis=1)

        grad_one_hot = self.classifier.loss_gradient(self.x_test_iris[0:3], y_one_hot)
        grad_index = self.classifier.loss_gradient(self.x_test_iris[0:3], y_index)
        grad_index_column = self.classifier.loss_gradient(self.x_test_iris[0:3], np.expand_dims(y_index, axis=1))

        np.testing.assert_array_almost_equal(grad_index, grad_one_hot, decimal=4)
        np.testing.assert_array_almost_equal(grad_index_column, grad_one_hot, decimal=4)

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

    def test_loss_gradient_index_labels(self):
        y_one_hot = self.y_test_iris[0:3]
        y_index = np.argmax(y_one_hot, axis=1)

        grad_one_hot = self.classifier.loss_gradient(self.x_test_iris[0:3], y_one_hot)
        grad_index = self.classifier.loss_gradient(self.x_test_iris[0:3], y_index)
        grad_index_column = self.classifier.loss_gradient(self.x_test_iris[0:3], np.expand_dims(y_index, axis=1))

        np.testing.assert_array_almost_equal(grad_index, grad_one_hot, decimal=4)
        np.testing.assert_array_almost_equal(grad_index_column, grad_one_hot, decimal=4)

    def test_class_gradient_none(self):
        grad_predicted = self.classifier.class_gradient(self.x_test_iris[0:1], label=None)
        grad_expected = [
            [
                [-0.34659522, 1.6376213, -3.51851979, -1.46078468],
                [-0.1121541, -2.51552211, 0.71569165, -0.44883585],
                [-0.380205, -0.57552204, 3.59972012, 2.31760663],
            ]
        ]
        np.testing.assert_array_almost_equal(grad_predicted, grad_expected, decimal=4)

    def test_class_gradient_int_1(self):
        grad_predicted = self.classifier.class_gradient(self.x_test_iris[0:1], label=1)
        grad_expected = [[[-0.1121541, -2.51552211, 0.71569165, -0.44883585]]]
        np.testing.assert_array_almost_equal(grad_predicted, grad_expected, decimal=4)

    def test_class_gradient_int_2(self):
        grad_predicted = self.classifier.class_gradient(self.x_test_iris[0:2], label=1)
        grad_expected = [
            [[-0.1121541, -2.51552211, 0.71569165, -0.44883585]],
            [[-0.1121541, -2.51552211, 0.71569165, -0.44883585]],
        ]
        np.testing.assert_array_almost_equal(grad_predicted, grad_expected, decimal=4)

    def test_class_gradient_list_1(self):
        grad_predicted = self.classifier.class_gradient(self.x_test_iris[0:1], label=[1])
        grad_expected = [[[-0.1121541, -2.51552211, 0.71569165, -0.44883585]]]
        np.testing.assert_array_almost_equal(grad_predicted, grad_expected, decimal=4)

    def test_class_gradient_list_2(self):
        grad_predicted = self.classifier.class_gradient(self.x_test_iris[0:2], label=[1, 2])
        grad_expected = [
            [[-0.1121541, -2.51552211, 0.71569165, -0.44883585]],
            [[-0.380205, -0.57552204, 3.59972012, 2.31760663]],
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
