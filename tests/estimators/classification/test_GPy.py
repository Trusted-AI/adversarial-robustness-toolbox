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
import GPy

from art.estimators.classification.GPy import GPyGaussianProcessClassifier

from tests.utils import TestBase, master_seed

logger = logging.getLogger(__name__)


class TestGPyGaussianProcessClassifier(TestBase):
    """
    This class tests the GPy Gaussian Process classifier.
    """

    @classmethod
    def setUpClass(cls):
        master_seed(seed=1234)
        super().setUpClass()

        # change iris to binary problem, so it is learnable for GPC
        cls.y_train_iris_binary = cls.y_train_iris[:, 1]
        cls.y_test_iris_binary = cls.y_test_iris[:, 1]

        # set up GPclassifier
        gpkern = GPy.kern.RBF(np.shape(cls.x_train_iris)[1])
        m = GPy.models.GPClassification(cls.x_train_iris, cls.y_train_iris_binary.reshape(-1, 1), kernel=gpkern)
        m.inference_method = GPy.inference.latent_function_inference.laplace.Laplace()
        m.optimize(messages=True, optimizer="lbfgs")

        # get ART classifier + clean accuracy
        cls.classifier = GPyGaussianProcessClassifier(m)

    def setUp(self):
        master_seed(seed=1234)
        super().setUp()

    def test_predict(self):
        # predictions should be correct
        self.assertTrue(
            np.mean((self.classifier.predict(self.x_test_iris[:3])[:, 0] > 0.5) == self.y_test_iris_binary[:3]) > 0.6
        )
        outlier = np.ones(np.shape(self.x_test_iris[:3])) * 10.0
        # output for random points should be 0.5 (as classifier is uncertain)
        self.assertTrue(np.sum(self.classifier.predict(outlier).flatten() == 0.5) == 6.0)

    def test_predict_unc(self):
        outlier = np.ones(np.shape(self.x_test_iris[:3])) * (np.max(self.x_test_iris.flatten()) * 10.0)
        # uncertainty should increase as we go deeper into data
        self.assertTrue(
            np.mean(
                self.classifier.predict_uncertainty(outlier) > self.classifier.predict_uncertainty(self.x_test_iris[:3])
            )
            == 1.0
        )

    def test_loss_gradient(self):
        grads = self.classifier.loss_gradient(self.x_test_iris[0:1], self.y_test_iris_binary[0:1])
        # grads with given seed should be [[-2.25244234e-11 -5.63282695e-11  1.74214328e-11 -1.21877914e-11]]
        # we test roughly: amount of positive/negative and largest gradient
        self.assertTrue(np.sum(grads < 0.0) == 3.0)
        self.assertTrue(np.sum(grads > 0.0) == 1.0)
        self.assertTrue(np.argmax(grads) == 2)

    def test_class_gradient(self):
        grads = self.classifier.class_gradient(self.x_test_iris[0:1], int(self.y_test_iris_binary[0:1]))
        # grads with given seed should be [[[2.25244234e-11  5.63282695e-11 -1.74214328e-11  1.21877914e-11]]]
        # we test roughly: amount of positive/negative and largest gradient
        self.assertTrue(np.sum(grads < 0.0) == 1.0)
        self.assertTrue(np.sum(grads > 0.0) == 3.0)
        self.assertTrue(np.argmax(grads) == 1)


if __name__ == "__main__":
    unittest.main()
