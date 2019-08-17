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

import numpy as np
import GPy

from art.classifiers import GPyGaussianProcessClassifier
from art.utils import load_dataset

logger = logging.getLogger('testLogger')
np.random.seed(seed=1234)


class TestGPyGaussianProcessClassifier(unittest.TestCase):

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
        (_, _), (x_test, y_test) = self.iris
        # predictions should be correct
        self.assertTrue(
            np.mean((self.classifier.predict(x_test[:3])[:, 0] > 0.5) == y_test[:3]) > 0.6)
        outlier = np.ones(np.shape(x_test[:3])) * 10.0
        # output for random points should be 0.5 (as classifier is uncertain)
        self.assertTrue(np.sum(self.classifier.predict(
            outlier).flatten() == 0.5) == 6.0)

    def test_predict_unc(self):
        (_, _), (x_test, y_test) = self.iris
        outlier = np.ones(np.shape(x_test[:3])) * (np.max(x_test.flatten()) * 10.0)
        # uncertainty should increase as we go deeper into data
        self.assertTrue(np.mean(self.classifier.predict_uncertainty(
            outlier) > self.classifier.predict_uncertainty(x_test[:3])) == 1.0)

    def test_loss_gradient(self):
        (_, _), (x_test, y_test) = self.iris
        grads = self.classifier.loss_gradient(x_test[0:1], y_test[0:1])
        # grads with given seed should be [[-2.25244234e-11 -5.63282695e-11  1.74214328e-11 -1.21877914e-11]]
        # we test roughly: amount of positive/negative and largest gradient
        self.assertTrue(np.sum(grads < 0.0) == 3.0)
        self.assertTrue(np.sum(grads > 0.0) == 1.0)
        self.assertTrue(np.argmax(grads) == 2)

    def test_class_gradient(self):
        (_, _), (x_test, y_test) = self.iris
        grads = self.classifier.class_gradient(x_test[0:1], int(y_test[0:1]))
        # grads with given seed should be [[[2.25244234e-11  5.63282695e-11 -1.74214328e-11  1.21877914e-11]]]
        # we test roughly: amount of positive/negative and largest gradient
        self.assertTrue(np.sum(grads < 0.0) == 1.0)
        self.assertTrue(np.sum(grads > 0.0) == 3.0)
        self.assertTrue(np.argmax(grads) == 1)
