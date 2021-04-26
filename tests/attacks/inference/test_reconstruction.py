# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
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

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from art.attacks.inference.reconstruction import DatabaseReconstruction
from art.estimators.classification.scikitlearn import ScikitlearnGaussianNB, ScikitlearnLogisticRegression


logger = logging.getLogger(__name__)


def test_database_reconstruction_gaussian_nb(get_iris_dataset):
    (x_train_iris, y_train_iris), (x_test_iris, y_test_iris) = get_iris_dataset
    y_train_iris = np.array([np.argmax(y) for y in y_train_iris])
    y_test_iris = np.array([np.argmax(y) for y in y_test_iris])

    x_private = x_test_iris[0, :].reshape(1, -1)
    y_private = y_test_iris[0]

    x_input = np.vstack((x_train_iris, x_private))
    y_input = np.hstack((y_train_iris, y_private))

    nb_private = GaussianNB()
    nb_private.fit(x_input, y_input)
    estimator_private = ScikitlearnGaussianNB(model=nb_private)

    recon = DatabaseReconstruction(estimator=estimator_private)
    x_recon, y_recon = recon.reconstruct(x_train_iris, y_train_iris)

    assert x_recon is not None
    assert x_recon.shape == (1, 4)
    assert y_recon.shape == (1, 3)
    assert np.isclose(x_recon, x_private).all()
    assert np.argmax(y_recon, axis=1) == y_private


def test_database_reconstruction_logistic_regression(get_iris_dataset):
    (x_train_iris, y_train_iris), (x_test_iris, y_test_iris) = get_iris_dataset
    y_train_iris = np.array([np.argmax(y) for y in y_train_iris])
    y_test_iris = np.array([np.argmax(y) for y in y_test_iris])

    x_private = x_test_iris[0, :].reshape(1, -1)
    y_private = y_test_iris[0]

    x_input = np.vstack((x_train_iris, x_private))
    y_input = np.hstack((y_train_iris, y_private))

    nb_private = LogisticRegression()
    nb_private.fit(x_input, y_input)
    estimator_private = ScikitlearnLogisticRegression(model=nb_private)

    recon = DatabaseReconstruction(estimator=estimator_private)
    x_recon, y_recon = recon.reconstruct(x_train_iris, y_train_iris)

    assert x_recon is not None
    assert x_recon.shape == (1, 4)
    assert y_recon.shape == (1, 3)
    assert np.isclose(x_recon, x_private, rtol=0.05).all()
    assert np.argmax(y_recon, axis=1) == y_private
