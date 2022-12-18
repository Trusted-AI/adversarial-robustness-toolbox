# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2022
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
import pytest

import numpy as np

from art.estimators.regression import BlackBoxRegressor
from art.config import ART_NUMPY_DTYPE
from tests.utils import ARTTestException


def mean_absolute_error(y_true, y_pred):
    return np.abs(y_pred - y_true)


def test_blackbox_existing_predictions(art_warning, get_diabetes_dataset):
    try:
        (x_train, _), (x_test, y_test) = get_diabetes_dataset

        bb = BlackBoxRegressor(
            (x_test.astype(ART_NUMPY_DTYPE), y_test),
            (10,),
        )
        pred = bb.predict(x_test.astype(ART_NUMPY_DTYPE))
        assert np.array_equal(pred, y_test.astype(ART_NUMPY_DTYPE))
        assert np.count_nonzero(bb.compute_loss(x_test, y_test)) == 0
        assert np.count_nonzero(bb.compute_loss_from_predictions(pred, y_test)) == 0

        with pytest.raises(ValueError):
            bb.predict(x_train)

    except ARTTestException as e:
        art_warning(e)


def test_blackbox_existing_predictions_fuzzy(art_warning):
    try:
        x = np.array([0, 3])
        fuzzy_x = np.array([0, 3.00001])
        y = np.array([2.1, 1.8])
        bb = BlackBoxRegressor((x, y), (1,), fuzzy_float_compare=True)
        assert np.array_equal(bb.predict(fuzzy_x), y.astype(ART_NUMPY_DTYPE))

    except ARTTestException as e:
        art_warning(e)


def test_blackbox_existing_predictions_custom_loss(art_warning, get_diabetes_dataset):
    try:
        (_, _), (x_test, y_test) = get_diabetes_dataset

        bb = BlackBoxRegressor((x_test.astype(ART_NUMPY_DTYPE), y_test), (10,), loss_fn=mean_absolute_error)
        pred = bb.predict(x_test.astype(ART_NUMPY_DTYPE))
        assert np.array_equal(pred, y_test.astype(ART_NUMPY_DTYPE))
        assert np.count_nonzero(bb.compute_loss(x_test, y_test)) == 0
        assert np.count_nonzero(bb.compute_loss_from_predictions(pred, y_test)) == 0

    except ARTTestException as e:
        art_warning(e)


def test_blackbox_predict_fn(art_warning, get_diabetes_dataset):
    try:
        from sklearn import linear_model

        (x_train, y_train), (x_test, y_test) = get_diabetes_dataset
        regr_model = linear_model.LinearRegression()
        regr_model.fit(x_train, y_train)

        bb = BlackBoxRegressor(regr_model.predict, (10,))
        pred = bb.predict(x_test)
        assert abs(y_test[0] - pred[0]) < 50
        assert abs(bb.compute_loss(x_test, y_test)[0] - 2000) < 100
        assert abs(bb.compute_loss_from_predictions(pred, y_test)[0] - 2000) < 100

        bbc = bb.get_classifier([50, 100, 200])
        pred_c = bbc.predict(x_test)
        assert pred_c.shape[1] == 4

    except ARTTestException as e:
        art_warning(e)


def test_blackbox_predict_fn_custom_loss(art_warning, get_diabetes_dataset):
    try:
        from sklearn import linear_model

        (x_train, y_train), (x_test, y_test) = get_diabetes_dataset
        regr_model = linear_model.LinearRegression()
        regr_model.fit(x_train, y_train)

        bb = BlackBoxRegressor(regr_model.predict, (10,), loss_fn=mean_absolute_error)
        pred = bb.predict(x_test)
        assert abs(y_test[0] - pred[0]) < 50
        assert abs(bb.compute_loss(x_test, y_test)[0] - 45) < 5
        assert abs(bb.compute_loss_from_predictions(pred, y_test)[0] - 45) < 5

    except ARTTestException as e:
        art_warning(e)
