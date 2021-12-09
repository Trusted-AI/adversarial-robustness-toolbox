# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2021
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

from art.estimators.classification.blackbox import BlackBoxClassifierNeuralNetwork, BlackBoxClassifier
from tests.utils import ARTTestException


def test_blackbox_existing_predictions(art_warning, get_mnist_dataset):
    try:
        _, (x_test, y_test) = get_mnist_dataset

        limited_x_test = x_test[:500]
        limited_y_test = y_test[:500]

        bb = BlackBoxClassifier((limited_x_test, limited_y_test), (28, 28, 1), 10, clip_values=(0, 255))
        assert np.array_equal(bb.predict(limited_x_test), limited_y_test)

        with pytest.raises(ValueError):
            bb.predict(x_test[:600])

    except ARTTestException as e:
        art_warning(e)


def test_blackbox_existing_predictions_fuzzy(art_warning):
    try:
        x = np.array([0, 3])
        fuzzy_x = np.array([0, 3.00001])
        y = np.array([[1, 0], [0, 1]])
        bb = BlackBoxClassifier((x, y), (1,), 2, fuzzy_float_compare=True)
        assert np.array_equal(bb.predict(fuzzy_x), y)
    except ARTTestException as e:
        art_warning(e)


def test_blackbox_nn_existing_predictions(art_warning, get_mnist_dataset):
    try:
        _, (x_test, y_test) = get_mnist_dataset

        limited_x_test = x_test[:500]
        limited_y_test = y_test[:500]

        bb = BlackBoxClassifierNeuralNetwork((limited_x_test, limited_y_test), (28, 28, 1), 10, clip_values=(0, 255))
        assert np.array_equal(bb.predict(limited_x_test), limited_y_test)

        with pytest.raises(ValueError):
            bb.predict(x_test[:600])

    except ARTTestException as e:
        art_warning(e)


def test_blackbox_nn_existing_predictions_fuzzy(art_warning):
    try:
        x = np.array([0, 3])
        fuzzy_x = np.array([0, 3.00001])
        y = np.array([[1, 0], [0, 1]])
        bb = BlackBoxClassifierNeuralNetwork((x, y), (1,), 2, fuzzy_float_compare=True)
        assert np.array_equal(bb.predict(fuzzy_x), y)
    except ARTTestException as e:
        art_warning(e)
