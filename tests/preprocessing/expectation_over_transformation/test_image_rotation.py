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
import logging

import numpy as np
import pytest

from art.preprocessing.expectation_over_transformation.image_rotation.tensorflow import EoTImageRotationTensorFlow
from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.fixture()
def fix_get_mnist_subset(get_mnist_dataset):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_mnist_dataset
    n_train = 10
    n_test = 10
    yield x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test]


@pytest.mark.only_with_platform("tensorflow2")
def test_eot_image_rotation_classification_tensorflow_v2(art_warning, fix_get_mnist_subset):
    try:
        x_train_mnist, y_train_mnist, _, _ = fix_get_mnist_subset

        nb_samples = 3

        eot = EoTImageRotationTensorFlow(
            nb_samples=nb_samples, angles=(45.0, 45.0), clip_values=(0.0, 1.0), label_type="classification"
        )
        x_eot, y_eot = eot.forward(x=x_train_mnist, y=y_train_mnist)

        assert x_eot.shape[0] == nb_samples * x_train_mnist.shape[0]
        assert y_eot.shape[0] == nb_samples * y_train_mnist.shape[0]

        x_eot_expected = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.07058824,
                0.3137255,
                0.6117647,
                0.05490196,
                0.0,
                0.54509807,
                0.04313726,
                0.13725491,
                0.31764707,
                0.9411765,
                0.1764706,
                0.0627451,
                0.3647059,
                0.0,
                0.99215686,
                0.99215686,
                0.98039216,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )

        np.testing.assert_almost_equal(x_eot.numpy()[0, 14, :, 0], x_eot_expected)

    except ARTTestException as e:
        art_warning(e)
