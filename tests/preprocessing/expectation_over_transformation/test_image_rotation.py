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
import torch

from art.preprocessing.expectation_over_transformation.image_rotation.tensorflow import EoTImageRotationTensorFlow
from art.preprocessing.expectation_over_transformation.image_rotation.pytorch import EoTImageRotationPyTorch
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

        from matplotlib import pyplot as plt

        plt.matshow(x_eot.numpy()[0, :, :, 0])
        plt.show()

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


@pytest.mark.only_with_platform("pytorch")
def test_eot_image_rotation_classification_pytorch(art_warning, fix_get_mnist_subset):
    try:
        x_train_mnist, y_train_mnist, _, _ = fix_get_mnist_subset

        x_train_mnist = torch.from_numpy(x_train_mnist)
        y_train_mnist = torch.from_numpy(y_train_mnist)

        nb_samples = 3

        eot = EoTImageRotationPyTorch(
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

        np.testing.assert_almost_equal(x_eot.numpy()[0, 0, 14, :], x_eot_expected)

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_eot_image_rotation_object_detection_pytorch(art_warning, fix_get_mnist_subset):
    try:
        x_train_mnist, _, _, _ = fix_get_mnist_subset

        x_train_mnist = torch.from_numpy(x_train_mnist[0:2])

        y_od = [
            {
                "boxes": torch.from_numpy(np.array([[1, 1, 3, 3]])),
                "labels": torch.from_numpy(np.ones((1,))),
                "scores": torch.from_numpy(np.ones((1,))),
            },
            {
                "boxes": torch.from_numpy(np.array([[15, 15, 18, 20]])),
                "labels": torch.from_numpy(np.ones((1,))),
                "scores": torch.from_numpy(np.ones((1,))),
            },
        ]

        nb_samples = 3

        eot = EoTImageRotationPyTorch(
            nb_samples=nb_samples, angles=[90.0], clip_values=(0.0, 1.0), label_type="object_detection"
        )
        x_eot, y_eot = eot.forward(x=x_train_mnist, y=y_od)

        assert x_eot.shape[0] == nb_samples * x_train_mnist.shape[0]

        x_eot_expected = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.07058824,
                0.99215686,
                0.99215686,
                0.99215686,
                0.8039216,
                0.3529412,
                0.74509805,
                0.99215686,
                0.94509804,
                0.31764707,
                0.0,
                0.0,
                0.0,
                0.0,
                0.5803922,
                0.99215686,
                0.99215686,
                0.7647059,
                0.04313726,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )

        np.testing.assert_almost_equal(x_eot.numpy()[0, 0, 14, :], x_eot_expected)
        np.testing.assert_almost_equal(y_eot[0]["boxes"], np.array([[1, 25, 3, 27]]))
        np.testing.assert_almost_equal(y_eot[nb_samples]["boxes"], np.array([[15, 10, 20, 13]]))

    except ARTTestException as e:
        art_warning(e)
