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
import logging

import numpy as np
import pytest

from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.fixture()
def fix_get_mnist_subset(get_mnist_dataset):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_mnist_dataset
    n_train = 10
    n_test = 10
    yield x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test]


@pytest.mark.only_with_platform("pytorch")
def test_eot_zoom_blur_pytorch(art_warning, fix_get_mnist_subset):
    try:
        import torch
        from art.preprocessing.expectation_over_transformation.natural_corruptions.zoom_blur.pytorch import (
            EoTZoomBlurPyTorch,
        )

        x_train_mnist, y_train_mnist, _, _ = fix_get_mnist_subset
        x_train_mnist = np.transpose(x_train_mnist, (0, 2, 3, 1))  # transpose to NHWC

        nb_samples = 3

        eot = EoTZoomBlurPyTorch(nb_samples=nb_samples, zoom=(1.0, 1.5), clip_values=(0.0, 1.0))
        x_eot, y_eot = eot.forward(x=torch.from_numpy(x_train_mnist), y=torch.from_numpy(y_train_mnist))

        assert x_eot.shape[0] == nb_samples * x_train_mnist.shape[0]
        assert y_eot.shape[0] == nb_samples * y_train_mnist.shape[0]

        x_eot_expected = np.array(
            [
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                1.3863939e-04,
                2.1103114e-02,
                3.4391126e-01,
                8.7550414e-01,
                9.4271177e-01,
                9.0662521e-01,
                5.0556481e-01,
                1.4227462e-01,
                1.8304091e-02,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
            ]
        )

        np.testing.assert_almost_equal(x_eot.numpy()[0, 14, :, 0], x_eot_expected)

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("tensorflow2")
def test_eot_zoom_blur_tensorflow_v2(art_warning, fix_get_mnist_subset):
    try:
        from art.preprocessing.expectation_over_transformation.natural_corruptions.zoom_blur.tensorflow import (
            EoTZoomBlurTensorFlow,
        )

        x_train_mnist, y_train_mnist, _, _ = fix_get_mnist_subset

        nb_samples = 3

        eot = EoTZoomBlurTensorFlow(nb_samples=nb_samples, zoom=(1.0, 1.5), clip_values=(0.0, 1.0))
        x_eot, y_eot = eot.forward(x=x_train_mnist, y=y_train_mnist)

        assert x_eot.shape[0] == nb_samples * x_train_mnist.shape[0]
        assert y_eot.shape[0] == nb_samples * y_train_mnist.shape[0]

        x_eot_expected = np.array(
            [
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                1.3863939e-04,
                2.1103114e-02,
                3.4391126e-01,
                8.7550414e-01,
                9.4271177e-01,
                9.0662521e-01,
                5.0556481e-01,
                1.4227462e-01,
                1.8304091e-02,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
            ]
        )

        np.testing.assert_almost_equal(x_eot.numpy()[0, 14, :, 0], x_eot_expected)

    except ARTTestException as e:
        art_warning(e)
