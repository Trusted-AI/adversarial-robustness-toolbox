# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2023
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

from art.utils import load_dataset
from art.estimators.certification.smoothed_vision_transformers import PyTorchSmoothedViT
from tests.utils import ARTTestException


@pytest.fixture()
def fix_get_mnist_data():
    """
    Get the first 128 samples of the mnist test set with channels first format

    :return: First 128 sample/label pairs of the MNIST test dataset.
    """
    nb_test = 128

    (_, _), (x_test, y_test), _, _ = load_dataset("mnist")
    x_test = np.squeeze(x_test).astype(np.float32)
    x_test = np.expand_dims(x_test, axis=1)
    y_test = np.argmax(y_test, axis=1)

    x_test, y_test = x_test[:nb_test], y_test[:nb_test]
    return x_test, y_test


@pytest.fixture()
def fix_get_cifar10_data():
    """
    Get the first 128 samples of the cifar10 test set

    :return: First 128 sample/label pairs of the cifar10 test dataset.
    """
    nb_test = 128

    (_, _), (x_test, y_test), _, _ = load_dataset("cifar10")
    y_test = np.argmax(y_test, axis=1)
    x_test, y_test = x_test[:nb_test], y_test[:nb_test]
    x_test = np.transpose(x_test, (0, 3, 1, 2))  # return in channels first format
    return x_test.astype(np.float32), y_test


@pytest.mark.skip_framework("mxnet", "non_dl_frameworks", "tensorflow1", "keras", "kerastf", "tensorflow2")
def test_ablation(art_warning, fix_get_mnist_data, fix_get_cifar10_data):
    """
    Check that the ablation is being performed correctly
    """
    from art.estimators.certification.smoothed_vision_transformers.smooth_vit import ColumnAblator
    import torch
    try:
        cifar_data = fix_get_cifar10_data[0]
        cifar_labels = fix_get_cifar10_data[1]

        col_ablator = ColumnAblator(ablation_size=4,
                                    channels_first=True,
                                    to_reshape=False,  # do not upsample initially
                                    original_shape=(3, 32, 32),
                                    output_shape=(3, 224, 224))

        cifar_data = torch.from_numpy(cifar_data)
        # check that the ablation functioned when in the middle of the image
        ablated = col_ablator.forward(cifar_data, column_pos=10)

        assert ablated.shape[1] == 4
        assert torch.sum(ablated[:, :, :, 0:10]) == 0
        assert torch.sum(ablated[:, :, :, 10:14]) > 0
        assert torch.sum(ablated[:, :, :, 14:]) == 0

        # check that the ablation wraps when on the edge of the image
        ablated = col_ablator.forward(cifar_data, column_pos=30)

        assert ablated.shape[1] == 4
        assert torch.sum(ablated[:, :, :, 30:]) > 0
        assert torch.sum(ablated[:, :, :, 2:30]) == 0
        assert torch.sum(ablated[:, :, :, :2]) > 0

        # check that upsampling works as expected
        col_ablator = ColumnAblator(ablation_size=4,
                                    channels_first=True,
                                    to_reshape=True,
                                    original_shape=(3, 32, 32),
                                    output_shape=(3, 224, 224))

        ablated = col_ablator.forward(cifar_data, column_pos=10)

        assert ablated.shape[1] == 4
        assert torch.sum(ablated[:, :, :, :10*7]) == 0
        assert torch.sum(ablated[:, :, :, 10*7:14*7]) > 0
        assert torch.sum(ablated[:, :, :, 14*7:]) == 0

        # check that the ablation wraps when on the edge of the image
        ablated = col_ablator.forward(cifar_data, column_pos=30)

        assert ablated.shape[1] == 4
        assert torch.sum(ablated[:, :, :, 30*7:]) > 0
        assert torch.sum(ablated[:, :, :, 2*7:30*7]) == 0
        assert torch.sum(ablated[:, :, :, :2*7]) > 0

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("mxnet", "non_dl_frameworks", "tensorflow1", "keras", "kerastf", "tensorflow2")
def test_pytorch_training(art_warning, fix_get_mnist_data, fix_get_cifar10_data):
    """
    Check that the training loop for pytorch does not result in errors
    """
    import torch
    try:
        cifar_data = fix_get_cifar10_data[0][0:50]
        cifar_labels = fix_get_cifar10_data[1][0:50]

        art_model = PyTorchSmoothedViT(model='vit_small_patch16_224',
                                       loss=torch.nn.CrossEntropyLoss(),
                                       optimizer=torch.optim.SGD,
                                       optimizer_params={"lr": 0.01},
                                       input_shape=(3, 32, 32),
                                       nb_classes=10,
                                       ablation_type='column',
                                       ablation_size=4,
                                       threshold=0.01,
                                       load_pretrained=True)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(art_model.optimizer, milestones=[1], gamma=0.1)
        art_model.fit(cifar_data, cifar_labels, nb_epochs=2, update_batchnorm=True, scheduler=scheduler)

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("mxnet", "non_dl_frameworks", "tensorflow1", "keras", "kerastf", "tensorflow2")
def test_certification_function(art_warning, fix_get_mnist_data, fix_get_cifar10_data):
    """
    Check that the training loop for pytorch does not result in errors
    """
    """
    Check that the ablation is being performed correctly
    """
    from art.estimators.certification.smoothed_vision_transformers.smooth_vit import ColumnAblator
    import torch

    try:
        col_ablator = ColumnAblator(ablation_size=4,
                                    channels_first=True,
                                    to_reshape=True,  # do not upsample initially
                                    original_shape=(3, 32, 32),
                                    output_shape=(3, 224, 224))
        pred_counts = torch.from_numpy(np.asarray([[20, 5, 1], [10, 5, 1], [1, 16, 1]]))
        cert, cert_and_correct, top_predicted_class = col_ablator.certify(pred_counts=pred_counts,
                                                                          size_to_certify=4,
                                                                          label=0,)
        assert torch.equal(cert, torch.tensor([True, False, True]))
        assert torch.equal(cert_and_correct, torch.tensor([True, False, False]))
    except ARTTestException as e:
        art_warning(e)

