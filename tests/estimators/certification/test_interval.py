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

import torch
import numpy as np

from art.estimators.certification.interval import PytorchInterval
from art.estimators.certification.interval import IntervalConv2D, IntervalDenseLayer, IntervalDenseLayer

from art.utils import load_dataset
from tests.utils import ARTTestException
from tests.utils import get_image_classifier_pt, get_cifar10_image_classifier_pt


class SyntheticIntervalModel(torch.nn.Module):
    def __init__(self, input_shape, output_channels, kernel_size, stride=1, bias=False, to_debug=True):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.conv1 = IntervalConv2D(in_channels=input_shape[1],
                                    out_channels=output_channels,
                                    kernel_size=kernel_size,
                                    input_shape=input_shape,
                                    stride=stride,
                                    bias=bias,
                                    to_debug=to_debug,
                                    device=self.device)

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        """
        Computes the forward pass though the neural network
        :param x: input data of the form [number of samples, interval, feature]
        :return:
        """

        x = self.conv1.concrete_forward(x)
        return x


@pytest.fixture()
def fix_get_mnist_data():
    """
    Get the first 100 samples of the mnist test set with channels first format
    :return: First 100 sample/label pairs of the MNIST test dataset.
    """
    nb_test = 100

    (_, _), (x_test, y_test), _, _ = load_dataset("mnist")
    x_test = np.squeeze(x_test)
    x_test = np.expand_dims(x_test, axis=1)
    y_test = np.argmax(y_test, axis=1)

    x_test, y_test = x_test[:nb_test], y_test[:nb_test]

    return x_test, y_test


@pytest.mark.skip_framework("mxnet", "non_dl_frameworks", "tensorflow1", "keras", "kerastf", "tensorflow2")
def test_conv_single_channel_in_multi_out(art_warning):
    """
    Check that the conversion works for a single input channel.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    synthetic_data = torch.rand(32, 1, 25, 25).to(device)
    model = SyntheticIntervalModel(input_shape=synthetic_data.shape,
                                   output_channels=4,
                                   kernel_size=5)
    output_from_equivalent = model.forward(synthetic_data)
    output_from_conv = model.conv1.conv(synthetic_data)
    output_from_equivalent = torch.reshape(output_from_equivalent, output_from_conv.shape)
    try:
        torch.equal(output_from_equivalent, output_from_conv)
    except ARTTestException as e:
        art_warning(e)


def test_conv_multi_channel_in_single_out():
    """
    Check that the conversion works for a single input channel.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    synthetic_data = torch.rand(32, 3, 25, 25).to(device)
    model = SyntheticIntervalModel(input_shape=synthetic_data.shape,
                                   output_channels=1,
                                   kernel_size=5)
    output_from_equivalent = model.forward(synthetic_data)
    output_from_conv = model.conv1.conv(synthetic_data)

    output_from_equivalent = output_from_equivalent.flatten()
    output_from_conv = output_from_conv.flatten()

    assert torch.allclose(output_from_equivalent, output_from_conv, atol=1e-05)


def test_conv_multi_channel_in_multi_out():
    """
    Check that the conversion works for a single input channel.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    synthetic_data = torch.rand(32, 3, 25, 25).to(device)
    model = SyntheticIntervalModel(input_shape=synthetic_data.shape,
                                   output_channels=12,
                                   kernel_size=5)
    output_from_equivalent = model.forward(synthetic_data)
    output_from_conv = model.conv1.conv(synthetic_data)

    assert torch.allclose(output_from_equivalent, output_from_conv, atol=1e-05)


def test_conv_layer_multi_channel_in_multi_out_with_stride():
    """
    Check that the conversion works for a single input channel.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    synthetic_data = torch.rand(32, 3, 25, 25).to(device)
    model = SyntheticIntervalModel(input_shape=synthetic_data.shape,
                                   output_channels=12,
                                   kernel_size=5,
                                   stride=2)

    output_from_equivalent = model.forward(synthetic_data)
    output_from_conv = model.conv1.conv(synthetic_data)

    assert torch.allclose(output_from_equivalent, output_from_conv, atol=1e-05)


def test_conv_layer_multi_channel_in_multi_out_with_stride_and_bias():
    """
    Check that the conversion works for a single input channel.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    synthetic_data = torch.rand(32, 3, 25, 25).to(device)
    model = SyntheticIntervalModel(input_shape=synthetic_data.shape,
                                   output_channels=12,
                                   kernel_size=5,
                                   bias=True,
                                   stride=1)
    output_from_equivalent = model.forward(synthetic_data)
    output_from_conv = model.conv1.conv(synthetic_data)

    assert torch.allclose(output_from_equivalent, output_from_conv, atol=1e-05)


def test_conv_layer_grads():

    output_channels = 12
    input_channels = 3

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    synthetic_data = torch.rand(32, input_channels, 25, 25).to(device)
    model = SyntheticIntervalModel(input_shape=synthetic_data.shape,
                                   output_channels=output_channels,
                                   kernel_size=5,
                                   bias=True,
                                   stride=1)
    output_from_equivalent = model.forward(synthetic_data)
    output_from_conv = model.conv1.conv(synthetic_data)
    target = torch.rand(size=output_from_equivalent.shape).to(device)

    loss = torch.sum(output_from_equivalent - target)
    loss.backward()

    equivalent_grads = model.conv1.conv_flat.weight.grad
    equivalent_grads = torch.reshape(equivalent_grads, shape=(output_channels, input_channels, 5, 5)).detach().clone()

    model.zero_grad()
    loss = torch.sum(output_from_conv - target)
    loss.backward()

    assert torch.allclose(equivalent_grads, model.conv1.conv.weight.grad, atol=1e-05)


@pytest.mark.skip_framework("mxnet", "non_dl_frameworks", "tensorflow1", "keras", "kerastf", "tensorflow2")
def test_mnist_certification(art_warning, fix_get_mnist_data):
    bound = 0.05

    ptc = get_image_classifier_pt(from_logits=True, use_maxpool=False)

    box_model = PytorchInterval(
        model=ptc.model, clip_values=(0, 1), loss=torch.nn.CrossEntropyLoss(), input_shape=(1, 28, 28), nb_classes=10
    )
