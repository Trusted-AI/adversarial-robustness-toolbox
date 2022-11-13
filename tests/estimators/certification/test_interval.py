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

import torch
from torch import nn

from art.estimators.certification.interval import PytorchInterval
from art.estimators.certification.interval import IntervalConv2D, IntervalDenseLayer, IntervalDenseLayer


class SyntheticIntervalModel(nn.Module):
    def __init__(self, input_shape, output_channels, kernel_size, stride=1, bias=False, to_debug=True):
        super().__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.conv1 = IntervalConv2D(in_channels=input_shape[1],
                                    out_channels=output_channels,
                                    kernel_size=kernel_size,
                                    input_shape=input_shape,
                                    stride=stride,
                                    bias=bias,
                                    to_debug=to_debug)

        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Computes the forward pass though the neural network
        :param x: input data of the form [number of samples, interval, feature]
        :return:
        """

        x = self.conv1.concrete_forward(x)
        return x


def test_conv_single_channel_in_multi_out():
    """
    Check that the conversion works for a single input channel.
    """
    synthetic_data = torch.rand(1, 1, 25, 25)
    model = SyntheticIntervalModel(input_shape=synthetic_data.shape,
                                   output_channels=4,
                                   kernel_size=5)
    output_from_equivalent = model.forward(synthetic_data)
    output_from_conv = model.conv1.conv(synthetic_data)
    output_from_equivalent = torch.reshape(output_from_equivalent, output_from_conv.shape)
    assert torch.equal(output_from_equivalent, output_from_conv)