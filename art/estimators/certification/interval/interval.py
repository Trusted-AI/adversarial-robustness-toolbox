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
"""
This module implements Interval bound propagation based layers

| Paper link: https://ieeexplore.ieee.org/document/8418593
"""
from typing import List, Union, Tuple

import torch
import numpy as np


class IntervalDenseLayer(torch.nn.Module):
    """
    Class implementing a dense layer for the interval (box) domain.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.weight = torch.nn.Parameter(
            torch.normal(mean=torch.zeros(out_features, in_features), std=torch.ones(out_features, in_features))
        )
        self.bias = torch.nn.Parameter(torch.normal(mean=torch.zeros(out_features), std=torch.ones(out_features)))

    def __call__(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.forward(x)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Performs the forward pass of the dense layer in the interval (box) domain.

        :param x: interval representation of the datapoint.
        :return: output of the convolutional layer on x
        """
        center = (x[:, 1] + x[:, 0]) / 2
        radius = (x[:, 1] - x[:, 0]) / 2

        center = torch.matmul(center, torch.transpose(self.weight, 0, 1)) + self.bias
        radius = torch.matmul(radius, torch.abs(torch.transpose(self.weight, 0, 1)))

        center = torch.unsqueeze(center, dim=1)
        radius = torch.unsqueeze(radius, dim=1)

        return torch.cat([center - radius, center + radius], dim=1)

    def concrete_forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Performs the forward pass of the dense layer.

        :param x: concrete input to the convolutional layer.
        :return: output of the convolutional layer on x
        """
        return torch.matmul(x, torch.transpose(self.weight, 0, 1)) + self.bias


class IntervalConv2D(torch.nn.Module):
    """
    Class implementing a convolutional layer in the interval/box domain.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        input_shape,
        stride: Union[int, Tuple[int, int]] = 1,
        bias: bool = False,
        supplied_input_weights=None,
        supplied_input_bias=None,
        to_debug: bool = False,
    ):
        """
        Creates the equivalent dense weights for the specified convolutional layer.

        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: shape of the convolutional kernel
        :param bias: if to include a bias term
        :param supplied_input_weights: If to load in a pre-defined set of convolutional weights with the correct specification.
        :param supplied_input_bias: If to load in a pre-defined set of convolutional bias with the correct specification.
        :param to_debug: helper parameter to help with debugging.
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        super().__init__()
        self.conv_flat = torch.nn.Conv2d(
            in_channels=1,
            out_channels=out_channels * in_channels,
            kernel_size=kernel_size,
            bias=False,
            stride=stride,
        )
        self.bias = None

        if bias:
            self.conv_bias = torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                bias=True,
                stride=stride,
            )
            if self.conv_bias.bias is not None:
                self.bias = self.conv_bias.bias.data

        if to_debug:
            self.conv = torch.nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias, stride=stride
            )

            if isinstance(kernel_size, tuple):
                self.conv_flat.weight = torch.nn.Parameter(
                    torch.reshape(
                        torch.tensor(self.conv.weight.data.cpu().detach().numpy()),
                        (out_channels * in_channels, 1, kernel_size[0], kernel_size[1]),
                    )
                )
            else:
                self.conv_flat.weight = torch.nn.Parameter(
                    torch.reshape(
                        torch.tensor(self.conv.weight.data.cpu().detach().numpy()),
                        (out_channels * in_channels, 1, kernel_size, kernel_size),
                    )
                )
            if bias and self.conv.bias is not None:
                self.bias = self.conv.bias.data

        if supplied_input_weights is not None:
            if isinstance(kernel_size, tuple):
                self.conv_flat.weight = torch.nn.Parameter(
                    torch.reshape(
                        torch.tensor(supplied_input_weights),
                        (out_channels * in_channels, 1, kernel_size[0], kernel_size[1]),
                    )
                )
            else:
                self.conv_flat.weight = torch.nn.Parameter(
                    torch.reshape(
                        torch.tensor(supplied_input_weights), (out_channels * in_channels, 1, kernel_size, kernel_size)
                    )
                )

        if supplied_input_bias is not None:
            self.bias = torch.tensor(supplied_input_bias)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]

        self.output_height: int = 0
        self.output_width: int = 0

        self.dense_weights, self.bias = self.convert_to_dense()
        self.dense_weights = self.dense_weights.to(self.device)
        self.bias = self.bias.to(self.device)

    def convert_to_dense(self) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        Converts the initialised convolutional layer into an equivalent dense layer.

        This function was adapted from:
        https://github.com/deepmind/interval-bound-propagation/blob/217a14d12686e08ebb5cfea1f2748cce58a55913/interval_bound_propagation/src/layer_utils.py#L90

        Here, we adapt the tf1 functionality to work with pytorch.

        Original license:

        coding=utf-8

        Copyright 2019 The Interval Bound Propagation Authors.

        Licensed under the Apache License, Version 2.0 (the "License");
        you may not use this file except in compliance with the License.
        You may obtain a copy of the License at

             http://www.apache.org/licenses/LICENSE-2.0

        Unless required by applicable law or agreed to in writing, software
        distributed under the License is distributed on an "AS IS" BASIS,
        WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

        See the License for the specific language governing permissions and
        limitations under the License.

        :return: The dense weights and bias equivalent to a Conv layer.
        """

        diagonal_input = torch.reshape(
            torch.eye(self.input_height * self.input_width),
            shape=[self.input_height * self.input_width, 1, self.input_height, self.input_width],
        )
        conv = self.conv_flat(diagonal_input)
        self.output_height = int(conv.shape[2])
        self.output_width = int(conv.shape[3])

        # conv is of shape (input_height * input_width, out_channels * in_channels, output_height, output_width).
        # Reshape it to (input_height * input_width * output_channels,
        #                output_height * output_width * input_channels).

        weights = torch.reshape(
            conv,
            shape=(
                [
                    self.input_height * self.input_width,
                    self.out_channels,
                    self.in_channels,
                    self.output_height,
                    self.output_width,
                ]
            ),
        )
        weights = torch.permute(weights, (2, 0, 1, 3, 4))
        weights = torch.reshape(
            weights,
            shape=(
                [
                    self.input_height * self.input_width * self.in_channels,
                    self.output_height * self.output_width * self.out_channels,
                ]
            ),
        )

        if self.bias is not None:
            self.bias = torch.unsqueeze(self.bias, dim=-1)
            bias = self.bias.expand(-1, self.output_height * self.output_width)
            bias = bias.flatten()
        else:
            bias = None

        return torch.transpose(weights, 0, 1), bias

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Performs the forward pass of the convolutional layer in the interval (box) domain by using
        the equivalent dense representation.

        :param x: interval representation of the datapoint.
        :return: output of the convolutional layer on x
        """
        x = torch.reshape(x, (x.shape[0], 2, -1))

        center = (x[:, 1] + x[:, 0]) / 2
        radius = (x[:, 1] - x[:, 0]) / 2

        center = torch.matmul(center, torch.transpose(self.dense_weights, 0, 1)) + self.bias
        radius = torch.matmul(radius, torch.abs(torch.transpose(self.dense_weights, 0, 1)))

        center = torch.unsqueeze(center, dim=1)
        radius = torch.unsqueeze(radius, dim=1)

        x = torch.cat([center - radius, center + radius], dim=1)
        return x.reshape((-1, 2, self.out_channels, self.output_height, self.output_width))

    def concrete_forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Performs the forward pass using the equivalent dense representation of the convolutional layer.

        :param x: concrete input to the convolutional layer.
        :return: output of the convolutional layer on x
        """
        x = torch.reshape(x, (x.shape[0], -1))
        if self.bias is None:
            x = torch.matmul(x, torch.transpose(self.dense_weights, 0, 1))
        else:
            x = torch.matmul(x, torch.transpose(self.dense_weights, 0, 1)) + self.bias
        return x.reshape((-1, self.out_channels, self.output_height, self.output_width))


class IntervalFlatten(torch.nn.Module):
    """
    Layer to handle flattening on both interval and concrete data
    """

    def __init__(self):
        super().__init__()

    def __call__(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.concrete_forward(x)

    @staticmethod
    def forward(x: "torch.Tensor") -> "torch.Tensor":
        """
        Flattens the provided abstract input

        :param x: datapoint in the interval domain
        :return: Flattened input preserving the batch and bounds dimensions.
        """
        return torch.reshape(x, (x.shape[0], 2, -1))

    @staticmethod
    def concrete_forward(x: "torch.Tensor") -> "torch.Tensor":
        """
        Flattens the provided concrete input

        :param x: datapoint in the concrete domain
        :return: Flattened input preserving the batch dimension.
        """
        return torch.reshape(x, (x.shape[0], -1))


class IntervalReLU(torch.nn.Module):
    """
    ReLU activation on both interval and concrete data
    """

    def __init__(self):
        super().__init__()
        self.activation = torch.nn.ReLU()

    def __call__(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.forward(x)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Abstract pass through the ReLU function

        :param x: abstract input to the activation function.
        :return: abstract outputs from the ReLU.
        """
        return self.activation(x)

    def concrete_forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Concrete pass through the ReLU function

        :param x: concrete input to the activation function.
        :return: concrete outputs from the ReLU.
        """
        return self.activation(x)


class IntervalBounds:
    """
    Class providing functionality for computing operations related to interval bounds
    """

    def __init__(self):
        pass

    @staticmethod
    def certify(preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Check if the data has been certifiably classified correct.

        :param preds: interval predictions
        :param labels: associated labels (not one-hot encoded).
        :return: array of True/False if predictions are certifiable
        """

        cert_bounds = np.copy(preds[:, 1])  # Take the upper bounds of all the predictions
        for i, label in enumerate(labels):
            cert_bounds[i, label] = preds[i, 0, label]  # Replace the correct prediction with its lower bound
        return np.argmax(cert_bounds, axis=1) == labels

    @staticmethod
    def concrete_to_interval(x: np.ndarray, bounds: Union[float, List[float], np.ndarray], limits=None):
        """
        Helper function converts a datapoint it into its interval representation

        :param x: input datapoint of shape [batch size, feature_1, feature_2, ...]
        :param bounds: Either a scalar to apply to the whole datapoint, or an array of [2, feature_1, feature_2]
        where bounds[0] are the lower bounds and bounds[1] are the upper bound
        :param limits: if to clip to a given range.

        :return: Data of the form [batch_size, 2, feature_1, feature_2, ...]
        where [batch_size, 0, x.shape] are the lower bounds and
        [batch_size, 1, x.shape] are the upper bounds.
        """

        x = np.expand_dims(x, axis=1)

        if isinstance(bounds, float):
            up_x = x + bounds
            lb_x = x - bounds
        elif isinstance(bounds, list):
            up_x = x + bounds[1]
            lb_x = x - bounds[0]
        elif isinstance(bounds, np.ndarray):
            pass
            # TODO: Implement
        else:
            raise ValueError("Bounds must be a float, list, or np array.")

        final_batched_input = np.concatenate((lb_x, up_x), axis=1)

        if limits is not None:
            return np.clip(final_batched_input, limits[0], limits[1])

        return final_batched_input
