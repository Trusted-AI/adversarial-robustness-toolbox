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
from typing import List, Union, Tuple, Optional

import torch
import numpy as np


class PyTorchIntervalDense(torch.nn.Module):
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


class PyTorchIntervalConv2D(torch.nn.Module):
    """
    Class implementing a convolutional layer in the interval/box domain.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        input_shape: Tuple[int, ...],
        device: Union[str, "torch.device"],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 0,
        bias: bool = True,
        supplied_input_weights: Union[None, "torch.Tensor"] = None,
        supplied_input_bias: Union[None, "torch.Tensor"] = None,
        to_debug: bool = False,
    ):
        """
        Creates the equivalent dense weights for the specified convolutional layer.

        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Shape of the convolutional kernel
        :param device: Device to put the weights onto
        :param stride: The convolution's stride
        :param padding: Size of padding to use
        :param dilation: Dilation to apply to the convolution
        :param bias: If to include a bias term
        :param supplied_input_weights: Load in a pre-defined set of convolutional weights with the correct specification
        :param supplied_input_bias: Load in a pre-defined set of convolutional bias with the correct specification
        :param to_debug: Helper parameter to help with debugging.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.stride = stride
        self.device = device
        self.include_bias = bias
        self.cnn: Optional["torch.nn.Conv2d"] = None

        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=1,
            out_channels=out_channels * in_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
            stride=stride,
        ).to(device)
        self.bias_to_grad = None

        if bias:
            self.bias_to_grad = torch.nn.Parameter(torch.rand(out_channels).to(device))

        if to_debug:
            self.conv_debug = torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                bias=bias,
                stride=stride,
                padding=padding,
                dilation=dilation,
            ).to(device)

            if isinstance(kernel_size, tuple):
                self.conv.weight = torch.nn.Parameter(
                    torch.reshape(
                        torch.tensor(self.conv_debug.weight.data.cpu().detach().numpy()),
                        (out_channels * in_channels, 1, kernel_size[0], kernel_size[1]),
                    ).to(device)
                )
            else:
                self.conv.weight = torch.nn.Parameter(
                    torch.reshape(
                        torch.tensor(self.conv_debug.weight.data.cpu().detach().numpy()),
                        (out_channels * in_channels, 1, kernel_size, kernel_size),
                    ).to(device)
                )
            if bias and self.conv_debug.bias is not None:
                self.bias_to_grad = torch.nn.Parameter(
                    torch.tensor(self.conv_debug.bias.data.cpu().detach().numpy()).to(device)
                )

        if supplied_input_weights is not None:
            if isinstance(kernel_size, tuple):
                self.conv.weight = torch.nn.Parameter(
                    torch.reshape(
                        supplied_input_weights,
                        (out_channels * in_channels, 1, kernel_size[0], kernel_size[1]),
                    )
                )
            else:
                self.conv.weight = torch.nn.Parameter(
                    torch.reshape(supplied_input_weights, (out_channels * in_channels, 1, kernel_size, kernel_size))
                )

        if supplied_input_bias is not None:
            self.bias_to_grad = torch.nn.Parameter(supplied_input_bias.to(device))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]

        self.output_height: int = 0
        self.output_width: int = 0

        self.dense_weights, self.bias = self.convert_to_dense(device)

        self.dense_weights = self.dense_weights.to(device)
        if self.bias is not None:
            self.bias = self.bias.to(device)

    def re_convert(self, device: Union[str, "torch.device"]) -> None:
        """
        Re converts the weights into a dense equivalent layer.
        Must be called after every backwards if multiple gradients wish to be taken (like for crafting pgd).
        """
        self.dense_weights, self.bias = self.convert_to_dense(device)
        self.dense_weights = self.dense_weights.to(device)
        if self.bias is not None:
            self.bias = self.bias.to(device)

    def convert_to_dense(self, device: Union[str, "torch.device"]) -> Tuple["torch.Tensor", "torch.Tensor"]:
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
        ).to(device)
        conv_output = self.conv(diagonal_input)
        self.output_height = int(conv_output.shape[2])
        self.output_width = int(conv_output.shape[3])

        # conv is of shape (input_height * input_width, out_channels * in_channels, output_height, output_width).
        # Reshape it to (input_height * input_width * output_channels,
        #                output_height * output_width * input_channels).

        weights = torch.reshape(
            conv_output,
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

        if self.bias_to_grad is not None:
            self.bias = torch.unsqueeze(self.bias_to_grad, dim=-1)
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

    def conv_forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Method for efficiently interfacing with adversarial attacks.

        Backpropagating through concrete_forward is too slow if adversarial attacks need to be generated on-the fly
        or require a large amount of iterations.

        This method will create a regular conv layer with the right parameters to use.

        :param x: concrete input to the convolutional layer.
        :return: output of the convolutional layer on x
        """
        if self.cnn is None:
            self.cnn = torch.nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                bias=self.include_bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            ).to(self.device)
        if isinstance(self.kernel_size, tuple):
            self.cnn.weight.data = torch.reshape(
                torch.tensor(self.conv.weight.data.cpu().detach().numpy()),
                (self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]),
            ).to(self.device)
        else:
            self.cnn.weight.data = torch.reshape(
                torch.tensor(self.conv.weight.data.cpu().detach().numpy()),
                (self.out_channels, self.in_channels, self.kernel_size, self.kernel_size),
            ).to(self.device)
        if self.cnn.bias is not None and self.bias_to_grad is not None:
            self.cnn.bias.data = torch.tensor(self.bias_to_grad.data.cpu().detach().numpy()).to(self.device)

        if self.cnn is not None:
            return self.cnn(x)
        raise ValueError("The convolutional layer for attack mode was not created properly")


class PyTorchIntervalFlatten(torch.nn.Module):
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


class PyTorchIntervalReLU(torch.nn.Module):
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


class PyTorchIntervalBounds:
    """
    Class providing functionality for computing operations related to interval bounds
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore

    @staticmethod
    def certify(preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Check if the data has been certifiably classified correct.

        :param preds: interval predictions
        :param labels: associated labels (not one-hot encoded).
        :return: array of True or False if predictions are certifiable
        """

        cert_bounds = np.copy(preds[:, 1])  # Take the upper bounds of all the predictions
        for i, label in enumerate(labels):
            cert_bounds[i, label] = preds[i, 0, label]  # Replace the correct prediction with its lower bound
        return np.argmax(cert_bounds, axis=1) == labels

    @staticmethod
    def concrete_to_interval(
        x: np.ndarray,
        bounds: Union[float, List[float], np.ndarray],
        limits: Optional[Union[List[float], np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Helper function converts a datapoint it into its interval representation

        :param x: input datapoint of shape [batch size, feature_1, feature_2, ...]
        :param bounds: Either a scalar to apply to the whole datapoint, or an array of [2, feature_1, feature_2]
                       where bounds[0] are the lower bounds and bounds[1] are the upper bound
        :param limits: if to clip to a range with limits[0] being the lower bounds and limits[1] being upper bounds.
        :return: Data of the form [batch_size, 2, feature_1, feature_2, ...]
                 where [batch_size, 0, x.shape] are the lower bounds and [batch_size, 1, x.shape] are the upper bounds.
        """

        x = np.expand_dims(x, axis=1)

        if isinstance(bounds, float):
            up_x = x + bounds
            lb_x = x - bounds
        elif isinstance(bounds, (list, np.ndarray)):
            up_x = x + bounds[1]
            lb_x = x - bounds[0]
        else:
            raise ValueError("Bounds must be a float, list, or numpy array.")

        final_batched_input = np.concatenate((lb_x, up_x), axis=1)

        if limits is not None:
            return np.clip(final_batched_input, limits[0], limits[1])

        return final_batched_input
