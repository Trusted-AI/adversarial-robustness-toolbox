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
This module implements DeepZ proposed in Fast and Effective Robustness Certification.

| Paper link: https://papers.nips.cc/paper/2018/file/f2f446980d8e971ef3da97af089481c3-Paper.pdf
"""
from typing import Tuple, Union

import numpy as np
import torch


class ZonoDenseLayer(torch.nn.Module):
    """
    Class implementing a dense layer on a zonotope.
    Bias is only added to the zeroth term.
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
        Abstract forward pass through the dense layer.

        :param x: input zonotope to the dense layer.
        :return: zonotope after being pushed through the dense layer.
        """
        x = self.zonotope_matmul(x)
        x = self.zonotope_add(x)
        return x

    def concrete_forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Concrete forward pass through the dense layer.

        :param x: concrete input to the dense layer.
        :return: concrete dense layer outputs.
        """
        x = torch.matmul(x, torch.transpose(self.weight, 0, 1)) + self.bias
        return x

    def zonotope_matmul(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Matrix multiplication for dense layer.

        :param x: input to the dense layer.
        :return: zonotope after weight multiplication.
        """
        return torch.matmul(x, torch.transpose(self.weight, 0, 1))

    def zonotope_add(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Modification required compared to the normal torch dense layer.
        The bias is added only to the central zonotope term and not the error terms.

        :param x: zonotope input to have the bias added.
        :return: zonotope with the bias added to the central (first) term.
        """
        x[0] = x[0] + self.bias
        return x


class ZonoBounds:
    """
    Class providing functionality for computing operations related to getting lower and upper bounds on zonotopes.
    """

    def __init__(self):
        pass

    @staticmethod
    def compute_lb(cent: "torch.Tensor", eps: "torch.Tensor") -> "torch.Tensor":
        """
        Compute the lower bound on a feature.

        :param eps: tensor with the eps terms
        :param cent: tensor with the zero zonotope term
        :return: lower bound on the given feature
        """
        return torch.sum(-1 * torch.abs(eps), dim=0) + cent

    @staticmethod
    def compute_ub(cent: "torch.Tensor", eps: "torch.Tensor") -> "torch.Tensor":
        """
        Compute the upper bound on a feature.

        :param eps: tensor with the eps terms
        :param cent: tensor with the zero zonotope term
        :return: upper bound on the given feature
        """
        return torch.sum(torch.abs(eps), dim=0) + cent

    @staticmethod
    def certify_via_subtraction(
        predicted_class: int, class_to_consider: int, cent: np.ndarray, eps: np.ndarray
    ) -> bool:
        """
        To perform the certification we subtract the zonotope of "class_to_consider"
        from the zonotope of the predicted class.

        :param predicted_class: class the model predicted.
        :param class_to_consider: class to check if the model could have classified to it.
        :param cent: center/zeroth zonotope term.
        :param eps: zonotope error terms.
        :return: True/False if the point has been certified
        """
        diff_in_bias = cent[class_to_consider] - cent[predicted_class]
        diff_in_eps = eps[:, class_to_consider] - eps[:, predicted_class]
        lbs = np.sum(-1 * np.abs(diff_in_eps)) + diff_in_bias
        ubs = np.sum(np.abs(diff_in_eps)) + diff_in_bias

        return np.sign(lbs) < 0 and np.sign(ubs) < 0

    def zonotope_get_bounds(self, cent: "torch.Tensor", eps: "torch.Tensor") -> Tuple[list, list]:
        """
        Compute the upper and lower bounds for the final zonotopes

        :param cent: center/zeroth zonotope term.
        :param eps: zonotope error terms.
        :return: lists with the upper and lower bounds.
        """
        upper_bounds_output = []
        lower_bounds_output = []

        for j in range(cent.shape[0]):
            # compute lower bounds
            lbs = self.compute_lb(eps=eps[:, j], cent=cent[j])
            # compute upper bounds
            ubs = self.compute_ub(eps=eps[:, j], cent=cent[j])

            upper_bounds_output.append(ubs)
            lower_bounds_output.append(lbs)
        return upper_bounds_output, lower_bounds_output

    @staticmethod
    def adjust_to_within_bounds(cent: np.ndarray, eps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple helper function to pre-process and adjust zonotope values to be within 0 - 1 range.
        This is written with image data from MNIST and CIFAR10 in mind using L-infty bounds.
        Each feature here starts with a single eps term.
        Users can implement custom pre-processors tailored to their data if it does not conform to these requirements.

        :param cent: original feature values between 0 - 1
        :param eps: the zonotope error terms.
        :return: adjusted center and eps values if center + eps exceed 1 or if center - eps falls below 0.
        """
        for j in range(cent.shape[1]):
            # we assume that each feature will start with just a single eps term
            row_of_eps = np.argmax(eps[:, j])
            if cent[:, j] < eps[row_of_eps, j]:
                eps[row_of_eps, j] = (eps[row_of_eps, j] + cent[:, j]) / 2
                cent[:, j] = eps[row_of_eps, j]
            elif cent[:, j] > (1 - eps[row_of_eps, j]):
                eps[row_of_eps, j] = (eps[row_of_eps, j] + (1 - cent[:, j])) / 2
                cent[:, j] = 1 - eps[row_of_eps, j]

        return cent, eps

    def pre_process(self, cent: np.ndarray, eps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple helper function to reshape and adjust the zonotope values before pushing through the neural network.
        This is written with image data from MNIST and CIFAR10 in mind using L-infty bounds.
        Each feature here starts with a single eps term.
        Users can implement custom pre-processors tailored to their data if it does not conform to these requirements.

        :param cent: original feature values between 0 - 1
        :param eps: the zonotope error terms.
        :return: adjusted center and eps values if center + eps exceed 1 or if center - eps falls below 0.
        """
        original_shape = cent.shape
        cent = np.reshape(np.copy(cent), (1, -1))
        num_of_error_terms = eps.shape[0]
        cent, eps = self.adjust_to_within_bounds(cent, np.copy(eps))
        cent = np.reshape(cent, original_shape)

        reshape_dim = (num_of_error_terms,) + original_shape
        eps = np.reshape(eps, reshape_dim)

        return cent, eps


class ZonoConv(torch.nn.Module):
    """
    Wrapper around pytorch's convolutional layer.
    We only add the bias to the zeroth element of the zonotope
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]],
        dilation: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
    ):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            bias=False,
        )
        self.bias = torch.nn.Parameter(
            torch.zeros(
                out_channels,
            )
        )

    def __call__(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.forward(x)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Abstract forward pass through the convolutional layer

        :param x: input zonotope to the convolutional layer.
        :return x: zonotope after being pushed through the convolutional layer.
        """
        x = self.conv(x)
        x = self.zonotope_add(x)
        return x

    def concrete_forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Concrete forward pass through the convolutional layer

        :param x: concrete input to the convolutional layer.
        :return: concrete convolutional layer outputs.
        """
        x = self.conv(x)
        bias = torch.unsqueeze(self.bias, dim=-1)
        bias = torch.unsqueeze(bias, dim=-1)
        bias = torch.unsqueeze(bias, dim=0)
        return x + bias

    def zonotope_add(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Modification required compared to the normal torch conv layers.
        The bias is added only to the central zonotope term and not the error terms.

        :param x: zonotope input to have the bias added.
        :return: zonotope with the bias added to the central (first) term.
        """
        # unsqueeze to broadcast along height and width
        bias = torch.unsqueeze(self.bias, dim=-1)
        bias = torch.unsqueeze(bias, dim=-1)
        x[0] = x[0] + bias
        return x


class ZonoReLU(torch.nn.Module, ZonoBounds):
    """
    Implements "DeepZ" for relu.

    | Paper link:  https://papers.nips.cc/paper/2018/file/f2f446980d8e971ef3da97af089481c3-Paper.pdf
    """

    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.concrete_activation = torch.nn.ReLU()

    def __call__(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.forward(x)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Forward pass through the relu

        :param x: input zonotope to the dense layer.
        :return x: zonotope after being pushed through the dense layer.
        """
        return self.zonotope_relu(x)

    def concrete_forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Concrete pass through the ReLU function

        :param x: concrete input to the activation function.
        :return: concrete outputs from the ReLU.
        """
        return self.concrete_activation(x)

    def zonotope_relu(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Implements "DeepZ" for relu.

        :param x: input zonotope
        :return x: zonotope after application of the relu. May have grown in dimension if crossing relus occur.
        """
        original_shape = x.shape
        x = x.reshape((x.shape[0], -1))

        # compute lower bounds
        lbs = self.compute_lb(cent=x[0], eps=x[1:])

        # compute upper bounds
        ubs = self.compute_ub(cent=x[0], eps=x[1:])
        slope = torch.div(ubs, (ubs - lbs))

        index_cent_vector = torch.zeros((x.shape[0], 1)).to(self.device)
        index_cent_vector[0] = 1

        cent_update = (slope * lbs) / 2
        cent_update = torch.tile(cent_update, (x.shape[0], 1))

        # find where we have a crossing relu
        bools = torch.logical_and(lbs < 0, ubs > 0)

        # where we have a crossing relu, update the terms. Else, do not change input.
        x = torch.where(bools, x * slope - cent_update * index_cent_vector, x)

        # where we have a feature that is < 0, relu always returns 0
        zeros = torch.from_numpy(np.zeros(1).astype("float32")).to(self.device)
        x = torch.where(ubs < 0, zeros, x)

        # vector containing all the (potential) new error terms. We will need to 1) select the ones
        # we need and zero out the rest. And 2) Shape the errors into the correct matrix.
        new_vector = torch.unsqueeze(-1 * ((slope * lbs) / 2), dim=0)

        # indexing_matrix is the shape we want the error terms to be.
        indexing_matrix = np.zeros((torch.sum(bools), x.shape[1]))

        tmp_crossing_relu = torch.logical_and(lbs < 0, ubs > 0)
        crossing_relu_index = 0
        for j, crossing_relu in enumerate(tmp_crossing_relu):
            if crossing_relu:
                indexing_matrix[crossing_relu_index, j] = 1
                crossing_relu_index += 1

        indexing_matrix_tensor = torch.from_numpy(indexing_matrix.astype("float32")).to(self.device)

        # where there is a crossing ReLU, select the error terms, else zero the vector.
        new_vector = torch.where(bools, new_vector, zeros)
        # tile the error vector to the correct shape and select the terms we want.
        # crossing_relu_index at this point is also the same as the number of crossing relus
        new_vector = torch.tile(new_vector, (crossing_relu_index, 1))
        new_vector = new_vector * indexing_matrix_tensor

        # add the new error terms to the zonotope.
        x = torch.cat((x, new_vector))

        if len(original_shape) > 2:
            x = x.reshape((-1, original_shape[1], original_shape[2], original_shape[3]))
        return x
