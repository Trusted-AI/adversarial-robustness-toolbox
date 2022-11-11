import torch
import numpy as np
from typing import List, Union


class IntervalDenseLayer(torch.nn.Module):
    """
    Class implementing a dense layer for the interval (box) domain.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.weight = torch.nn.Parameter(torch.normal(mean=torch.zeros(out_features, in_features),
                                                      std=torch.ones(out_features, in_features)))
        self.bias = torch.nn.Parameter(torch.normal(mean=torch.zeros(out_features),
                                                    std=torch.ones(out_features)))

    def __call__(self, x: "torch.Tensor"):
        return self.interval_matmul(x)

    def interval_matmul(self, x: "torch.Tensor") -> "torch.Tensor":

        u = (x[:, 1] + x[:, 0])/2
        r = (x[:, 1] - x[:, 0])/2

        u = torch.matmul(u, torch.transpose(self.weight, 0, 1)) + self.bias
        r = torch.matmul(r, torch.abs(torch.transpose(self.weight, 0, 1)))

        u = torch.unsqueeze(u, dim=1)
        r = torch.unsqueeze(r, dim=1)

        return torch.cat([u-r, u+r], dim=1)

    def concrete_forward(self, x):
        return torch.matmul(x, torch.transpose(self.weight, 0, 1)) + self.bias


class BoxConvLayer(torch.nn.Module):
    """
    Class implementing a convolutional layer in the box domain.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, input_shape, stride: int = 1,
                 bias: bool = False, to_debug: bool = False):
        super().__init__()

        self.conv_flat = torch.nn.Conv2d(
            in_channels=1, out_channels=out_channels * in_channels, kernel_size=(kernel_size, kernel_size), bias=False, stride=stride,
        )
        self.b = None

        if bias:
            self.conv_bias = torch.nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, kernel_size), bias=True, stride=stride,
            )
            self.b = self.conv_bias.bias.data

        if to_debug:
            self.conv = torch.nn.Conv2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=(kernel_size, kernel_size), bias=bias, stride=stride)
            self.conv_flat.weight = torch.nn.Parameter(
                torch.reshape(torch.tensor(self.conv.weight.data.cpu().detach().numpy()),
                              (out_channels * in_channels, 1, kernel_size, kernel_size))
            )
            if bias:
                self.b = self.conv.bias.data

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]

        self.output_height = None
        self.output_width = None

        self.dense_weights, self.b = self.convert_to_dense_pt()

    def convert_to_dense_pt(self) -> "torch.Tensor":
        """
        Converts the initialised convolutional layer into an equivalent dense layer.
        """

        diagonal_input = torch.reshape(
            torch.eye(self.input_height * self.input_width),
            shape=[self.input_height * self.input_width, 1, self.input_height, self.input_width],
        )
        print("the input shape is ", diagonal_input.shape)
        conv = self.conv_flat(diagonal_input)
        self.output_height = conv.shape[2]
        self.output_width = conv.shape[3]
        print("conv shape is ", conv.shape)

        # conv is of shape (input_height * input_width, out_channels * in_channels, output_height, output_width).
        # Reshape it to (input_height * input_width * output_channels,
        #                output_height * output_width * input_channels).

        weights = torch.reshape(
            conv,
            shape=(
                [self.input_height * self.input_width, self.out_channels, self.in_channels, self.output_height, self.output_width]
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

        if self.b is not None:
            self.b = torch.unsqueeze(self.b, dim=-1)
            b = self.b.expand(-1, self.output_height * self.output_width)
            b = b.flatten()
        else:
            b = None

        return torch.transpose(weights, 0, 1), b

    def concrete_forward_via_dense(self, x: "torch.Tensor") -> "torch.Tensor":
        # TODO: implement interval arithmetic here.
        x = torch.reshape(x, (x.shape[0], -1))
        if self.b is None:
            x = torch.matmul(x, torch.transpose(self.dense_weights, 0, 1))
            return x.reshape((-1, self.out_channels, self.output_height, self.output_width))
        else:
            x = torch.matmul(x, torch.transpose(self.dense_weights, 0, 1)) + self.b
            return x.reshape((-1, self.out_channels, self.output_height, self.output_width))

    def abstract_forward(self, x):

        u = (x[:, 1] + x[:, 0])/2
        r = (x[:, 1] - x[:, 0])/2

        u = torch.matmul(u, torch.transpose(self.weight, 0, 1)) + self.bias
        r = torch.matmul(r, torch.abs(torch.transpose(self.weight, 0, 1)))

        u = torch.unsqueeze(u, dim=1)
        r = torch.unsqueeze(r, dim=1)

        return torch.cat([u-r, u+r], dim=1)


class IntervalReLU(torch.nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.concrete_activation = torch.nn.ReLU()

    def __call__(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.forward(x)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Concrete pass through the ReLU function
        :param x: concrete input to the activation function.
        :return: concrete outputs from the ReLU.
        """
        return self.concrete_activation(x)

    def concrete_forward(self, x):
        return self.concrete_activation(x)


def convert_to_interval(x: np.ndarray, bounds: Union[float, List[float], np.ndarray], to_clip=False, limits=None):
    """
    Helper function which takes in a datapoint and converts it into its interval representation based on
    the provided bounds.
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
        raise ValueError("bounds must be a A, B, or C")

    final_batched_input = np.concatenate((lb_x, up_x), axis=1)
    # final_batched_input = torch.squeeze(final_batched_input)

    if to_clip:
        return np.clip(final_batched_input, limits[0], limits[1])
    else:
        return final_batched_input
