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


class IntervalConv2D(torch.nn.Module):
    pass


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
