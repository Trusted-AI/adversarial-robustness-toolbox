# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
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
This module implements the thermometer encoding defence `ThermometerEncoding`.

| Paper link: https://openreview.net/forum?id=S18Su--CW

| Please keep in mind the limitations of defences. For more information on the limitations of this defence,
    see https://arxiv.org/abs/1802.00420 . For details on how to evaluate classifier security in general, see
    https://arxiv.org/abs/1902.06705
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np

from art.config import ART_NUMPY_DTYPE
from art.defences.preprocessor.preprocessor import Preprocessor
from art.utils import to_categorical

if TYPE_CHECKING:
    from art.utils import CLIP_VALUES_TYPE

logger = logging.getLogger(__name__)


class ThermometerEncoding(Preprocessor):
    """
    Implement the thermometer encoding defence approach.

    | Paper link: https://openreview.net/forum?id=S18Su--CW

    | Please keep in mind the limitations of defences. For more information on the limitations of this defence,
        see https://arxiv.org/abs/1802.00420 . For details on how to evaluate classifier security in general, see
        https://arxiv.org/abs/1902.06705
    """

    params = ["clip_values", "num_space", "channels_first"]

    def __init__(
        self,
        clip_values: "CLIP_VALUES_TYPE",
        num_space: int = 10,
        channels_first: bool = False,
        apply_fit: bool = True,
        apply_predict: bool = True,
    ) -> None:
        """
        Create an instance of thermometer encoding.

        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :param num_space: Number of evenly spaced levels within the interval of minimum and maximum clip values.
        :param channels_first: Set channels first or last.
        :param apply_fit: True if applied during fitting/training.
        :param apply_predict: True if applied during predicting.
        """
        super().__init__(is_fitted=True, apply_fit=apply_fit, apply_predict=apply_predict)
        self.clip_values = clip_values
        self.num_space = num_space
        self.channels_first = channels_first
        self._check_params()

    def __call__(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply thermometer encoding to sample `x`. The new axis with the encoding is added as last dimension.

        :param x: Sample to encode with shape `(batch_size, width, height, depth)`.
        :param y: Labels of the sample `x`. This function does not affect them in any way.
        :return: Encoded sample with shape `(batch_size, width, height, depth x num_space)`.
        """
        # First normalize the input to be in [0, 1]:
        np.clip(x, self.clip_values[0], self.clip_values[1], out=x)
        x = (x - self.clip_values[0]) / (self.clip_values[1] - self.clip_values[0])

        # Now apply the encoding:
        channel_index = 1 if self.channels_first else x.ndim - 1
        result = np.apply_along_axis(self._perchannel, channel_index, x)
        np.clip(result, 0, 1, out=result)
        return result.astype(ART_NUMPY_DTYPE), y

    def _perchannel(self, x: np.ndarray) -> np.ndarray:
        """
        Apply thermometer encoding to one channel.

        :param x: Sample to encode with shape `(batch_size, width, height)`.
        :return: Encoded sample with shape `(batch_size, width, height, num_space)`.
        """
        pos = np.zeros(shape=x.shape)
        for i in range(1, self.num_space):
            pos[x > float(i) / self.num_space] += 1

        onehot_rep = to_categorical(pos.reshape(-1), self.num_space)

        for i in range(self.num_space - 1):
            onehot_rep[:, i] += np.sum(onehot_rep[:, i + 1 :], axis=1)

        return onehot_rep.flatten()

    def estimate_gradient(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """
        Provide an estimate of the gradients of the defence for the backward pass. For thermometer encoding,
        the gradient estimate is the one used in https://arxiv.org/abs/1802.00420, where the thermometer encoding
        is replaced with a differentiable approximation:
        `g(x_{i,j,c})_k = min(max(x_{i,j,c} - k / self.num_space, 0), 1)`.

        :param x: Input data for which the gradient is estimated. First dimension is the batch size.
        :param grad: Gradient value so far.
        :return: The gradient (estimate) of the defence.
        """
        if self.channels_first:
            x = np.transpose(x, (0,) + tuple(range(2, len(x.shape))) + (1,))
            grad = np.transpose(grad, (0,) + tuple(range(2, len(x.shape))) + (1,))

        thermometer_grad = np.zeros(x.shape[:-1] + (x.shape[-1] * self.num_space,))
        mask = np.array([x > k / self.num_space for k in range(self.num_space)])
        mask = np.moveaxis(mask, 0, -1)
        mask = mask.reshape(thermometer_grad.shape)
        thermometer_grad[mask] = 1

        grad = grad * thermometer_grad
        grad = np.reshape(grad, grad.shape[:-1] + (grad.shape[-1] // self.num_space, self.num_space))
        grad = np.sum(grad, -1)

        if self.channels_first:
            x = np.transpose(x, (0,) + (len(x.shape) - 1,) + tuple(range(1, len(x.shape) - 1)))
            grad = np.transpose(grad, (0,) + (len(x.shape) - 1,) + tuple(range(1, len(x.shape) - 1)))

        return grad / (self.clip_values[1] - self.clip_values[0])

    def _check_params(self) -> None:
        if not isinstance(self.num_space, (int, np.int)) or self.num_space <= 0:
            logger.error("Number of evenly spaced levels must be a positive integer.")
            raise ValueError("Number of evenly spaced levels must be a positive integer.")

        if len(self.clip_values) != 2:
            raise ValueError("`clip_values` should be a tuple of 2 floats containing the allowed data range.")

        if self.clip_values[0] >= self.clip_values[1]:
            raise ValueError("first entry of `clip_values` should be strictly smaller than the second one.")
