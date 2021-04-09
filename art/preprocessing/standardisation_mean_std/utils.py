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
"""
This module implements utilities for standardisation with mean and standard deviation.
"""

from typing import Tuple

import numpy as np


def broadcastable_mean_std(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ensure that the mean and standard deviation are broadcastable with respect to input `x`.

    :param x: Input samples to standardise.
    :param mean: Mean.
    :param std: Standard Deviation.
    """
    if mean.shape != std.shape:
        raise ValueError("The shape of mean and the standard deviation must be identical.")

    # catch non-broadcastable input, when mean and std are vectors
    if mean.ndim == 1 and mean.shape[0] > 1 and mean.shape[0] != x.shape[-1]:
        # allow input shapes NC* (batch) and C* (non-batch)
        channel_idx = 1 if x.shape[1] == mean.shape[0] else 0
        broadcastable_shape = [1] * x.ndim
        broadcastable_shape[channel_idx] = mean.shape[0]

        # expand mean and std to new shape
        mean = mean.reshape(broadcastable_shape)
        std = std.reshape(broadcastable_shape)
    return mean, std
