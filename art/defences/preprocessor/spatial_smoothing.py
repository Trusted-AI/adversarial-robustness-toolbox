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
This module implements the local spatial smoothing defence in `SpatialSmoothing`.

| Paper link: https://arxiv.org/abs/1704.01155


| Please keep in mind the limitations of defences. For more information on the limitations of this defence,
    see https://arxiv.org/abs/1803.09868 . For details on how to evaluate classifier security in general, see
    https://arxiv.org/abs/1902.06705
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np
from scipy.ndimage.filters import median_filter

from art.defences.preprocessor.preprocessor import Preprocessor

logger = logging.getLogger(__name__)


class SpatialSmoothing(Preprocessor):
    """
    Implement the local spatial smoothing defence approach.

    | Paper link: https://arxiv.org/abs/1704.01155

    | Please keep in mind the limitations of defences. For more information on the limitations of this defence,
        see https://arxiv.org/abs/1803.09868 . For details on how to evaluate classifier security in general, see
        https://arxiv.org/abs/1902.06705
    """

    params = ["window_size", "channel_index", "clip_values"]

    def __init__(self, window_size=3, channel_index=3, clip_values=None, apply_fit=False, apply_predict=True):
        """
        Create an instance of local spatial smoothing.

        :param channel_index: Index of the axis in data containing the color channels or features.
        :type channel_index: `int`
        :param window_size: The size of the sliding window.
        :type window_size: `int`
        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :type clip_values: `tuple`
        :param apply_fit: True if applied during fitting/training.
        :type apply_fit: `bool`
        :param apply_predict: True if applied during predicting.
        :type apply_predict: `bool`
        """
        super(SpatialSmoothing, self).__init__()
        self._is_fitted = True
        self._apply_fit = apply_fit
        self._apply_predict = apply_predict
        self.set_params(channel_index=channel_index, window_size=window_size, clip_values=clip_values)

    @property
    def apply_fit(self):
        return self._apply_fit

    @property
    def apply_predict(self):
        return self._apply_predict

    def __call__(self, x, y=None):
        """
        Apply local spatial smoothing to sample `x`.

        :param x: Sample to smooth with shape `(batch_size, width, height, depth)`.
        :type x: `np.ndarray`
        :param y: Labels of the sample `x`. This function does not affect them in any way.
        :type y: `np.ndarray`
        :return: Smoothed sample
        :rtype: `np.ndarray`
        """

        x_ndim = x.ndim
        if x_ndim == 2:
            raise ValueError("Feature vectors detected. Smoothing can only be applied to data with spatial dimensions.")

        if x_ndim > 5:
            raise ValueError(
                "Unrecognized input dimension. Spatial smoothing can only be applied to image and video data."
            )

        filter_size = [self.window_size] * x_ndim
        # set filter_size at batch and channel indices to 1
        filter_size[0] = 1
        filter_size[self.channel_index] = 1
        # set filter_size at temporal index to 1
        if x_ndim == 5:
            # check if NCFHW or NFHWC
            temporal_index = 2 if self.channel_index == 1 else 1
            filter_size[temporal_index] = 1
        # Note median_filter:
        # * center pixel located lower right
        # * if window size even, use larger value (e.g. median(4,5)=5)
        result = median_filter(x, size=tuple(filter_size), mode="reflect")

        if self.clip_values is not None:
            np.clip(result, self.clip_values[0], self.clip_values[1], out=result)

        return result, y

    def estimate_gradient(self, x, grad):
        return grad

    def fit(self, x, y=None, **kwargs):
        """
        No parameters to learn for this method; do nothing.
        """
        pass

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and applies defence-specific checks before saving them as attributes.

        :param window_size: The size of the sliding window.
        :type window_size: `int`
        :param channel_index: Index of the axis in data containing the color channels or features.
        :type channel_index: `int`
        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :type clip_values: `tuple`
        """
        # Save defence-specific parameters
        super(SpatialSmoothing, self).set_params(**kwargs)

        if not (isinstance(self.window_size, (int, np.int)) and self.window_size > 0):
            raise ValueError("Sliding window size must be a positive integer.")

        if not (isinstance(self.channel_index, (int, np.int)) and self.channel_index in [1, 3, 4]):
            raise ValueError(
                "Data channel must be an integer equal to 1, 3 or 4. The batch dimension is not a valid channel."
            )

        if self.clip_values is not None and len(self.clip_values) != 2:
            raise ValueError("'clip_values' should be a tuple of 2 floats or arrays containing the allowed data range.")

        if self.clip_values is not None and np.array(self.clip_values[0] >= self.clip_values[1]).any():
            raise ValueError("Invalid 'clip_values': min >= max.")

        return True
