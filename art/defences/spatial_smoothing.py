# MIT License
#
# Copyright (C) IBM Corporation 2018
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
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np
from scipy import ndimage

from art.defences.preprocessor import Preprocessor
from art import NUMPY_DTYPE

logger = logging.getLogger(__name__)


class SpatialSmoothing(Preprocessor):
    """
    Implement the local spatial smoothing defence approach.

    | Paper link: https://arxiv.org/abs/1704.01155
    """
    params = ['window_size', 'channel_index', 'clip_values']

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
        if len(x.shape) == 2:
            raise ValueError('Feature vectors detected. Smoothing can only be applied to data with spatial '
                             'dimensions.')
        if self.channel_index >= len(x.shape):
            raise ValueError('Channel index does not match input shape.')

        size = [1] + [self.window_size] * (len(x.shape) - 1)
        size[self.channel_index] = 1
        size = tuple(size)

        result = ndimage.filters.median_filter(x, size=size, mode="reflect")
        if self.clip_values is not None:
            np.clip(result, self.clip_values[0], self.clip_values[1], out=result)

        return result.astype(NUMPY_DTYPE), y

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
        # Save attack-specific parameters
        super(SpatialSmoothing, self).set_params(**kwargs)

        if not isinstance(self.window_size, (int, np.int)) or self.window_size <= 0:
            logger.error('Sliding window size must be a positive integer.')
            raise ValueError('Sliding window size must be a positive integer.')

        if not isinstance(self.channel_index, (int, np.int)) or self.channel_index <= 0:
            logger.error('Data channel for smoothing must be a positive integer. The batch dimension is not a'
                         'valid channel.')
            raise ValueError('Data channel for smoothing must be a positive integer. The batch dimension is not a'
                             'valid channel.')

        if self.clip_values is not None:

            if len(self.clip_values) != 2:
                raise ValueError('`clip_values` should be a tuple of 2 floats containing the allowed data range.')

            if np.array(self.clip_values[0] >= self.clip_values[1]).any():
                raise ValueError('Invalid `clip_values`: min >= max.')

        return True
