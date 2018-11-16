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
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np
from scipy import ndimage

from art.defences.preprocessor import Preprocessor
from art import NUMPY_DTYPE

logger = logging.getLogger(__name__)


class SpatialSmoothing(Preprocessor):
    """
    Implement the local spatial smoothing defence approach. Defence method from https://arxiv.org/abs/1704.01155.
    """
    params = ['window_size', 'channel_index']

    def __init__(self, window_size=3, channel_index=3):
        """
        Create an instance of local spatial smoothing.

        :param window_size: The size of the sliding window.
        :type window_size: `int`
        :param channel_index: Index of the axis in data containing the color channels or features.
        :type channel_index: `int`
        """
        super(SpatialSmoothing, self).__init__()
        self._is_fitted = True
        self.set_params(window_size=window_size, channel_index=channel_index)

    def __call__(self, x, y=None, window_size=None, clip_values=(0, 1)):
        """
        Apply local spatial smoothing to sample `x`.

        :param x: Sample to smooth with shape `(batch_size, width, height, depth)`.
        :type x: `np.ndarray`
        :param y: Labels of the sample `x`. This function does not affect them in any way.
        :type y: `np.ndarray`
        :param window_size: The size of the sliding window.
        :type window_size: `int`
        :return: Smoothed sample
        :rtype: `np.ndarray`
        """
        if window_size is not None:
            self.set_params(window_size=window_size)

        assert self.channel_index < len(x.shape)
        size = [1] + [self.window_size] * (len(x.shape) - 1)
        size[self.channel_index] = 1
        size = tuple(size)

        result = ndimage.filters.median_filter(x, size=size, mode="reflect")
        result = np.clip(result, clip_values[0], clip_values[1])

        return result.astype(NUMPY_DTYPE)

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
        """
        # Save attack-specific parameters
        super(SpatialSmoothing, self).set_params(**kwargs)

        if type(self.window_size) is not int or self.window_size <= 0:
            logger.error('Sliding window size must be a positive integer.')
            raise ValueError('Sliding window size must be a positive integer.')

        if type(self.channel_index) is not int or self.channel_index <= 0:
            logger.error('Data channel for smoothing must be a positive integer. The batch dimension is not a'
                         'valid channel.')
            raise ValueError('Data channel for smoothing must be a positive integer. The batch dimension is not a'
                             'valid channel.')

        return True



