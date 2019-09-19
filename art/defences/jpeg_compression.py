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
This module implements the JPEG compression defence `JpegCompression`.

| Paper link: https://arxiv.org/abs/1705.02900, https://arxiv.org/abs/1608.00853
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from io import BytesIO
import logging

import numpy as np

from art.defences.preprocessor import Preprocessor
from art import NUMPY_DTYPE

logger = logging.getLogger(__name__)


class JpegCompression(Preprocessor):
    """
    Implement the JPEG compression defence approach.

    | Paper link: https://arxiv.org/abs/1705.02900, https://arxiv.org/abs/1608.00853
    """
    params = ['quality', 'channel_index', 'clip_values']

    def __init__(self, clip_values, quality=50, channel_index=3, apply_fit=True, apply_predict=False):
        """
        Create an instance of JPEG compression.

        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :type clip_values: `tuple`
        :param quality: The image quality, on a scale from 1 (worst) to 95 (best). Values above 95 should be avoided.
        :type quality: `int`
        :param channel_index: Index of the axis in data containing the color channels or features.
        :type channel_index: `int`
        :param apply_fit: True if applied during fitting/training.
        :type apply_fit: `bool`
        :param apply_predict: True if applied during predicting.
        :type apply_predict: `bool`
        """
        super(JpegCompression, self).__init__()
        self._is_fitted = True
        self._apply_fit = apply_fit
        self._apply_predict = apply_predict
        self.set_params(quality=quality, channel_index=channel_index, clip_values=clip_values)

    @property
    def apply_fit(self):
        return self._apply_fit

    @property
    def apply_predict(self):
        return self._apply_predict

    def __call__(self, x, y=None):
        """
        Apply JPEG compression to sample `x`.

        :param x: Sample to compress with shape `(batch_size, width, height, depth)`. `x` values are expected to be in
               the data range [0, 1].
        :type x: `np.ndarray`
        :param y: Labels of the sample `x`. This function does not affect them in any way.
        :type y: `np.ndarray`
        :return: compressed sample.
        :rtype: `np.ndarray`
        """
        from PIL import Image

        if len(x.shape) == 2:
            raise ValueError('Feature vectors detected. JPEG compression can only be applied to data with spatial'
                             'dimensions.')

        if self.channel_index >= len(x.shape):
            raise ValueError('Channel index does not match input shape.')

        if np.min(x) < 0.0:
            raise ValueError('Negative values in input `x` detected. The JPEG compression defence requires unnormalized'
                             'input.')

        # Swap channel index
        if self.channel_index < 3 and len(x.shape) == 4:
            x_local = np.swapaxes(x, self.channel_index, 3)
        else:
            x_local = x.copy()

        # Convert into `uint8`
        if self.clip_values[1] == 1.0:
            x_local = x_local * 255
        x_local = x_local.astype("uint8")

        # Convert to 'L' mode
        if x_local.shape[-1] == 1:
            x_local = np.reshape(x_local, x_local.shape[:-1])

        # Compress one image at a time
        for i, x_i in enumerate(x_local):
            if len(x_i.shape) == 2:
                x_i = Image.fromarray(x_i, mode='L')
            elif x_i.shape[-1] == 3:
                x_i = Image.fromarray(x_i, mode='RGB')
            else:
                logger.log(level=40, msg="Currently only support `RGB` and `L` images.")
                raise NotImplementedError("Currently only support `RGB` and `L` images.")

            out = BytesIO()
            x_i.save(out, format="jpeg", quality=self.quality)
            x_i = Image.open(out)
            x_i = np.array(x_i)
            x_local[i] = x_i
            del out

        # Expand dim if black/white images
        if len(x_local.shape) < 4:
            x_local = np.expand_dims(x_local, 3)

        # Convert to old dtype
        if self.clip_values[1] == 1.0:
            x_local = x_local / 255.0
        x_local = x_local.astype(NUMPY_DTYPE)

        # Swap channel index
        if self.channel_index < 3:
            x_local = np.swapaxes(x_local, self.channel_index, 3)

        return x_local, y

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

        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :type clip_values: `tuple`
        :param quality: The image quality, on a scale from 1 (worst) to 95 (best). Values above 95 should be avoided.
        :type quality: `int`
        :param channel_index: Index of the axis in data containing the color channels or features.
        :type channel_index: `int`
        """
        # Save defense-specific parameters
        super(JpegCompression, self).set_params(**kwargs)

        if not isinstance(self.quality, (int, np.int)) or self.quality <= 0 or self.quality > 100:
            logger.error('Image quality must be a positive integer <= 100.')
            raise ValueError('Image quality must be a positive integer <= 100.')

        if not isinstance(self.channel_index, (int, np.int)) or self.channel_index <= 0:
            logger.error('Data channel must be a positive integer. The batch dimension is not a valid channel.')
            raise ValueError('Data channel must be a positive integer. The batch dimension is not a valid channel.')

        if len(self.clip_values) != 2:
            raise ValueError('`clip_values` should be a tuple of 2 floats or arrays containing the allowed'
                             'data range.')

        if np.array(self.clip_values[0] >= self.clip_values[1]).any():
            raise ValueError('Invalid `clip_values`: min >= max.')

        if self.clip_values[0] != 0:
            raise ValueError('`clip_values` min value must be 0.')

        if self.clip_values[1] != 1.0 and self.clip_values[1] != 255:
            raise ValueError('`clip_values` max value must be either 1 or 255.')

        return True
