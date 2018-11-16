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

from io import BytesIO
import logging

import numpy as np
from PIL import Image

from art.defences.preprocessor import Preprocessor
from art import NUMPY_DTYPE

logger = logging.getLogger(__name__)


class JpegCompression(Preprocessor):
    """
    Implement the jpeg compression defence approach. Some related papers: https://arxiv.org/pdf/1705.02900.pdf,
    https://arxiv.org/abs/1608.00853
    """
    params = ['quality', 'channel_index']

    def __init__(self, quality=50, channel_index=3):
        """
        Create an instance of jpeg compression.

        :param quality: The image quality, on a scale from 1 (worst) to 95 (best). Values above 95 should be avoided.
        :type quality: `int`
        :param channel_index: Index of the axis in data containing the color channels or features.
        :type channel_index: `int`
        """
        super(JpegCompression, self).__init__()
        self._is_fitted = True
        self.set_params(quality=quality, channel_index=channel_index)

    def __call__(self, x, y=None, quality=None, clip_values=(0, 1)):
        """
        Apply jpeg compression to sample `x`.

        :param x: Sample to compress with shape `(batch_size, width, height, depth)`.
        :type x: `np.ndarray`
        :param y: Labels of the sample `x`. This function does not affect them in any way.
        :type y: `np.ndarray`
        :param quality: The image quality, on a scale from 1 (worst) to 95 (best). Values above 95 should be avoided.
        :type quality: `int`
        :return: compressed sample
        :rtype: `np.ndarray`
        """
        if quality is not None:
            self.set_params(quality=quality)

        assert self.channel_index < len(x.shape)

        # Swap channel index
        if self.channel_index < 3:
            x_ = np.swapaxes(x, self.channel_index, 3)
        else:
            x_ = x.copy()

        # Convert into `uint8`
        x_ = x_ * 255
        x_ = x_.astype("uint8")

        # Convert to 'L' mode
        if x_.shape[-1] == 1:
            x_ = np.reshape(x_, x_.shape[:-1])

        # Compress one image per time
        for i, xi in enumerate(x_):
            if len(xi.shape) == 2:
                xi = Image.fromarray(xi, mode='L')
            elif xi.shape[-1] == 3:
                xi = Image.fromarray(xi, mode='RGB')
            else:
                logger.log(level=40, msg="Currently only support `RGB` and `L` images.")
                raise NotImplementedError("Currently only support `RGB` and `L` images.")

            out = BytesIO()
            xi.save(out, format="jpeg", quality=self.quality)
            xi = Image.open(out)
            xi = np.array(xi)
            x_[i] = xi
            del out

        # Expand dim if black/white images
        if len(x_.shape) < 4:
            x_ = np.expand_dims(x_, 3)

        # Convert to old dtype
        x_ = x_ / 255.0
        x_ = x_.astype(NUMPY_DTYPE)

        # Swap channel index
        if self.channel_index < 3:
            x_ = np.swapaxes(x_, self.channel_index, 3)

        x_ = np.clip(x_, clip_values[0], clip_values[1])

        return x_

    def fit(self, x, y=None, **kwargs):
        """
        No parameters to learn for this method; do nothing.
        """
        pass

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and applies defence-specific checks before saving them as attributes.

        :param quality: The image quality, on a scale from 1 (worst) to 95 (best). Values above 95 should be avoided.
        :type quality: `int`
        :param channel_index: Index of the axis in data containing the color channels or features.
        :type channel_index: `int`
        """
        # Save defense-specific parameters
        super(JpegCompression, self).set_params(**kwargs)

        if type(self.quality) is not int or self.quality <= 0 or self.quality > 100:
            logger.error('Image quality must be a positive integer and smaller than 101.')
            raise ValueError('Image quality must be a positive integer and smaller than 101.')

        if type(self.channel_index) is not int or self.channel_index <= 0:
            logger.error('Data channel must be a positive integer. The batch dimension is not a valid channel.')
            raise ValueError('Image quality must be a positive integer and smaller than 101.')

        return True



