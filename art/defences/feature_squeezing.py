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

from art.defences.preprocessor import Preprocessor

logger = logging.getLogger(__name__)


class FeatureSqueezing(Preprocessor):
    """
    Reduces the sensibility of the features of a sample. Defence method from https://arxiv.org/abs/1704.01155.
    """
    params = ['bit_depth']

    def __init__(self, bit_depth=8):
        """
        Create an instance of feature squeezing.

        :param bit_depth: The number of bits per channel for encoding the data.
        :type bit_depth: `int`
        """
        super(FeatureSqueezing, self).__init__()
        self._is_fitted = True
        self.set_params(bit_depth=bit_depth)

    def __call__(self, x, y=None, bit_depth=None, clip_values=(0, 1)):
        """
        Apply feature squeezing to sample `x`.

        :param x: Sample to squeeze. `x` values are expected to be in the data range provided by `clip_values`.
        :type x: `np.ndarrray`
        :param y: Labels of the sample `x`. This function does not affect them in any way.
        :type y: `np.ndarray`
        :param bit_depth: The number of bits per channel for encoding the data.
        :type bit_depth: `int`
        :return: Squeezed sample
        :rtype: `np.ndarray`
        """
        if bit_depth is not None:
            self.set_params(bit_depth=bit_depth)

        x_ = x - clip_values[0]
        if clip_values[1] != 0:
            x_ = x_ / (clip_values[1] - clip_values[0])

        max_value = np.rint(2 ** self.bit_depth - 1)
        res = (np.rint(x_ * max_value) / max_value) * (clip_values[1] - clip_values[0]) + clip_values[0]

        return res

    def fit(self, x, y=None, **kwargs):
        """No parameters to learn for this method; do nothing."""
        pass

    def set_params(self, **kwargs):
        """Take in a dictionary of parameters and applies defence-specific checks before saving them as attributes.

        Defense-specific parameters:
        :param bit_depth: The number of bits per channel for encoding the data.
        :type bit_depth: `int`
        """
        # Save attack-specific parameters
        super(FeatureSqueezing, self).set_params(**kwargs)

        if type(self.bit_depth) is not int or self.bit_depth <= 0 or self.bit_depth > 64:
            raise ValueError("The bit depth must be between 1 and 64.")

        return True
