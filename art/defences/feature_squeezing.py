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

import numpy as np
import tensorflow as tf

from art.defences.preprocessor import Preprocessor


class FeatureSqueezing(Preprocessor):
    """
    Reduces the sensibility of the features of a sample. Defence method from https://arxiv.org/abs/1704.01155.
    """
    params = ['bit_depth']

    def __init__(self, bit_depth=8):
        """
        Create an instance of feature squeezing.

        :param bit_depth: The number of bits to encode data on
        :type bit_depth: `int`
        """
        super(FeatureSqueezing, self).__init__()
        self.is_fitted = True
        self.set_params(bit_depth=bit_depth)

    def __call__(self, x_val, bit_depth=None):
        """
        Apply feature squeezing to sample `x_val`.

        :param x_val: Sample to squeeze. `x_val` values are supposed to be in the range [0,1]
        :type x_val: `np.ndarrray`
        :param bit_depth: The number of bits to encode data on
        :type bit_depth: `int`
        :return: Squeezed sample
        :rtype: `np.ndarray`
        """
        if bit_depth is not None:
            self.set_params(bit_depth=bit_depth)

        max_value = np.rint(2 ** self.bit_depth - 1)
        return np.rint(x_val * max_value) / max_value

    def fit(self, x_val, y_val=None, **kwargs):
        """No parameters to learn for this method; do nothing."""
        pass

    def _tf_predict(self, x, bit_depth=None):
        """
        Apply feature squeezing on `tf.Tensor`.

        :param x: Sample to squeeze. Values are supposed to be in the range [0,1]
        :type x: `tf.Tensor`
        :param bit_depth: The number of bits to encode data on
        :type bit_depth: `int`
        :return: Squeezed sample
        :rtype: tf.Tensor
        """
        if bit_depth is not None:
            self.set_params(bit_depth=bit_depth)

        max_value = int(2 ** self.bit_depth - 1)
        x = tf.rint(x * max_value) / max_value
        return x

    def set_params(self, **kwargs):
        """Take in a dictionary of parameters and applies defense-specific checks before saving them as attributes.

        Defense-specific parameters:
        :param bit_depth: The number of bits to encode data on
        :type bit_depth: `int`
        """
        # Save attack-specific parameters
        super(FeatureSqueezing, self).set_params(**kwargs)

        if type(self.bit_depth) is not int or self.bit_depth <= 0 or self.bit_depth > 60:
            raise ValueError("The bit depth must be between 1 and 60.")

        return True
