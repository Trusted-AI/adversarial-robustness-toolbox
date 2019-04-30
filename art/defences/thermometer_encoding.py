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
from art.utils import to_categorical
from art import NUMPY_DTYPE

logger = logging.getLogger(__name__)


class ThermometerEncoding(Preprocessor):
    """
    Implement the thermometer encoding defence approach. Defence method from https://openreview.net/forum?id=S18Su--CW.
    """
    params = ['num_space', 'clip_values']

    def __init__(self, num_space=10, clip_values=(0, 1)):
        """
        Create an instance of thermometer encoding.

        :param num_space: Number of evenly spaced levels within [0, 1].
        :type num_space: `int`
        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :type clip_values: `tuple`
        """
        super(ThermometerEncoding, self).__init__()
        self._is_fitted = True
        self.set_params(num_space=num_space, clip_values=clip_values)

    @property
    def apply_fit(self):
        return True

    @property
    def apply_predict(self):
        return True

    def __call__(self, x, y=None, num_space=None, clip_values=None):
        """
        Apply thermometer encoding to sample `x`.

        :param x: Sample to encode with shape `(batch_size, width, height, depth)`.
        :type x: `np.ndarray`
        :param y: Labels of the sample `x`. This function does not affect them in any way.
        :type y: `np.ndarray`
        :param num_space: Number of evenly spaced levels within [0, 1].
        :type num_space: `int`
        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :type clip_values: `tuple`
        :return: Encoded sample with shape `(batch_size, width, height, depth x num_space)`.
        :rtype: `np.ndarray`
        """
        params = {}
        if num_space is not None:
            params['num_space'] = num_space

        if clip_values is not None:
            params['clip_values'] = clip_values

        self.set_params(**params)

        result = []
        for c in range(x.shape[-1]):
            result.append(self._perchannel(x[:, :, :, c]))

        result = np.concatenate(result, axis=3)
        result = np.clip(result, self.clip_values[0], self.clip_values[1])

        return result.astype(NUMPY_DTYPE), y

    def _perchannel(self, x):
        """
        Apply thermometer encoding to one channel.

        :param x: Sample to encode with shape `(batch_size, width, height)`.
        :type x: `np.ndarray`
        :return: Encoded sample with shape `(batch_size, width, height, num_space)`.
        :rtype: `np.ndarray`
        """
        pos = np.zeros(shape=x.shape)
        for i in range(1, self.num_space):
            pos[x > float(i) / self.num_space] += 1

        onehot_rep = to_categorical(pos.reshape(-1), self.num_space)

        for i in reversed(range(1, self.num_space)):
            onehot_rep[:, i] += np.sum(onehot_rep[:, :i], axis=1)

        result = onehot_rep.reshape(list(x.shape) + [self.num_space])

        return result

    def estimate_gradient(self, x, grad):
        """
        Provide an estimate of the gradients of the defence for the backward pass. For thermometer encoding,
        the gradient estimate is the one used in https://arxiv.org/abs/1802.00420, where the thermometer encoding
        is replaced with a differentiable approximation:
        `g(x_{i,j,c})_k = min(max(x_{i,j,c} - k / self.num_space, 0), 1)`.

        :param x: Input data for which the gradient is estimated. First dimension is the batch size.
        :type x: `np.ndarray`
        :param grad: Gradient value so far.
        :type grad: `np.ndarray`
        :return: The gradient (estimate) of the defence.
        :rtype: `np.ndarray`
        """
        thermometer_grad = np.zeros(x.shape + (self.num_space,))
        mask = np.array([x > k / self.num_space for k in range(self.num_space)])
        mask = np.moveaxis(mask, 0, -1)
        thermometer_grad[mask] = 1

        return grad * thermometer_grad

    def fit(self, x, y=None, **kwargs):
        """
        No parameters to learn for this method; do nothing.
        """
        pass

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and applies defence-specific checks before saving them as attributes.

        :param num_space: Number of evenly spaced levels within [0, 1].
        :type num_space: `int`
        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :type clip_values: `tuple`
        """
        # Save attack-specific parameters
        super(ThermometerEncoding, self).set_params(**kwargs)

        if not isinstance(self.num_space, (int, np.int)) or self.num_space <= 0:
            logger.error('Number of evenly spaced levels must be a positive integer.')
            raise ValueError('Number of evenly spaced levels must be a positive integer.')

        if len(self.clip_values) != 2:
            raise ValueError('`clip_values` should be a tuple of 2 floats containing the allowed data range.')
        if self.clip_values[0] >= self.clip_values[1]:
            raise ValueError('Invalid `clip_values`: min >= max.')

        return True
