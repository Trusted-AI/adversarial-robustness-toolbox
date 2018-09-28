from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np

from art.defences.preprocessor import Preprocessor
from art.utils import to_categorical

logger = logging.getLogger(__name__)


class ThermometerEncoding(Preprocessor):
    """
    Implement the thermometer encoding defence approach. Defence method from https://openreview.net/forum?id=S18Su--CW.
    """
    params = ['num_space']

    def __init__(self, num_space=10):
        """
        Create an instance of thermometer encoding.

        :param num_space: Number of evenly spaced levels within [0, 1].
        :type num_space: `int`
        """
        super(ThermometerEncoding, self).__init__()
        self._is_fitted = True
        self.set_params(num_space=num_space)

    def __call__(self, x, y=None, num_space=None):
        """
        Apply thermometer encoding to sample `x`.

        :param x: Sample to encode with shape `(batch_size, width, height, depth)`.
        :type x: `np.ndarray`
        :param y: Labels of the sample `x`. This function does not affect them in any way.
        :type y: `np.ndarray`
        :param num_space: Number of evenly spaced levels within [0, 1].
        :type num_space: `int`
        :return: Encoded sample with shape `(batch_size, width, height, depth x num_space)`.
        :rtype: `np.ndarray`
        """
        if num_space is not None:
            self.set_params(num_space=num_space)

        result = []
        for c in range(x.shape[-1]):
            result.append(self._perchannel(x[:, :, :, c]))

        result = np.concatenate(result, axis=3)

        return result

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
        """
        # Save attack-specific parameters
        super(ThermometerEncoding, self).set_params(**kwargs)

        if type(self.num_space) is not int or self.num_space <= 0:
            raise ValueError('Number of evenly spaced levels must be a positive integer.')

        return True



