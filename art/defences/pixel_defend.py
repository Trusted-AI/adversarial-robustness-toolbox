from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np

from art.defences.preprocessor import Preprocessor

logger = logging.getLogger(__name__)


class PixelDefend(Preprocessor):
    """
    Implement the pixel defence approach. Defense based on PixelCNN that projects samples back to the data manifold.
    Paper link: https://arxiv.org/abs/1710.10766
    """
    params = ['eps']

    def __init__(self, eps=16):
        """
        Create an instance of pixel defence.

        :param eps: Defense parameter 0-255.
        :type eps: `int`
        """
        super(PixelDefend, self).__init__()
        self._is_fitted = True
        self.set_params(eps=eps)

    def __call__(self, x, y=None, eps=None):
        """
        Apply pixel defence to sample `x`.

        :param x: Sample to defense. `x` values are expected to be in the data range [0, 1].
        :type x: `np.ndarrray`
        :param y: Labels of the sample `x`. This function does not affect them in any way.
        :type y: `np.ndarray`
        :param eps: Defense parameter 0-255.
        :type eps: `int`
        :return: Purified sample.
        :rtype: `np.ndarray`
        """
        clip_values = (0, 1)
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

        if not isinstance(self.bit_depth, (int, np.int)) or self.bit_depth <= 0 or self.bit_depth > 64:
            raise ValueError("The bit depth must be between 1 and 64.")

        return True
