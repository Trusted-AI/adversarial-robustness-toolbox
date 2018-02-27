from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from tensorflow import rint

from src.defences.preprocessor import Preprocessor


class FeatureSqueezing(Preprocessor):
    """
    Reduces the sensibility of the features of a sample. Defence method from https://arxiv.org/abs/1704.01155.
    """
    params = ['bit_depth']

    def __init__(self, bit_depth=8):
        """
        Create an instance of feature squeezing.

        :param bit_depth: (int) The number of bits to encode data on
        """
        self.is_fitted = True
        self.set_params(bit_depth=bit_depth)

    def __call__(self, x_val, bit_depth=8):
        """
        Apply feature squeezing to sample x_val.

        :param x_val: (np.ndarray) Sample to squeeze. `x_val` values are supposed to be in the range [0,1]
        :param bit_depth: (int) The number of bits to encode data on
        :return: Squeezed sample
        :rtype: np.ndarray
        """
        self.set_params(bit_depth=bit_depth)

        max_value = int(2 ** bit_depth - 1)
        return np.rint(x_val * max_value) / max_value

    def fit(self, x_val, y_val=None, **kwargs):
        """No parameters to learn for this method; do nothing."""
        pass

    def _tf_predict(self, x, bit_depth=8):
        """
        Apply feature squeezing on tf.Tensor.

        :param x: (tf.Tensor) Sample to squeeze. Values are supposed to be in the range [0,1]
        :param bit_depth: (int) The number of bits to encode data on
        :return: Squeezed sample
        :rtype: tf.Tensor
        """
        self.set_params(bit_depth=bit_depth)

        max_value = int(2 ** bit_depth - 1)
        x = rint(x * max_value) / max_value
        return x

    def set_params(self, **kwargs):
        """Take in a dictionary of parameters and applies defense-specific checks before saving them as attributes.

        Defense-specific parameters:
        :param bit_depth: (int) The number of bits to encode data on
        """
        # Save attack-specific parameters
        super(FeatureSqueezing, self).set_params(**kwargs)

        if type(self.bit_depth) is not int or self.bit_depth <= 0 or self.bit_depth > 60:
            raise ValueError("The bit depth must be between 1 and 60.")

        return True
