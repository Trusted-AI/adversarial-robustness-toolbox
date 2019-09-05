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
This module implement the pixel defence in `PixelDefend`. It is based on PixelCNN that projects samples back to the data
manifold.

| Paper link: https://arxiv.org/abs/1710.10766
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np

from art.defences.preprocessor import Preprocessor
from art import NUMPY_DTYPE

logger = logging.getLogger(__name__)


class PixelDefend(Preprocessor):
    """
    Implement the pixel defence approach. Defense based on PixelCNN that projects samples back to the data manifold.

    | Paper link: https://arxiv.org/abs/1710.10766
    """
    params = ['clip_values', 'eps', 'pixel_cnn']

    def __init__(self, clip_values=(0, 1), eps=16, pixel_cnn=None, apply_fit=False, apply_predict=True):
        """
        Create an instance of pixel defence.

        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :type clip_values: `tuple`
        :param eps: Defense parameter 0-255.
        :type eps: `int`
        :param pixel_cnn: Pre-trained PixelCNN model.
        :type pixel_cnn: :class:`.Classifier`
        """
        super(PixelDefend, self).__init__()
        self._is_fitted = True
        self._apply_fit = apply_fit
        self._apply_predict = apply_predict
        if pixel_cnn is not None:
            self.set_params(clip_values=clip_values, eps=eps, pixel_cnn=pixel_cnn)
        else:
            self.set_params(clip_values=clip_values, eps=eps)

    @property
    def apply_fit(self):
        return self._apply_fit

    @property
    def apply_predict(self):
        return self._apply_predict

    def __call__(self, x, y=None):
        """
        Apply pixel defence to sample `x`.

        :param x: Sample to defense with shape `(batch_size, width, height, depth)`. `x` values are expected to be in
                the data range [0, 1].
        :type x: `np.ndarrray`
        :param y: Labels of the sample `x`. This function does not affect them in any way.
        :type y: `np.ndarray`
        :return: Purified sample.
        :rtype: `np.ndarray`
        """
        # Convert into `uint8`
        original_shape = x.shape
        probs = self.pixel_cnn.get_activations(x, layer=-1).reshape((x.shape[0], -1, 256))
        x = x * 255
        x = x.astype("uint8")
        x = x.reshape((x.shape[0], -1))

        # Start defence one image at a time
        for i, x_i in enumerate(x):
            for feat_index in range(x.shape[1]):
                # Setup the search space
                f_probs = probs[i, feat_index, :]
                f_range = range(int(max(x_i[feat_index] - self.eps, 0)), int(min(x_i[feat_index] + self.eps, 255) + 1))

                # Look in the search space
                best_prob = -1
                best_idx = -1
                for idx in f_range:
                    if f_probs[idx] > best_prob:
                        best_prob = f_probs[idx]
                        best_idx = idx

                # Update result
                x_i[feat_index] = best_idx

            # Update in batch
            x[i] = x_i

        # Convert to old dtype
        x = x / 255.0
        x = x.astype(NUMPY_DTYPE).reshape(original_shape)

        # Clip to clip_values
        x = np.clip(x, self.clip_values[0], self.clip_values[1])

        return x, y

    def estimate_gradient(self, x, grad):
        raise grad

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
        :param eps: Defense parameter 0-255.
        :type eps: `int`
        :param pixel_cnn: Pre-trained PixelCNN model.
        :type pixel_cnn: :class:`.Classifier`
        """
        from art.classifiers import Classifier

        # Save defence-specific parameters
        super(PixelDefend, self).set_params(**kwargs)

        if not isinstance(self.eps, (int, np.int)) or self.eps < 0 or self.eps > 255:
            raise ValueError("The defense parameter must be between 0 and 255.")

        if hasattr(self, 'pixel_cnn') and not isinstance(self.pixel_cnn, Classifier):
            raise TypeError("PixelCNN model must be of type Classifier.")

        if np.array(self.clip_values[0] >= self.clip_values[1]).any():
            raise ValueError('Invalid `clip_values`: min >= max.')

        if self.clip_values[0] != 0:
            raise ValueError('`clip_values` min value must be 0.')

        if self.clip_values[1] != 1:
            raise ValueError('`clip_values` max value must be 1.')

        return True
