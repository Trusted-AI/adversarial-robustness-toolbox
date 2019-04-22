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
from art import NUMPY_DTYPE

logger = logging.getLogger(__name__)


class PixelDefend(Preprocessor):
    """
    Implement the pixel defence approach. Defense based on PixelCNN that projects samples back to the data manifold.
    Paper link: https://arxiv.org/abs/1710.10766
    """
    params = ['eps', 'pixel_cnn']

    def __init__(self, eps=16, pixel_cnn=None):
        """
        Create an instance of pixel defence.

        :param eps: Defense parameter 0-255.
        :type eps: `int`
        :param pixel_cnn: Pre-trained PixelCNN model.
        :type pixel_cnn: :class:`.Classifier`
        """
        super(PixelDefend, self).__init__()
        self._is_fitted = True
        if pixel_cnn is not None:
            self.set_params(eps=eps, pixel_cnn=pixel_cnn)
        else:
            self.set_params(eps=eps)

    @property
    def apply_fit(self):
        return False

    @property
    def apply_predict(self):
        return True

    def __call__(self, x, y=None, eps=None, pixel_cnn=None):
        """
        Apply pixel defence to sample `x`.

        :param x: Sample to defense with shape `(batch_size, width, height, depth)`. `x` values are expected to be in
                the data range [0, 1].
        :type x: `np.ndarrray`
        :param y: Labels of the sample `x`. This function does not affect them in any way.
        :type y: `np.ndarray`
        :param eps: Defense parameter 0-255.
        :type eps: `int`
        :param pixel_cnn: Pre-trained PixelCNN model.
        :type pixel_cnn: :class:`.Classifier`
        :return: Purified sample.
        :rtype: `np.ndarray`
        """
        clip_values = (0, 1)

        params = {}
        if eps is not None:
            params['eps'] = eps

        if pixel_cnn is not None:
            params['pixel_cnn'] = pixel_cnn

        self.set_params(**params)

        # Convert into `uint8`
        x_ = x.copy()
        x_ = x_ * 255
        x_ = x_.astype("uint8")

        # Start defence one image at a time
        for i, xi in enumerate(x_):
            for r in range(x_.shape[1]):
                for c in range(x_.shape[2]):
                    for k in range(x_.shape[3]):
                        # Setup the search space
                        # probs = self.pixel_cnn.predict(np.array([xi / 255.0]), logits=False)
                        probs = self.pixel_cnn.get_activations(np.array([xi / 255.0]), -1)
                        f_probs = probs[0, r, c, k]
                        f_range = range(int(max(xi[r, c, k] - self.eps, 0)), int(min(xi[r, c, k] + self.eps, 255) + 1))

                        # Look in the search space
                        best_prob = -1
                        best_idx = -1
                        for idx in f_range:
                            if f_probs[idx] > best_prob:
                                best_prob = f_probs[idx]
                                best_idx = idx

                        # Update result
                        xi[r, c, k] = best_idx

            # Update in batch
            x_[i] = xi

        # Convert to old dtype
        x_ = x_ / 255.0
        x_ = x_.astype(NUMPY_DTYPE)

        # Clip values into the range [0, 1]
        x_ = np.clip(x_, clip_values[0], clip_values[1])

        return x_, y

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

        Defense-specific parameters:
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

        return True
