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

    def __call__(self, x, y=None, eps=None, pixelcnn=None):
        """
        Apply pixel defence to sample `x`.

        :param x: Sample to defense. `x` values are expected to be in the data range [0, 1].
        :type x: `np.ndarrray`
        :param y: Labels of the sample `x`. This function does not affect them in any way.
        :type y: `np.ndarray`
        :param eps: Defense parameter 0-255.
        :type eps: `int`
        :param pixelcnn: Pre-trained PixelCNN model.
        :type pixelcnn: :class:`.Classifier`
        :return: Purified sample.
        :rtype: `np.ndarray`
        """
        from art.classifiers.classifier import Classifier

        if not isinstance(pixelcnn, Classifier):
            raise ("PixelCNN model must be of type Classifier.")

        clip_values = (0, 1)

        if eps is not None:
            self.set_params(eps=eps)




        return res

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
        """
        # Save defence-specific parameters
        super(PixelDefend, self).set_params(**kwargs)

        if not isinstance(self.eps, (int, np.int)) or self.eps < 0 or self.eps > 255:
            raise ValueError("The defense parameter must be between 0 and 255.")

        return True
