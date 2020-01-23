# MIT License
#
# Copyright (C) IBM Corporation 2019
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
This module implements Gaussian noise added to the classifier output.
"""
import logging

import numpy as np

from art.defences import Postprocessor
from art.utils import is_probability

logger = logging.getLogger(__name__)


class GaussianNoise(Postprocessor):
    """
    Implementation of a postprocessor based on adding Gaussian noise to classifier output.
    """

    def __init__(self, scale=0.2, apply_fit=False, apply_predict=True):
        """
        Create a GaussianNoise postprocessor.

        :param scale: Standard deviation of the distribution.
        :type scale: `float`
        :param apply_fit: True if applied during fitting/training.
        :type apply_fit: `bool`
        :param apply_predict: True if applied during predicting.
        :type apply_predict: `bool`
        """
        super(GaussianNoise, self).__init__()
        self._is_fitted = True
        self._apply_fit = apply_fit
        self._apply_predict = apply_predict
        self.set_params(scale=scale)

    @property
    def apply_fit(self):
        return self._apply_fit

    @property
    def apply_predict(self):
        return self._apply_predict

    def __call__(self, preds):
        """
        Perform model postprocessing and return postprocessed output.

        :param preds: model output to be postprocessed.
        :type preds: `np.ndarray`
        :return: Postprocessed model output.
        :rtype: `np.ndarray`
        """
        # Generate random noise
        noise = np.random.normal(loc=0.0, scale=self.scale, size=preds.shape)

        if preds.shape[1] > 1:
            # Check if model output is logit or probability
            are_probability = [is_probability(x) for x in preds]
            all_probability = np.sum(are_probability) == preds.shape[0]

            # Add noise to model output
            preds += noise

            # Finally normalize probability output
            if all_probability:
                preds[preds < 0.0] = 0.0
                sums = np.sum(preds, axis=1)
                preds /= sums

        else:
            # Add noise to model output
            preds += noise
            preds[preds < 0.0] = 0.0

        return preds

    def estimate_gradient(self, x, grad):
        """
        Provide an estimate of the gradients of the defence for the backward pass. If the defence is not differentiable,
        this is an estimate of the gradient, most often replacing the computation performed by the defence with the
        identity function.

        :param x: Input data for which the gradient is estimated. First dimension is the batch size.
        :type x: `np.ndarray`
        :param grad: Gradient value so far.
        :type grad: `np.ndarray`
        :return: The gradient (estimate) of the defence.
        :rtype: `np.ndarray`
        """
        return grad
