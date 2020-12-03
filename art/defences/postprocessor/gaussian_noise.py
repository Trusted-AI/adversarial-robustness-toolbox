# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
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

from art.defences.postprocessor.postprocessor import Postprocessor
from art.utils import is_probability

logger = logging.getLogger(__name__)


class GaussianNoise(Postprocessor):
    """
    Implementation of a postprocessor based on adding Gaussian noise to classifier output.
    """

    params = ["scale"]

    def __init__(self, scale: float = 0.2, apply_fit: bool = False, apply_predict: bool = True) -> None:
        """
        Create a GaussianNoise postprocessor.

        :param scale: Standard deviation of the distribution.
        :param apply_fit: True if applied during fitting/training.
        :param apply_predict: True if applied during predicting.
        """
        super().__init__(is_fitted=True, apply_fit=apply_fit, apply_predict=apply_predict)
        self.scale = scale
        self._check_params()

    def __call__(self, preds: np.ndarray) -> np.ndarray:
        """
        Perform model postprocessing and return postprocessed output.

        :param preds: model output to be postprocessed.
        :return: Postprocessed model output.
        """
        # Generate random noise
        noise = np.random.normal(loc=0.0, scale=self.scale, size=preds.shape)

        # Add noise to model output
        post_preds = preds.copy()
        post_preds += noise

        if preds.shape[1] > 1:
            # Check if model output is logits or probability
            are_probability = [is_probability(x) for x in preds]
            all_probability = np.sum(are_probability) == preds.shape[0]

            # Finally normalize probability output
            if all_probability:
                post_preds[post_preds < 0.0] = 0.0
                sums = np.sum(post_preds, axis=1)
                post_preds /= sums
        else:
            post_preds[post_preds < 0.0] = 0.0

        return post_preds

    def _check_params(self) -> None:
        if self.scale <= 0:
            raise ValueError("Standard deviation must be positive.")
