# MIT License
#
# Copyright (C) IBM Corporation 2020
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
This module implements confidence added to the classifier output.
"""
import logging

from art.defences.postprocessor.postprocessor import Postprocessor

logger = logging.getLogger(__name__)


class HighConfidence(Postprocessor):
    """
    Implementation of a postprocessor based on selecting high confidence predictions to return as classifier output.
    """

    params = ["cutoff"]

    def __init__(self, cutoff=0.25, apply_fit=False, apply_predict=True):
        """
        Create a HighConfidence postprocessor.

        :param cutoff: Minimal value for returned prediction output.
        :type cutoff: `float`
        :param apply_fit: True if applied during fitting/training.
        :type apply_fit: `bool`
        :param apply_predict: True if applied during predicting.
        :type apply_predict: `bool`
        """
        super(HighConfidence, self).__init__()
        self._is_fitted = True
        self._apply_fit = apply_fit
        self._apply_predict = apply_predict
        self.set_params(cutoff=cutoff)

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
        post_preds = preds.copy()
        post_preds[post_preds < self.cutoff] = 0.0

        return post_preds

    def fit(self, preds, **kwargs):
        """
        No parameters to learn for this method; do nothing.
        """
        pass

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and apply checks before saving them as attributes.

        :param cutoff: Minimal value for returned prediction output.
        :type cutoff: `float`
        :return: `True` when parsing was successful
        """
        # Save defence-specific parameters
        super(HighConfidence, self).set_params(**kwargs)

        if self.cutoff <= 0:
            raise ValueError("Minimal value must be positive.")

        return True
