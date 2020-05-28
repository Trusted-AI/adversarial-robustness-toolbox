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
This module implements the resampling defence `Resample`.

| Paper link: https://arxiv.org/abs/1809.10875

| Please keep in mind the limitations of defences. For details on how to evaluate classifier security in general,
    see https://arxiv.org/abs/1902.06705.
"""

import logging

import numpy as np

from art.defences.preprocessor.preprocessor import Preprocessor

logger = logging.getLogger(__name__)


class Resample(Preprocessor):
    """
    Implement the resampling defense approach.

    Resampling implicitly consists of a step that applies a lowpass filter. The underlying filter in this implementation
    is a Windowed Sinc Interpolation function.
    """

    params = ["sr_original", "sr_new", "channel_index"]

    def __init__(self, sr_original, sr_new, channel_index=2, apply_fit=True, apply_predict=False):
        """
        Create an instance of the resample preprocessor.

        :param sr_original: Original sampling rate of sample.
        :type sr_original: `int`
        :param sr_new: New sampling rate of sample.
        :type sr_new: `int`
        :param channel_index: Index of the axis containing the audio channels.
        :type channel_index: `int`
        :param apply_fit: True if applied during fitting/training.
        :type apply_fit: `bool`
        :param apply_predict: True if applied during predicting.
        :type apply_predict: `bool`
        """
        super().__init__()
        self._is_fitted = True
        self._apply_fit = apply_fit
        self._apply_predict = apply_predict
        self.set_params(sr_original=sr_original, sr_new=sr_new, channel_index=channel_index)

    @property
    def apply_fit(self):
        return self._apply_fit

    @property
    def apply_predict(self):
        return self._apply_predict

    def __call__(self, x, y=None):
        """
        Resample `x` to a new sampling rate.

        :param x: Sample to resample of shape `(batch_size, length, channel)` or `(batch_size,
        channel, length)`.
        :type x: `np.ndarray`
        :param y: Labels of the sample `x`. This function does not affect them in any way.
        :type y: `np.ndarray`
        :return: Resampled audio sample.
        :rtype: `np.ndarray`
        """
        import resampy

        if x.ndim != 3:
            raise ValueError("Resampling can only be applied to temporal data across at least one channel.")

        sample_index = 2 if self.channel_index == 1 else 1

        return resampy.resample(x, self.sr_original, self.sr_new, axis=sample_index, filter="sinc_window"), y

    def estimate_gradient(self, x, grad):
        return grad

    def fit(self, x, y=None, **kwargs):
        """
        No parameters to learn for this method; do nothing.
        """
        pass

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and apply defence-specific checks before saving them as attributes.
        """
        super().set_params(**kwargs)

        if not (isinstance(self.channel_index, (int, np.int)) and self.channel_index in [1, 2]):
            raise ValueError(
                "Data channel must be an integer equal to 1 or 2. The batch dimension is not a valid channel."
            )

        if not (isinstance(self.sr_original, (int, np.int)) and self.sr_original > 0):
            raise ValueError("Original sampling rate be must a positive integer.")

        if not (isinstance(self.sr_new, (int, np.int)) and self.sr_new > 0):
            raise ValueError("New sampling rate be must a positive integer.")
        return True
