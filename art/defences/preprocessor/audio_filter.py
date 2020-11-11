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
This module implements the filter function for audio signals. It provides with an infinite impulse response (IIR) or
finite impulse response (FIR) filter. This implementation is a wrapper around the `scipy.signal.lfilter` function in
the `scipy` package.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm

from art.config import ART_NUMPY_DTYPE
from art.defences.preprocessor.preprocessor import Preprocessor

if TYPE_CHECKING:
    from art.utils import CLIP_VALUES_TYPE

logger = logging.getLogger(__name__)


class AudioFilter(Preprocessor):
    """
    This module implements the filter function for audio signals. It provides with an infinite impulse response (IIR)
    or finite impulse response (FIR) filter. This implementation is a wrapper around the `scipy.signal.lfilter`
    function in the `scipy` package.
    """

    params = ["numerator_coef", "denominator_coef", "axis", "initial_cond"]

    def __init__(
        self,
        numerator_coef: np.ndarray,
        denumerator_coef: np.ndarray,
        axis: int = -1,
        initial_cond: Optional[np.ndarray] = None,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        apply_fit: bool = False,
        apply_predict: bool = True,
    ):
        """
        Create an instance of AudioFilter.

        :param numerator_coef: The numerator coefficient vector in a 1-D sequence.
        :param denominator_coef: The denominator coefficient vector in a 1-D sequence. By simply setting the array of
                                 denominator coefficients to [1, 0, 0,...], this preprocessor can be used to apply a
                                 FIR filter.
        :param axis: The axis of the input data array along which to apply the linear filter. The filter is applied to
                     each subarray along this axis.
        :param initial_cond: Initial conditions for the filter delays.
        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :param apply_fit: True if applied during fitting/training.
        :param apply_predict: True if applied during predicting.
        """
        super().__init__()
        self._is_fitted = True
        self._apply_fit = apply_fit
        self._apply_predict = apply_predict
        self.numerator_coef = numerator_coef
        self.denumerator_coef = denumerator_coef
        self.axis = axis
        self.initial_cond = initial_cond
        self.clip_values = clip_values
        self._check_params()

    @property
    def apply_fit(self) -> bool:
        return self._apply_fit

    @property
    def apply_predict(self) -> bool:
        return self._apply_predict

    def __call__(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply total variance minimization to sample `x`.

        :param x: Sample to compress with shape `(batch_size, width, height, depth)`.
        :param y: Labels of the sample `x`. This function does not affect them in any way.
        :return: Similar samples.
        """
        if len(x.shape) == 2:
            raise ValueError(
                "Feature vectors detected. Variance minimization can only be applied to data with spatial dimensions."
            )
        x_preproc = x.copy()

        # Minimize one input at a time
        for i, x_i in enumerate(tqdm(x_preproc, desc="Variance minimization", disable=not self.verbose)):
            mask = (np.random.rand(*x_i.shape) < self.prob).astype("int")
            x_preproc[i] = self._minimize(x_i, mask)

        if self.clip_values is not None:
            np.clip(x_preproc, self.clip_values[0], self.clip_values[1], out=x_preproc)

        return x_preproc.astype(ART_NUMPY_DTYPE), y

    def estimate_gradient(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        return grad

    def fit(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> None:
        """
        No parameters to learn for this method; do nothing.
        """
        pass

    def _check_params(self) -> None:
        if not isinstance(self.denumerator_coef, np.ndarray) or self.denumerator_coef[0] == 0:
            raise ValueError("The first element of the denominator coefficient vector must be non zero.")

        if self.clip_values is not None:
            if len(self.clip_values) != 2:
                raise ValueError("`clip_values` should be a tuple of 2 floats containing the allowed data range.")

            if np.array(self.clip_values[0] >= self.clip_values[1]).any():
                raise ValueError("Invalid `clip_values`: min >= max.")

        if not isinstance(self.numerator_coef, np.ndarray):
            raise ValueError("The numerator coefficient vector has to be of type `np.ndarray`.")

        if not isinstance(self.axis, int):
            raise ValueError("The axis of the input data array has to be of type `int`.")

        if self.initial_cond is not None and not isinstance(self.initial_cond, np.ndarray):
            raise ValueError("The initial conditions for the filter delays must be of type `np.ndarray`.")
