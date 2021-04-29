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
This module implements the filter function for audio signals in PyTorch. It provides with an infinite impulse response
(IIR) or finite impulse response (FIR) filter. This implementation is a wrapper around the
`torchaudio.functional.lfilter` function in the `torchaudio` package.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np
from tqdm.auto import tqdm

from art.preprocessing.preprocessing import PreprocessorPyTorch

if TYPE_CHECKING:
    # pylint: disable=C0412
    import torch
    from art.utils import CLIP_VALUES_TYPE

logger = logging.getLogger(__name__)


class LFilterPyTorch(PreprocessorPyTorch):
    """
    This module implements the filter function for audio signals in PyTorch. It provides with an infinite impulse
    response (IIR) or finite impulse response (FIR) filter. This implementation is a wrapper around the
    `torchaudio.functional.lfilter` function in the `torchaudio` package.
    """

    params = ["numerator_coef", "denominator_coef", "verbose"]

    def __init__(
        self,
        numerator_coef: np.ndarray = np.array([1.0]),
        denominator_coef: np.ndarray = np.array([1.0]),
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        apply_fit: bool = False,
        apply_predict: bool = True,
        verbose: bool = False,
        device_type: str = "gpu",
    ) -> None:
        """
        Create an instance of LFilterPyTorch.

        :param numerator_coef: The numerator coefficient vector in a 1-D sequence.
        :param denominator_coef: The denominator coefficient vector in a 1-D sequence. By simply setting the array of
                                 denominator coefficients to np.array([1.0]), this preprocessor can be used to apply a
                                 FIR filter.
        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :param apply_fit: True if applied during fitting/training.
        :param apply_predict: True if applied during predicting.
        :param verbose: Show progress bars.
        :param device_type: Type of device on which the classifier is run, either `gpu` or `cpu`.
        """
        import torch  # lgtm [py/repeated-import]

        super().__init__(is_fitted=True, apply_fit=apply_fit, apply_predict=apply_predict)

        self.numerator_coef = numerator_coef
        self.denominator_coef = denominator_coef
        self.clip_values = clip_values
        self.verbose = verbose
        self._check_params()

        # Set device
        if device_type == "cpu" or not torch.cuda.is_available():
            self._device = torch.device("cpu")
        else:
            cuda_idx = torch.cuda.current_device()
            self._device = torch.device("cuda:{}".format(cuda_idx))

    def forward(
        self, x: "torch.Tensor", y: Optional["torch.Tensor"] = None
    ) -> Tuple["torch.Tensor", Optional["torch.Tensor"]]:
        """
        Apply filter to a single sample `x`.

        :param x: A single audio sample.
        :param y: Label of the sample `x`. This function does not affect them in any way.
        :return: Similar sample.
        """
        import torch  # lgtm [py/repeated-import]
        import torchaudio

        if int(torchaudio.__version__.split(".")[1]) > 5:
            x_preprocess = torchaudio.functional.lfilter(
                b_coeffs=torch.tensor(self.numerator_coef, device=self._device, dtype=x.dtype),
                a_coeffs=torch.tensor(self.denominator_coef, device=self._device, dtype=x.dtype),
                waveform=x,
                clamp=False,
            )
        else:
            x_preprocess = torchaudio.functional.lfilter(
                b_coeffs=torch.tensor(self.numerator_coef, device=self._device, dtype=x.dtype),
                a_coeffs=torch.tensor(self.denominator_coef, device=self._device, dtype=x.dtype),
                waveform=x,
            )

        if self.clip_values is not None:
            x_preprocess = x_preprocess.clamp(min=self.clip_values[0], max=self.clip_values[1])

        return x_preprocess, y

    def __call__(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply filter to sample `x`.

        :param x: Samples of shape (nb_samples, seq_length). Note that, it is allowable that sequences in the batch
                  could have different lengths. A possible example of `x` could be:
                  `x = np.array([np.array([0.1, 0.2, 0.1, 0.4]), np.array([0.3, 0.1])])`.
        :param y: Labels of the sample `x`. This function does not affect them in any way.
        :return: Similar samples.
        """
        import torch  # lgtm [py/repeated-import]

        x_preprocess = x.copy()

        # Filter one input at a time
        for i, x_preprocess_i in enumerate(tqdm(x_preprocess, desc="Apply audio filter", disable=not self.verbose)):
            if np.min(x_preprocess_i) < -1.0 or np.max(x_preprocess_i) > 1.0:
                raise ValueError(
                    "Audio signals must be normalized to the range `[-1.0, 1.0]` to apply the audio filter function."
                )

            x_preprocess_i = torch.tensor(x_preprocess_i, device=self._device)

            with torch.no_grad():
                x_preprocess_i, _ = self.forward(x_preprocess_i)

            x_preprocess[i] = x_preprocess_i.cpu().numpy()

        return x_preprocess, y

    def _check_params(self) -> None:
        if not isinstance(self.denominator_coef, np.ndarray) or self.denominator_coef[0] == 0:
            raise ValueError("The first element of the denominator coefficient vector must be non zero.")

        if self.clip_values is not None:
            if len(self.clip_values) != 2:
                raise ValueError("`clip_values` should be a tuple of 2 floats containing the allowed data range.")

            if np.array(self.clip_values[0] >= self.clip_values[1]).any():
                raise ValueError("Invalid `clip_values`: min >= max.")

        if not isinstance(self.numerator_coef, np.ndarray):
            raise ValueError("The numerator coefficient vector has to be of type `np.ndarray`.")

        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be of type bool.")

        if len(self.denominator_coef) != len(self.numerator_coef):
            raise ValueError(
                "The denominator coefficient vector and the numerator coefficient vector must have the same length."
            )
