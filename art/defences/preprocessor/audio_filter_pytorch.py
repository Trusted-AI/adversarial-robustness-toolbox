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

from torchaudio.functional import lfilter
import numpy as np

from art.defences.preprocessor.preprocessor import PreprocessorPyTorch

if TYPE_CHECKING:
    import torch
    from art.utils import CLIP_VALUES_TYPE

logger = logging.getLogger(__name__)


class AudioFilterPyTorch(PreprocessorPyTorch):
    """
    This module implements the filter function for audio signals in PyTorch. It provides with an infinite impulse
    response (IIR) or finite impulse response (FIR) filter. This implementation is a wrapper around the
    `torchaudio.functional.lfilter` function in the `torchaudio` package.
    """

    params = ["numerator_coef", "denominator_coef", "verbose"]

    def __init__(
        self,
        numerator_coef: np.ndarray,
        denumerator_coef: np.ndarray,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        apply_fit: bool = False,
        apply_predict: bool = True,
        verbose: bool = False,
        device_type: str = "gpu",
    ) -> None:
        """
        Create an instance of AudioFilterPyTorch.

        :param numerator_coef: The numerator coefficient vector in a 1-D sequence.
        :param denominator_coef: The denominator coefficient vector in a 1-D sequence. By simply setting the array of
                                 denominator coefficients to [1, 0, 0,...], this preprocessor can be used to apply a
                                 FIR filter.
        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :param apply_fit: True if applied during fitting/training.
        :param apply_predict: True if applied during predicting.
        :param verbose: Show progress bars.
        :param device_type: Type of device on which the classifier is run, either `gpu` or `cpu`.
        """
        import torch  # lgtm [py/repeated-import]

        super().__init__()

        self._apply_fit = apply_fit
        self._apply_predict = apply_predict
        self.numerator_coef = numerator_coef
        self.denumerator_coef = denumerator_coef
        self.clip_values = clip_values
        self.verbose = verbose
        self._check_params()

        # Set device
        if device_type == "cpu" or not torch.cuda.is_available():
            self._device = torch.device("cpu")
        else:
            cuda_idx = torch.cuda.current_device()
            self._device = torch.device("cuda:{}".format(cuda_idx))

    @property
    def apply_fit(self) -> bool:
        return self._apply_fit

    @property
    def apply_predict(self) -> bool:
        return self._apply_predict

    def forward(
        self, x: "torch.Tensor", y: Optional["torch.Tensor"] = None
    ) -> Tuple["torch.Tensor", Optional["torch.Tensor"]]:
        """
        Apply local spatial smoothing to sample `x`.
        """
        x_ndim = x.ndim

        # NHWC/NCFHW/NFHWC --> NCHW.
        if x_ndim == 4:
            if self.channels_first:
                x_nchw = x
            else:
                # NHWC --> NCHW
                x_nchw = x.permute(0, 3, 1, 2)
        elif x_ndim == 5:
            if self.channels_first:
                # NCFHW --> NFCHW --> NCHW
                nb_clips, channels, clip_size, height, width = x.shape
                x_nchw = x.permute(0, 2, 1, 3, 4).reshape(nb_clips * clip_size, channels, height, width)
            else:
                # NFHWC --> NHWC --> NCHW
                nb_clips, clip_size, height, width, channels = x.shape
                x_nchw = x.reshape(nb_clips * clip_size, height, width, channels).permute(0, 3, 1, 2)
        else:
            raise ValueError(
                "Unrecognized input dimension. Spatial smoothing can only be applied to image and video data."
            )

        x_nchw = self.median_blur(x_nchw)

        # NHWC/NCFHW/NFHWC <-- NCHW.
        if x_ndim == 4:
            if self.channels_first:
                x = x_nchw
            else:
                #   NHWC <-- NCHW
                x = x_nchw.permute(0, 2, 3, 1)
        elif x_ndim == 5:  # lgtm [py/redundant-comparison]
            if self.channels_first:
                # NCFHW <-- NFCHW <-- NCHW
                x_nfchw = x_nchw.reshape(nb_clips, clip_size, channels, height, width)
                x = x_nfchw.permute(0, 2, 1, 3, 4)
            else:
                # NFHWC <-- NHWC <-- NCHW
                x_nhwc = x_nchw.permute(0, 2, 3, 1)
                x = x_nhwc.reshape(nb_clips, clip_size, height, width, channels)

        if self.clip_values is not None:
            x = x.clamp(min=self.clip_values[0], max=self.clip_values[1])

        return x, y

    def estimate_forward(self, x: "torch.Tensor", y: Optional["torch.Tensor"] = None) -> "torch.Tensor":
        """
        No need to estimate, since the forward pass is differentiable.
        """
        return self.forward(x, y)[0]

    def __call__(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply local spatial smoothing to sample `x`.

        :param x: Sample to smooth with shape `(batch_size, width, height, depth)`.
        :param y: Labels of the sample `x`. This function does not affect them in any way.
        :return: Smoothed sample.
        """
        import torch  # lgtm [py/repeated-import]

        x = torch.tensor(x, device=self._device)
        if y is not None:
            y = torch.tensor(y, device=self._device)

        with torch.no_grad():
            x, y = self.forward(x, y)

        result = x.cpu().numpy()
        if y is not None:
            y = y.cpu().numpy()
        return result, y

    # Backward compatibility.
    def estimate_gradient(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        import torch  # lgtm [py/repeated-import]

        x = torch.tensor(x, device=self._device, requires_grad=True)
        grad = torch.tensor(grad, device=self._device)

        x_prime = self.estimate_forward(x)
        x_prime.backward(grad)
        x_grad = x.grad.detach().cpu().numpy()
        if x_grad.shape != x.shape:
            raise ValueError("The input shape is {} while the gradient shape is {}".format(x.shape, x_grad.shape))
        return x_grad

    def fit(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> None:
        """
        No parameters to learn for this method; do nothing.
        """
        pass

    def _check_params(self) -> None:
        if not (isinstance(self.window_size, (int, np.int)) and self.window_size > 0):
            raise ValueError("Sliding window size must be a positive integer.")

        if self.clip_values is not None and len(self.clip_values) != 2:
            raise ValueError("'clip_values' should be a tuple of 2 floats or arrays containing the allowed data range.")

        if self.clip_values is not None and np.array(self.clip_values[0] >= self.clip_values[1]).any():
            raise ValueError("Invalid 'clip_values': min >= max.")
