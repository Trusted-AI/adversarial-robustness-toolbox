# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2021
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
This module implements EoT of changes in contrast with uniformly sampled factor.
"""
import logging
from typing import Tuple, Union, TYPE_CHECKING, Optional

import numpy as np

from art.preprocessing.expectation_over_transformation.pytorch import EoTPyTorch

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


class EoTContrastPyTorch(EoTPyTorch):
    """
    This module implements EoT of changes in contrast with uniformly sampled factor.
    """

    def __init__(
        self,
        nb_samples: int,
        clip_values: Tuple[float, float],
        contrast_factor: Union[float, Tuple[float, float]],
        apply_fit: bool = False,
        apply_predict: bool = True,
    ) -> None:
        """
        Create an instance of EoTContrastPyTorch.

        :param nb_samples: Number of random samples per input sample.
        :param clip_values: Tuple of float representing minimum and maximum values of input `(min, max)`.
        :param contrast_factor: Range to sample the contrast factor adjust the contrast. A single float is translated to
               range [-delta, delta] or a tuple of floats is used to create sampling range [delta[0], delta[1]]. The
               applied delta is sampled uniformly from this range for each image.
        :param apply_fit: True if applied during fitting/training.
        :param apply_predict: True if applied during predicting.
        """
        super().__init__(
            apply_fit=apply_fit, apply_predict=apply_predict, nb_samples=nb_samples, clip_values=clip_values
        )

        self.contrast_factor = contrast_factor
        self.contrast_factor_range = (
            (0, contrast_factor) if isinstance(contrast_factor, (int, float)) else contrast_factor
        )
        self._check_params()

    def _transform(
        self, x: "torch.Tensor", y: Optional["torch.Tensor"], **kwargs
    ) -> Tuple["torch.Tensor", Optional["torch.Tensor"]]:
        """
        Transformation of an image with randomly sampled contrast.

        :param x: Input samples.
        :param y: Label of the samples `x`.
        :return: Transformed samples and labels.
        """
        import torch  # lgtm [py/repeated-import]

        contrast_factor_i = np.random.uniform(low=self.contrast_factor_range[0], high=self.contrast_factor_range[1])
        if x.shape[2] == 3:
            red, green, blue = x[:, :, 0], x[:, :, 1], x[:, :, 2]
            x_gray = 0.2989 * red + 0.587 * green + 0.114 * blue
        elif x.shape[2] == 1:
            x_gray = x[:, :, 0]
        else:
            raise ValueError("Number of color channels is not 1 or 3 in input `x` of format HWC.")
        mean = torch.mean(x_gray, dim=(-2, -1), keepdim=True)

        return (
            torch.clamp(
                contrast_factor_i * x + (1.0 - contrast_factor_i) * mean,
                min=self.clip_values[0],
                max=self.clip_values[1],
            ),
            y,
        )

    def _check_params(self) -> None:

        # pylint: disable=R0916
        if not isinstance(self.contrast_factor, (int, float, tuple)) or (
            isinstance(self.contrast_factor, tuple)
            and (
                len(self.contrast_factor) != 2
                or not isinstance(self.contrast_factor[0], (int, float))
                or not isinstance(self.contrast_factor[1], (int, float))
                or self.contrast_factor[0] > self.contrast_factor[1]
            )
        ):
            raise ValueError(
                "The argument `contrast_factor` has to be a float or tuple of two float values as (min, max)."
            )
