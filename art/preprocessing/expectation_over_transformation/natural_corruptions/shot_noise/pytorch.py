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
This module implements EoT of adding shot noise (Poisson) with uniformly sampled rate parameter.
"""
import logging
from typing import Dict, List, Tuple, Union, TYPE_CHECKING, Optional

import numpy as np

from art.preprocessing.expectation_over_transformation.pytorch import EoTPyTorch

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


class EoTShotNoisePyTorch(EoTPyTorch):
    """
    This module implements EoT of adding shot noise (Poisson) with uniformly sampled rate parameter.
    """

    def __init__(
        self,
        nb_samples: int,
        clip_values: Tuple[float, float],
        lam: Union[float, Tuple[float, float]],
        apply_fit: bool = False,
        apply_predict: bool = True,
    ) -> None:
        """
        Create an instance of EoTShotNoisePyTorch.

        :param nb_samples: Number of random samples per input sample.
        :param clip_values: Tuple of float representing minimum and maximum values of input `(min, max)`.
        :param lam: Range to sample the rate of the Poisson distribution. A single float
                    is translated to range [0.0, lam] or a tuple of floats is used to create sampling range
                    [lam[0], lam[1]]. The applied delta is sampled uniformly from this range for each image.
        :param apply_fit: True if applied during fitting/training.
        :param apply_predict: True if applied during predicting.
        """
        super().__init__(
            apply_fit=apply_fit, apply_predict=apply_predict, nb_samples=nb_samples, clip_values=clip_values
        )

        self.lam = lam
        self.lam_range = (0.0, lam) if isinstance(lam, (int, float)) else lam
        self._check_params()

    def _transform(
        self, x: "torch.Tensor", y: Optional[Union["torch.Tensor", List[Dict[str, "torch.Tensor"]]]], **kwargs
    ) -> Tuple["torch.Tensor", Optional[Union["torch.Tensor", List[Dict[str, "torch.Tensor"]]]]]:
        """
        Transformation of an image with randomly sampled shot (Poisson) noise.

        :param x: Input samples.
        :param y: Label of the samples `x`.
        :return: Transformed samples and labels.
        """
        import torch  # lgtm [py/repeated-import]

        lam_i = np.random.uniform(low=self.lam_range[0], high=self.lam_range[1])
        delta_i = torch.poisson(input=torch.ones_like(x) * lam_i) / lam_i * self.clip_values[1]
        return torch.clamp(x + delta_i, min=self.clip_values[0], max=self.clip_values[1]), y

    def _check_params(self) -> None:

        # pylint: disable=R0916
        if not isinstance(self.lam, (int, float, tuple)) or (
            isinstance(self.lam, tuple)
            and (
                len(self.lam) != 2
                or not isinstance(self.lam[0], (int, float))
                or not isinstance(self.lam[1], (int, float))
                or self.lam[0] > self.lam[1]
                or self.lam[0] < 0.0
            )
        ):
            raise ValueError("The argument `lam` has to be a float or tuple of two float values as (min, max).")
