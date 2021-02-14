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
This module implements EoT of changes in brightness.
"""
import logging
from typing import Optional, Tuple, Union, TYPE_CHECKING

import numpy as np

from art.preprocessing.preprocessing import PreprocessorPyTorch

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


class EOTBrightnessPyTorch(PreprocessorPyTorch):
    """
    This module implements EoT of changes in brightness.
    """

    params = ["nb_samples", "brightness"]

    def __init__(
        self,
        nb_samples: int = 1,
        brightness_range: Tuple[float, float] = [0.0, 2.0],
        apply_fit: bool = False,
        apply_predict: bool = True,
    ) -> None:
        """
        Create an instance of EOTBrightnessPyTorch.

        :param nb_samples: Number of random samples per input sample.
        :param brightness_range: Range to sample the factor adjusting the brightness. For example, 0 would gives a
            black image, 1 gives the original image while 2 increases the brightness by a factor of 2. The values are
            sampled uniformly from this range.
        :param apply_fit: True if applied during fitting/training.
        :param apply_predict: True if applied during predicting.
        """
        super().__init__(is_fitted=True, apply_fit=apply_fit, apply_predict=apply_predict)

        self.nb_samples = nb_samples
        self.brightness_range = brightness_range
        self._check_params()

    def forward(
        self, x: "torch.Tensor", y: Optional["torch.Tensor"] = None
    ) -> Tuple["torch.Tensor", Optional["torch.Tensor"]]:
        """
        Apply audio filter to a single sample `x`.

        # :param x: A single audio sample.
        # :param y: Label of the sample `x`. This function does not modify `y`.
        # :return: Similar sample.
        """
        import torch  # lgtm [py/repeated-import]
        import torchvision

        if torch.max(x) > 1.0:
            raise ValueError("Input data `x` has to be float in range [0.0, 1.0].")

        x_preprocess_list = list()
        y_preprocess_list = list()

        for i_image in range(x.shape[0]):
            for i_sample in range(self.nb_samples):
                brightness_factor = np.random.uniform(low=self.brightness_range[0], high=self.brightness_range[1])

                image = x[i_image]

                x_preprocess_i = torchvision.transforms.functional.adjust_brightness(
                    img=image, brightness_factor=brightness_factor
                )
                x_preprocess_list.append(x_preprocess_i)
                if y is not None:
                    y_preprocess_list.append(y[i_image])

        x_preprocess = torch.stack(x_preprocess_list, dim=0)
        if y is None:
            y_preprocess = y
        else:
            y_preprocess = torch.stack(y_preprocess_list, dim=0)

        return x_preprocess, y_preprocess

    def _check_params(self) -> None:

        if not isinstance(self.nb_samples, int) or self.nb_samples < 1:
            raise ValueError("The number of samples needs to be an integer greater than or equal to 1.")

        if (
            not isinstance(self.brightness_range, tuple)
            or self.brightness_range[0] > self.brightness_range[1]
            or len(self.brightness_range) != 2
            or not isinstance(self.brightness_range[0], float)
            or not isinstance(self.brightness_range[1], float)
        ):
            raise ValueError("The argument `brightness_range` has to be a Tuple of two float values as (min, max).")
