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
This module implements EoT of adding Gaussian noise with uniformly sampled standard deviation.
"""
import logging
from typing import Tuple, Union, TYPE_CHECKING, Optional

import numpy as np

from art.preprocessing.expectation_over_transformation.tensorflow import EoTTensorFlowV2

if TYPE_CHECKING:
    import tensorflow as tf

logger = logging.getLogger(__name__)


class EoTGaussianNoiseTensorFlow(EoTTensorFlowV2):
    """
    This module implements EoT of adding Gaussian noise with uniformly sampled standard deviation.
    """

    def __init__(
        self,
        nb_samples: int,
        clip_values: Tuple[float, float],
        std: Union[float, Tuple[float, float]],
        apply_fit: bool = False,
        apply_predict: bool = True,
    ) -> None:
        """
        Create an instance of EoTGaussianNoiseTensorFlow.

        :param nb_samples: Number of random samples per input sample.
        :param clip_values: Tuple of float representing minimum and maximum values of input `(min, max)`.
        :param std: Range to sample the standard deviation for the Gaussian distribution. A single float
                    is translated to range [0, std]. The applied delta is sampled uniformly from this range for each
                    image.
        :param apply_fit: True if applied during fitting/training.
        :param apply_predict: True if applied during predicting.
        """
        super().__init__(
            apply_fit=apply_fit, apply_predict=apply_predict, nb_samples=nb_samples, clip_values=clip_values
        )

        self.std = std
        self.std_range = (0.0, std) if isinstance(std, (int, float)) else std
        self._check_params()

    def _transform(
        self, x: "tf.Tensor", y: Optional["tf.Tensor"], **kwargs
    ) -> Tuple["tf.Tensor", Optional["tf.Tensor"]]:
        """
        Transformation of an image with randomly sampled Gaussian noise.

        :param x: Input samples.
        :param y: Label of the samples `x`.
        :return: Transformed samples and labels.
        """
        import tensorflow as tf  # lgtm [py/repeated-import]

        std_i = np.random.uniform(low=self.std_range[0], high=self.std_range[1])
        delta_i = tf.random.normal(shape=x.shape, mean=0.0, stddev=std_i, seed=None)
        return tf.clip_by_value(x + delta_i, clip_value_min=self.clip_values[0], clip_value_max=self.clip_values[1]), y

    def _check_params(self) -> None:

        # pylint: disable=R0916
        if not isinstance(self.std, (int, float, tuple)) or (
            isinstance(self.std, tuple)
            and (
                len(self.std) != 2
                or not isinstance(self.std[0], (int, float))
                or not isinstance(self.std[1], (int, float))
                or self.std[0] > self.std[1]
                or self.std[0] < 0.0
            )
        ):
            raise ValueError("The argument `std` has to be a float or tuple of two float values as (min, max).")
