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
This module implements EoT of changes in brightness by addition of uniformly sampled delta.
"""
import logging
from typing import Optional, Tuple, Union, TYPE_CHECKING

import numpy as np

from art.preprocessing.preprocessing import PreprocessorTensorFlowV2

if TYPE_CHECKING:
    import tensorflow as tf

logger = logging.getLogger(__name__)


class EOTBrightnessTensorFlowV2(PreprocessorTensorFlowV2):
    """
    This module implements EoT of changes in brightness by addition of uniformly sampled delta.
    """

    params = ["nb_samples", "brightness"]

    def __init__(
        self,
        nb_samples: int,
        clip_values: Tuple[float, float],
        delta: Union[float, Tuple[float, float]],
        apply_fit: bool = False,
        apply_predict: bool = True,
    ) -> None:
        """
        Create an instance of EOTBrightnessTensorFlowV2.

        :param nb_samples: Number of random samples per input sample.
        :param clip_values: Tuple of float representing minimum and maximum values of input `(min, max)`.
        :param delta: Range to sample the delta (addition) to the pixel values to adjust the brightness. A single float
            is translated to range [-delta, delta] or a tuple of floats is used to create sampling range
            [delta[0], delta[1]]. The applied delta is sampled uniformly from this range for each image.
        :param apply_fit: True if applied during fitting/training.
        :param apply_predict: True if applied during predicting.
        """
        super().__init__(is_fitted=True, apply_fit=apply_fit, apply_predict=apply_predict)

        self.nb_samples = nb_samples
        self.clip_values = clip_values
        self.delta = delta
        self.delta_range = (-delta, delta) if isinstance(delta, float) else delta
        self._check_params()

    def forward(self, x: "tf.Tensor", y: Optional["tf.Tensor"] = None) -> Tuple["tf.Tensor", Optional["tf.Tensor"]]:
        """
        Apply audio filter to a single sample `x`.

        # :param x: A single audio sample.
        # :param y: Label of the sample `x`. This function does not modify `y`.
        # :return: Similar sample.
        """
        import tensorflow as tf  # lgtm [py/repeated-import]

        if tf.reduce_max(x) > 1.0:
            raise ValueError("Input data `x` has to be float in range [0.0, 1.0].")

        x_preprocess_list = list()
        y_preprocess_list = list()

        for i_image in range(x.shape[0]):
            for i_sample in range(self.nb_samples):
                delta_i = np.random.uniform(low=self.delta_range[0], high=self.delta_range[1])
                x_i = x[i_image]
                x_preprocess_i = tf.clip_by_value(
                    x_i + delta_i, clip_value_min=self.clip_values[0], clip_value_max=self.clip_values[1]
                )
                x_preprocess_list.append(x_preprocess_i)

                if y is not None:
                    y_preprocess_list.append(y[i_image])

        x_preprocess = tf.stack(x_preprocess_list, axis=0)
        if y is None:
            y_preprocess = y
        else:
            y_preprocess = tf.stack(y_preprocess_list, axis=0)

        return x_preprocess, y_preprocess

    def _check_params(self) -> None:

        if not isinstance(self.nb_samples, int) or self.nb_samples < 1:
            raise ValueError("The number of samples needs to be an integer greater than or equal to 1.")

        if not isinstance(self.clip_values, tuple) or (
            len(self.clip_values) != 2
            or not isinstance(self.clip_values[0], (int, float))
            or not isinstance(self.clip_values[1], (int, float))
            or self.clip_values[0] > self.clip_values[1]
        ):
            raise ValueError("The argument `clip_Values` has to be a float or tuple of two float values as (min, max).")

        if not (isinstance(self.delta, (int, float)) or isinstance(self.delta, tuple)) or (
            isinstance(self.delta, tuple)
            and (
                len(self.delta) != 2
                or not isinstance(self.delta[0], (int, float))
                or not isinstance(self.delta[1], (int, float))
                or self.delta[0] > self.delta[1]
            )
        ):
            raise ValueError("The argument `delta` has to be a float or tuple of two float values as (min, max).")
