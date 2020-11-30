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
This module implements
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np

from art.preprocessing.preprocessing import PreprocessorTensorFlowV2

if TYPE_CHECKING:
    import tensorflow as tf
    from art.utils import CLIP_VALUES_TYPE

logger = logging.getLogger(__name__)


class EOTImageRotationTensorFlowV2(PreprocessorTensorFlowV2):
    """
    This module implements Expectation over Transformation preprocessing.
    """

    params = ["nb_samples", "angles_range", "clip_values", "label_type"]

    label_types = ["classification"]

    def __init__(
        self,
        nb_samples: int = 1,
        angles_range: float = 3.14,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        label_type: str = "classification",
        apply_fit: bool = False,
        apply_predict: bool = True,
    ) -> None:
        """
        Create an instance of EOTImageRotationTensorFlowV2.

        :param nb_samples: Number of random samples per input sample.
        :param angles_range: A positive scalar angle in radians defining the uniform sampling range from negative and
                             positive angles_range.
        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
                            for features.
        :param label_type: String defining the type of labels. Currently supported: `classification`
        :param apply_fit: True if applied during fitting/training.
        :param apply_predict: True if applied during predicting.
        """
        super().__init__(is_fitted=True, apply_fit=apply_fit, apply_predict=apply_predict)

        self.nb_samples = nb_samples
        self.angles_range = angles_range
        self.clip_Values = clip_values
        self.label_type = label_type
        self._check_params()

    def forward(self, x: "tf.Tensor", y: Optional["tf.Tensor"] = None) -> Tuple["tf.Tensor", Optional["tf.Tensor"]]:
        """
        Apply audio filter to a single sample `x`.

        :param x: A single audio sample.
        :param y: Label of the sample `x`. This function does not affect them in any way.
        :return: Similar sample.
        """
        import tensorflow as tf  # lgtm [py/repeated-import]
        import tensorflow_addons as tfa

        x_preprocess_list = list()
        y_preprocess_list = list()

        for i_image in range(x.shape[0]):
            for i_sample in range(self.nb_samples):
                angles = tf.random.uniform(shape=(), minval=-self.angles_range, maxval=self.angles_range)
                images = x[i_image]
                x_preprocess_i = tfa.image.rotate(images=images, angles=angles, interpolation="NEAREST", name=None)
                if self.clip_Values is not None:
                    x_preprocess_i = tf.clip_by_value(
                        t=x_preprocess_i, clip_value_min=-self.angles_range, clip_value_max=self.angles_range, name=None
                    )
                x_preprocess_list.append(x_preprocess_i)
                if y is not None:
                    y_preprocess_list.append(y[i_image])

        x_preprocess = tf.stack(x_preprocess_list, axis=0, name=None)
        if y is None:
            y_preprocess = y
        else:
            y_preprocess = tf.stack(y_preprocess_list, axis=0, name=None)

        return x_preprocess, y_preprocess

    def _check_params(self) -> None:

        if not isinstance(self.nb_samples, int) or self.nb_samples < 1:
            raise ValueError("The number of samples needs to be an integer greater than or equal to 1.")

        if not isinstance(self.angles_range, float) or np.pi / 2 < self.angles_range or self.angles_range <= 0.0:
            raise ValueError("The range of angles must be a float in the range (0.0, Pi/2].")

        if self.label_type not in self.label_types:
            raise ValueError(
                "The input for label_type needs to be one of {}, currently receiving `{}`.".format(
                    self.label_types, self.label_type
                )
            )
