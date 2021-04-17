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
This module implements Expectation over Transformation preprocessing for image rotation in TensorFlow.
"""
import logging
from typing import Optional, TYPE_CHECKING, Tuple, Union

import numpy as np

from art.preprocessing.expectation_over_transformation.tensorflow import EoTTensorFlowV2

if TYPE_CHECKING:
    # pylint: disable=C0412
    import tensorflow as tf
    from art.utils import CLIP_VALUES_TYPE

logger = logging.getLogger(__name__)


class EoTImageRotationTensorFlow(EoTTensorFlowV2):
    """
    This module implements Expectation over Transformation preprocessing for image rotation in TensorFlow.
    """

    params = ["nb_samples", "angles", "clip_values", "label_type"]

    label_types = ["classification"]

    def __init__(
        self,
        nb_samples: int,
        clip_values: "CLIP_VALUES_TYPE",
        angles: Union[float, Tuple[float, float]] = 45.0,
        label_type: str = "classification",
        apply_fit: bool = False,
        apply_predict: bool = True,
    ) -> None:
        """
        Create an instance of EoTImageRotationTensorFlow.

        :param nb_samples: Number of random samples per input sample.
        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
                            for features.
        :param angles: A positive scalar angle in degrees defining the uniform sampling range from negative to
                       positive angles_range.
        :param label_type: String defining the type of labels. Currently supported: `classification`
        :param apply_fit: True if applied during fitting/training.
        :param apply_predict: True if applied during predicting.
        """
        super().__init__(
            apply_fit=apply_fit, apply_predict=apply_predict, nb_samples=nb_samples, clip_values=clip_values
        )

        self.angles = angles
        self.angles_range = (-angles, angles) if isinstance(angles, (int, float)) else angles
        self.label_type = label_type
        self._check_params()

    def _transform(
        self, x: "tf.Tensor", y: Optional["tf.Tensor"], **kwargs
    ) -> Tuple["tf.Tensor", Optional["tf.Tensor"]]:
        """
        Transformation of an input image and its label by randomly sampled rotation.

        :param x: Input samples.
        :param y: Label of the samples `x`.
        :return: Transformed samples and labels.
        """
        import tensorflow as tf  # lgtm [py/repeated-import]
        import tensorflow_addons as tfa

        # pylint: disable=E1120,E1123
        angles = tf.random.uniform(shape=(), minval=self.angles_range[0], maxval=self.angles_range[1])
        angles = angles / 360.0 * 2.0 * np.pi
        x_preprocess = tfa.image.rotate(images=x, angles=angles, interpolation="NEAREST", name=None)
        x_preprocess = tf.clip_by_value(
            t=x_preprocess, clip_value_min=-self.clip_values[0], clip_value_max=self.clip_values[1], name=None
        )
        return x_preprocess, y

    def _check_params(self) -> None:

        # pylint: disable=R0916
        if not isinstance(self.angles, (int, float, tuple)) or (
            isinstance(self.angles, tuple)
            and (
                len(self.angles) != 2
                or not isinstance(self.angles[0], (int, float))
                or not isinstance(self.angles[1], (int, float))
                or self.angles[0] > self.angles[1]
            )
        ):
            raise ValueError("The range of angles must be a float in the range (0.0, 180.0].")

        if self.label_type not in self.label_types:
            raise ValueError(
                "The input for label_type needs to be one of {}, currently receiving `{}`.".format(
                    self.label_types, self.label_type
                )
            )
