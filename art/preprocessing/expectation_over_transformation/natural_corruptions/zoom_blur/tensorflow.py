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
This module implements EoT of zoom blur with uniformly sampled zoom factor.
"""
import logging
from typing import Tuple, Union, TYPE_CHECKING, Optional

import numpy as np

from art.preprocessing.expectation_over_transformation.tensorflow import EoTTensorFlowV2

if TYPE_CHECKING:
    import tensorflow as tf

logger = logging.getLogger(__name__)


class EoTZoomBlurTensorFlow(EoTTensorFlowV2):
    """
    This module implements EoT of zoom blur with uniformly sampled zoom factor.
    """

    def __init__(
        self,
        nb_samples: int,
        clip_values: Tuple[float, float],
        zoom: Union[float, Tuple[float, float]],
        apply_fit: bool = False,
        apply_predict: bool = True,
    ) -> None:
        """
        Create an instance of EoTZoomBlurTensorFlow.

        :param nb_samples: Number of random samples per input sample.
        :param clip_values: Tuple of float representing minimum and maximum values of input `(min, max)`.
        :param zoom: Range to sample the zoom factor. A single float is translated to range [1.0, zoom] or a tuple of
                     floats is used to create sampling range [zoom[0], zoom[1]]. The applied zoom is sampled uniformly
                     from this range for each image.
        :param apply_fit: True if applied during fitting/training.
        :param apply_predict: True if applied during predicting.
        """
        super().__init__(
            apply_fit=apply_fit, apply_predict=apply_predict, nb_samples=nb_samples, clip_values=clip_values
        )

        self.zoom = zoom
        self.zoom_range = (1.0, zoom) if isinstance(zoom, (int, float)) else zoom
        self._check_params()

    def _transform(
        self, x: "tf.Tensor", y: Optional["tf.Tensor"], **kwargs
    ) -> Tuple["tf.Tensor", Optional["tf.Tensor"]]:
        """
        Transformation of an image with randomly sampled zoom blur.

        :param x: Input samples.
        :param y: Label of the samples `x`.
        :return: Transformed samples and labels.
        """
        import tensorflow as tf  # lgtm [py/repeated-import]

        nb_zooms = 10
        x_blur = tf.zeros_like(x)
        max_zoom_i = np.random.uniform(low=self.zoom_range[0], high=self.zoom_range[1])
        zooms = np.arange(start=1.0, stop=max_zoom_i, step=(max_zoom_i - 1.0) / nb_zooms)

        height = x.shape[0]
        width = x.shape[1]

        for zoom in zooms:
            size = [int(a * zoom) for a in x.shape[0:2]]
            x_resized = tf.image.resize(
                images=x,
                size=size,
                method=tf.image.ResizeMethod.BILINEAR,
                preserve_aspect_ratio=True,
                antialias=False,
                name=None,
            )

            trim_top = (x_resized.shape[0] - height) // 2
            trim_left = (x_resized.shape[0] - width) // 2

            x_blur += x_resized[trim_top : trim_top + height, trim_left : trim_left + width, :]

        x_out = (x + x_blur) / (nb_zooms + 1)
        return tf.clip_by_value(x_out, clip_value_min=self.clip_values[0], clip_value_max=self.clip_values[1]), y

    def _check_params(self) -> None:

        # pylint: disable=R0916
        if not isinstance(self.zoom, (int, float, tuple)) or (
            isinstance(self.zoom, tuple)
            and (
                len(self.zoom) != 2
                or not isinstance(self.zoom[0], (int, float))
                or not isinstance(self.zoom[1], (int, float))
                or self.zoom[0] > self.zoom[1]
                or self.zoom[0] < 1.0
            )
        ):
            raise ValueError("The argument `lam` has to be a float or tuple of two float values as (min, max).")
