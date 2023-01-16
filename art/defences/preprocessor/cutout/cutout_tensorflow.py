# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2022
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
This module implements the Cutout data augmentation defence in TensorFlow.

| Paper link: https://arxiv.org/abs/1708.04552

| Please keep in mind the limitations of defences. For more information on the limitations of this defence,
    see https://arxiv.org/abs/1803.09868 . For details on how to evaluate classifier security in general, see
    https://arxiv.org/abs/1902.06705
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, Tuple, TYPE_CHECKING

from art.defences.preprocessor.preprocessor import PreprocessorTensorFlowV2

if TYPE_CHECKING:
    # pylint: disable=C0412
    import tensorflow as tf

logger = logging.getLogger(__name__)


class CutoutTensorFlowV2(PreprocessorTensorFlowV2):
    """
    Implement the Cutout data augmentation defence approach in TensorFlow v2.

    | Paper link: https://arxiv.org/abs/1708.04552

    | Please keep in mind the limitations of defences. For more information on the limitations of this defence,
        see https://arxiv.org/abs/1803.09868 . For details on how to evaluate classifier security in general, see
        https://arxiv.org/abs/1902.06705
    """

    params = ["length", "channels_first", "verbose"]

    def __init__(
        self,
        length: int,
        channels_first: bool = False,
        apply_fit: bool = True,
        apply_predict: bool = False,
        verbose: bool = False,
    ):
        """
        Create an instance of a Cutout data augmentation object.

        :param length: Maximum length of the bounding box.
        :param channels_first: Set channels first or last.
        :param apply_fit: True if applied during fitting/training.
        :param apply_predict: True if applied during predicting.
        :param verbose: Show progress bars.
        """
        super().__init__(is_fitted=True, apply_fit=apply_fit, apply_predict=apply_predict)
        self.length = length
        self.channels_first = channels_first
        self.verbose = verbose
        self._check_params()

    def forward(self, x: "tf.Tensor", y: Optional["tf.Tensor"] = None) -> Tuple["tf.Tensor", Optional["tf.Tensor"]]:
        """
        Apply Cutout data augmentation to sample `x`.

        :param x: Sample to cut out with shape of `NCHW`, `NHWC`, `NCFHW` or `NFHWC`.
                  `x` values are expected to be in the data range [0, 1] or [0, 255].
        :param y: Labels of the sample `x`. This function does not affect them in any way.
        :return: Data augmented sample.
        """
        import tensorflow as tf  # lgtm [py/repeated-import]
        import tensorflow_addons as tfa

        x_ndim = len(x.shape)

        # NCHW/NCFHW/NFHWC --> NHWC
        if x_ndim == 4:
            if self.channels_first:
                # NCHW --> NHWC
                x_nhwc = tf.transpose(x, (0, 2, 3, 1))
            else:
                # NHWC
                x_nhwc = x
        elif x_ndim == 5:
            if self.channels_first:
                # NCFHW --> NFHWC --> NHWC
                nb_clips, channels, clip_size, height, width = x.shape
                x_nfhwc = tf.transpose(x, (0, 2, 3, 4, 1))
                x_nhwc = tf.reshape(x_nfhwc, (nb_clips * clip_size, height, width, channels))
            else:
                # NFHWC --> NHWC
                nb_clips, clip_size, height, width, channels = x.shape
                x_nhwc = tf.reshape(x, (nb_clips * clip_size, height, width, channels))
        else:
            raise ValueError("Unrecognized input dimension. Cutout can only be applied to image and video data.")

        # round down length to be divisible by 2
        length = self.length if self.length % 2 == 0 else max(self.length - 1, 2)

        # apply random cutout
        x_nhwc = tfa.image.random_cutout(x_nhwc, (length, length))

        # NCHW/NCFHW/NFHWC <-- NHWC
        if x_ndim == 4:
            if self.channels_first:
                # NHWC <-- NCHW
                x_aug = tf.transpose(x_nhwc, (0, 3, 1, 2))
            else:
                # NHWC
                x_aug = x_nhwc
        elif x_ndim == 5:  # lgtm [py/redundant-comparison]
            if self.channels_first:
                # NCFHW <-- NFHWC <-- NHWC
                x_nfhwc = tf.reshape(x_nhwc, (nb_clips, clip_size, height, width, channels))
                x_aug = tf.transpose(x_nfhwc, (0, 4, 1, 2, 3))
            else:
                # NFHWC <-- NHWC
                x_aug = tf.reshape(x_nhwc, (nb_clips, clip_size, height, width, channels))

        return x_aug, y

    def _check_params(self) -> None:
        if self.length <= 0:
            raise ValueError("Bounding box length must be positive.")
