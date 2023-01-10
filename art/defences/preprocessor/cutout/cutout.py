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
This module implements the Cutout data augmentation defence.

| Paper link: https://arxiv.org/abs/1708.04552

| Please keep in mind the limitations of defences. For more information on the limitations of this defence,
    see https://arxiv.org/abs/1803.09868 . For details on how to evaluate classifier security in general, see
    https://arxiv.org/abs/1902.06705
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, Tuple

import numpy as np
from tqdm.auto import trange

from art.defences.preprocessor.preprocessor import Preprocessor

logger = logging.getLogger(__name__)


class Cutout(Preprocessor):
    """
    Implement the Cutout data augmentation defence approach.

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
    ) -> None:
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

    def __call__(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply Cutout data augmentation to sample `x`.

        :param x: Sample to cut out with shape of `NCHW`, `NHWC`, `NCFHW` or `NFHWC`.
                  `x` values are expected to be in the data range [0, 1] or [0, 255].
        :param y: Labels of the sample `x`. This function does not affect them in any way.
        :return: Data augmented sample.
        """
        x_ndim = len(x.shape)

        # NCHW/NCFHW/NFHWC --> NHWC
        if x_ndim == 4:
            if self.channels_first:
                # NCHW --> NHWC
                x_nhwc = np.transpose(x, (0, 2, 3, 1))
            else:
                # NHWC
                x_nhwc = x
        elif x_ndim == 5:
            if self.channels_first:
                # NCFHW --> NFHWC --> NHWC
                nb_clips, channels, clip_size, height, width = x.shape
                x_nfhwc = np.transpose(x, (0, 2, 3, 4, 1))
                x_nhwc = np.reshape(x_nfhwc, (nb_clips * clip_size, height, width, channels))
            else:
                # NFHWC --> NHWC
                nb_clips, clip_size, height, width, channels = x.shape
                x_nhwc = np.reshape(x, (nb_clips * clip_size, height, width, channels))
        else:
            raise ValueError("Unrecognized input dimension. Cutout can only be applied to image and video data.")

        n, height, width, _ = x_nhwc.shape
        x_nhwc = x_nhwc.copy()

        # generate a random bounding box per image
        for idx in trange(n, desc="Cutout", disable=not self.verbose):
            # uniform sampling
            center_y = np.random.randint(height)
            center_x = np.random.randint(width)
            bby1 = np.clip(center_y - self.length // 2, 0, height)
            bbx1 = np.clip(center_x - self.length // 2, 0, width)
            bby2 = np.clip(center_y + self.length // 2, 0, height)
            bbx2 = np.clip(center_x + self.length // 2, 0, width)

            # zero out the bounding box
            x_nhwc[idx, bbx1:bbx2, bby1:bby2, :] = 0

        # NCHW/NCFHW/NFHWC <-- NHWC
        if x_ndim == 4:
            if self.channels_first:
                # NHWC <-- NCHW
                x_aug = np.transpose(x_nhwc, (0, 3, 1, 2))
            else:
                # NHWC
                x_aug = x_nhwc
        elif x_ndim == 5:  # lgtm [py/redundant-comparison]
            if self.channels_first:
                # NCFHW <-- NFHWC <-- NHWC
                x_nfhwc = np.reshape(x_nhwc, (nb_clips, clip_size, height, width, channels))
                x_aug = np.transpose(x_nfhwc, (0, 4, 1, 2, 3))
            else:
                # NFHWC <-- NHWC
                x_aug = np.reshape(x_nhwc, (nb_clips, clip_size, height, width, channels))

        return x_aug, y

    def _check_params(self) -> None:
        if self.length <= 0:
            raise ValueError("Bounding box length must be positive.")
