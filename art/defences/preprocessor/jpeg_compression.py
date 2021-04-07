# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
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
This module implements the JPEG compression defence `JpegCompression`.

| Paper link: https://arxiv.org/abs/1705.02900, https://arxiv.org/abs/1608.00853

| Please keep in mind the limitations of defences. For more information on the limitations of this defence, see
    https://arxiv.org/abs/1802.00420 . For details on how to evaluate classifier security in general, see
    https://arxiv.org/abs/1902.06705
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from io import BytesIO
import logging
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np
from tqdm.auto import tqdm

from art.config import ART_NUMPY_DTYPE
from art.defences.preprocessor.preprocessor import Preprocessor

if TYPE_CHECKING:
    from art.utils import CLIP_VALUES_TYPE

logger = logging.getLogger(__name__)


class JpegCompression(Preprocessor):
    """
    Implement the JPEG compression defence approach.

    For input images or videos with 3 color channels the compression is applied in mode `RGB`
    (3x8-bit pixels, true color), for all other numbers of channels the compression is applied for each channel with
    mode `L` (8-bit pixels, black and white).

    | Paper link: https://arxiv.org/abs/1705.02900, https://arxiv.org/abs/1608.00853


    | Please keep in mind the limitations of defences. For more information on the limitations of this defence,
        see https://arxiv.org/abs/1802.00420 . For details on how to evaluate classifier security in general, see
        https://arxiv.org/abs/1902.06705
    """

    params = ["quality", "channels_first", "clip_values", "verbose"]

    def __init__(
        self,
        clip_values: "CLIP_VALUES_TYPE",
        quality: int = 50,
        channels_first: bool = False,
        apply_fit: bool = True,
        apply_predict: bool = True,
        verbose: bool = False,
    ):
        """
        Create an instance of JPEG compression.

        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :param quality: The image quality, on a scale from 1 (worst) to 95 (best). Values above 95 should be avoided.
        :param channels_first: Set channels first or last.
        :param apply_fit: True if applied during fitting/training.
        :param apply_predict: True if applied during predicting.
        :param verbose: Show progress bars.
        """

        super().__init__(is_fitted=True, apply_fit=apply_fit, apply_predict=apply_predict)
        self.quality = quality
        self.channels_first = channels_first
        self.clip_values = clip_values
        self.verbose = verbose
        self._check_params()

    def _compress(self, x: np.ndarray, mode: str) -> np.ndarray:
        """
        Apply JPEG compression to image input.
        """
        from PIL import Image

        tmp_jpeg = BytesIO()
        x_image = Image.fromarray(x, mode=mode)
        x_image.save(tmp_jpeg, format="jpeg", quality=self.quality)
        x_jpeg = np.array(Image.open(tmp_jpeg))
        tmp_jpeg.close()
        return x_jpeg

    def __call__(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply JPEG compression to sample `x`.

        For input images or videos with 3 color channels the compression is applied in mode `RGB`
        (3x8-bit pixels, true color), for all other numbers of channels the compression is applied for each channel with
        mode `L` (8-bit pixels, black and white).

        :param x: Sample to compress with shape of `NCHW`, `NHWC`, `NCFHW` or `NFHWC`. `x` values are expected to be in
                  the data range [0, 1] or [0, 255].
        :param y: Labels of the sample `x`. This function does not affect them in any way.
        :return: compressed sample.
        """
        x_ndim = x.ndim
        if x_ndim not in [4, 5]:
            raise ValueError(
                "Unrecognized input dimension. JPEG compression can only be applied to image and video data."
            )

        if x.min() < 0.0:
            raise ValueError(
                "Negative values in input `x` detected. The JPEG compression defence requires unnormalized input."
            )

        # Swap channel index
        if self.channels_first and x_ndim == 4:
            # image shape NCHW to NHWC
            x = np.transpose(x, (0, 2, 3, 1))
        elif self.channels_first and x_ndim == 5:
            # video shape NCFHW to NFHWC
            x = np.transpose(x, (0, 2, 3, 4, 1))

        # insert temporal dimension to image data
        if x_ndim == 4:
            x = np.expand_dims(x, axis=1)

        # Convert into uint8
        if self.clip_values[1] == 1.0:
            x = x * 255
        x = x.astype("uint8")

        # Compress one image at a time
        x_jpeg = x.copy()
        for idx in tqdm(np.ndindex(x.shape[:2]), desc="JPEG compression", disable=not self.verbose):
            if x.shape[-1] == 3:
                x_jpeg[idx] = self._compress(x[idx], mode="RGB")
            else:
                for i_channel in range(x.shape[-1]):
                    x_channel = x[idx[0], idx[1], ..., i_channel]
                    x_channel = self._compress(x_channel, mode="L")
                    x_jpeg[idx[0], idx[1], :, :, i_channel] = x_channel

        # Convert to ART dtype
        if self.clip_values[1] == 1.0:
            x_jpeg = x_jpeg / 255.0
        x_jpeg = x_jpeg.astype(ART_NUMPY_DTYPE)

        # remove temporal dimension for image data
        if x_ndim == 4:
            x_jpeg = np.squeeze(x_jpeg, axis=1)

        # Swap channel index
        if self.channels_first and x_jpeg.ndim == 4:
            # image shape NHWC to NCHW
            x_jpeg = np.transpose(x_jpeg, (0, 3, 1, 2))
        elif self.channels_first and x_ndim == 5:
            # video shape NFHWC to NCFHW
            x_jpeg = np.transpose(x_jpeg, (0, 4, 1, 2, 3))
        return x_jpeg, y

    def _check_params(self) -> None:
        if not isinstance(self.quality, (int, np.int)) or self.quality <= 0 or self.quality > 100:
            raise ValueError("Image quality must be a positive integer <= 100.")

        if len(self.clip_values) != 2:
            raise ValueError("'clip_values' should be a tuple of 2 floats or arrays containing the allowed data range.")

        if np.array(self.clip_values[0] >= self.clip_values[1]).any():
            raise ValueError("Invalid 'clip_values': min >= max.")

        if self.clip_values[0] != 0:
            raise ValueError("'clip_values' min value must be 0.")

        if self.clip_values[1] != 1.0 and self.clip_values[1] != 255:
            raise ValueError("'clip_values' max value must be either 1 or 255.")

        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be of type bool.")
