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

import logging
from io import BytesIO

import numpy as np

from art.config import ART_NUMPY_DTYPE
from art.defences.preprocessor.preprocessor import Preprocessor

logger = logging.getLogger(__name__)


class JpegCompression(Preprocessor):
    """
    Implement the JPEG compression defence approach.

    | Paper link: https://arxiv.org/abs/1705.02900, https://arxiv.org/abs/1608.00853


    | Please keep in mind the limitations of defences. For more information on the limitations of this defence,
        see https://arxiv.org/abs/1802.00420 . For details on how to evaluate classifier security in general, see
        https://arxiv.org/abs/1902.06705
    """

    params = ["quality", "channel_index", "clip_values"]

    def __init__(self, clip_values, quality=50, channel_index=3, apply_fit=True, apply_predict=False):
        """
        Create an instance of JPEG compression.

        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :type clip_values: `tuple`
        :param quality: The image quality, on a scale from 1 (worst) to 95 (best). Values above 95 should be avoided.
        :type quality: `int`
        :param channel_index: Index of the axis in data containing the color channels or features.
        :type channel_index: `int`
        :param apply_fit: True if applied during fitting/training.
        :type apply_fit: `bool`
        :param apply_predict: True if applied during predicting.
        :type apply_predict: `bool`
        """
        super(JpegCompression, self).__init__()
        self._is_fitted = True
        self._apply_fit = apply_fit
        self._apply_predict = apply_predict
        self.set_params(quality=quality, channel_index=channel_index, clip_values=clip_values)

    @property
    def apply_fit(self):
        return self._apply_fit

    @property
    def apply_predict(self):
        return self._apply_predict

    def _compress(self, x, mode):
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

    def __call__(self, x, y=None):
        """
        Apply JPEG compression to sample `x`.

        :param x: Sample to compress with shape of `NCHW`, `NHWC`, `NCFHW` or `NFHWC`. `x` values are
        expected to be in the data range [0, 1] or [0, 255].
        :type x: `np.ndarray`
        :param y: Labels of the sample `x`. This function does not affect them in any way.
        :type y: `np.ndarray`
        :return: compressed sample.
        :rtype: `np.ndarray`
        """

        x_ndim = x.ndim
        if x_ndim == 2:
            raise ValueError(
                "Feature vectors detected. JPEG compression can only be applied to data with spatial dimensions."
            )

        if x_ndim > 5:
            raise ValueError(
                "Unrecognized input dimension. JPEG compression can only be applied to image and video data."
            )

        if x.min() < 0.0:
            raise ValueError(
                "Negative values in input `x` detected. The JPEG compression defence requires unnormalized input."
            )

        # Swap channel index
        if self.channel_index == 1 and x_ndim == 4:
            # image shape NCHW to NHWC
            x = np.transpose(x, (0, 2, 3, 1))
        elif self.channel_index == 1 and x_ndim == 5:
            # video shape NCFHW to NFHWC
            x = np.transpose(x, (0, 2, 3, 4, 1))

        # insert temporal dimension to image data
        if x_ndim == 4:
            x = np.expand_dims(x, axis=1)

        # Convert into uint8
        if self.clip_values[1] == 1.0:
            x = x * 255
        x = x.astype("uint8")

        # Set image mode
        if x.shape[-1] == 1:
            image_mode = "L"
        elif x.shape[-1] == 3:
            image_mode = "RGB"
        else:
            raise NotImplementedError("Currently only support `RGB` and `L` images.")

        # Prepare grayscale images for "L" mode
        if image_mode == "L":
            x = np.squeeze(x, axis=-1)

        # Compress one image at a time
        x_jpeg = x.copy()
        for idx in np.ndindex(x.shape[:2]):
            x_jpeg[idx] = self._compress(x[idx], image_mode)

        # Undo preparation grayscale images for "L" mode
        if image_mode == "L":
            x_jpeg = np.expand_dims(x_jpeg, axis=-1)

        # Convert to ART dtype
        if self.clip_values[1] == 1.0:
            x_jpeg = x_jpeg / 255.0
        x_jpeg = x_jpeg.astype(ART_NUMPY_DTYPE)

        # remove temporal dimension for image data
        if x_ndim == 4:
            x_jpeg = np.squeeze(x_jpeg, axis=1)

        # Swap channel index
        if self.channel_index == 1 and x_jpeg.ndim == 4:
            # image shape NHWC to NCHW
            x_jpeg = np.transpose(x_jpeg, (0, 3, 1, 2))
        elif self.channel_index == 1 and x_ndim == 5:
            # video shape NFHWC to NCFHW
            x_jpeg = np.transpose(x_jpeg, (0, 4, 1, 2, 3))
        return x_jpeg, y

    def estimate_gradient(self, x, grad):
        return grad

    def fit(self, x, y=None, **kwargs):
        """
        No parameters to learn for this method; do nothing.
        """
        pass

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and applies defence-specific checks before saving them as attributes.

        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :type clip_values: `tuple`
        :param quality: The image quality, on a scale from 1 (worst) to 95 (best). Values above 95 should be avoided.
        :type quality: `int`
        :param channel_index: Index of the axis in data containing the color channels or features.
        :type channel_index: `int`
        """
        # Save defense-specific parameters
        super(JpegCompression, self).set_params(**kwargs)

        if not isinstance(self.quality, (int, np.int)) or self.quality <= 0 or self.quality > 100:
            raise ValueError("Image quality must be a positive integer <= 100.")

        if not (isinstance(self.channel_index, (int, np.int)) and self.channel_index in [1, 3, 4]):
            raise ValueError(
                "Data channel must be an integer equal to 1, 3 or 4. The batch dimension is not a valid channel."
            )

        if len(self.clip_values) != 2:
            raise ValueError("'clip_values' should be a tuple of 2 floats or arrays containing the allowed data range.")

        if np.array(self.clip_values[0] >= self.clip_values[1]).any():
            raise ValueError("Invalid 'clip_values': min >= max.")

        if self.clip_values[0] != 0:
            raise ValueError("'clip_values' min value must be 0.")

        if self.clip_values[1] != 1.0 and self.clip_values[1] != 255:
            raise ValueError("'clip_values' max value must be either 1 or 255.")

        return True
