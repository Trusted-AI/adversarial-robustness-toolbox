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
This module implements the local spatial smoothing defence in `SpatialSmoothing` in PyTorch.

| Paper link: https://arxiv.org/abs/1704.01155

| Please keep in mind the limitations of defences. For more information on the limitations of this defence,
    see https://arxiv.org/abs/1803.09868 . For details on how to evaluate classifier security in general, see
    https://arxiv.org/abs/1902.06705
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np

from art.defences.preprocessor.preprocessor import PreprocessorPyTorch

if TYPE_CHECKING:
    # pylint: disable=C0412
    import torch
    from art.utils import CLIP_VALUES_TYPE

logger = logging.getLogger(__name__)


class SpatialSmoothingPyTorch(PreprocessorPyTorch):
    """
    Implement the local spatial smoothing defence approach in PyTorch.

    | Paper link: https://arxiv.org/abs/1704.01155

    | Please keep in mind the limitations of defences. For more information on the limitations of this defence,
        see https://arxiv.org/abs/1803.09868 . For details on how to evaluate classifier security in general, see
        https://arxiv.org/abs/1902.06705
    """

    def __init__(
        self,
        window_size: int = 3,
        channels_first: bool = False,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        apply_fit: bool = False,
        apply_predict: bool = True,
        device_type: str = "gpu",
    ) -> None:
        """
        Create an instance of local spatial smoothing.

        :param window_size: Size of spatial smoothing window.
        :param channels_first: Set channels first or last.
        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :param apply_fit: True if applied during fitting/training.
        :param apply_predict: True if applied during predicting.
        :param device_type: Type of device on which the classifier is run, either `gpu` or `cpu`.
        """
        import torch  # lgtm [py/repeated-import]

        super().__init__(apply_fit=apply_fit, apply_predict=apply_predict)

        self.channels_first = channels_first
        self.window_size = window_size
        self.clip_values = clip_values
        self._check_params()

        # Set device
        if device_type == "cpu" or not torch.cuda.is_available():
            self._device = torch.device("cpu")
        else:
            cuda_idx = torch.cuda.current_device()
            self._device = torch.device("cuda:{}".format(cuda_idx))

        from kornia.filters import MedianBlur

        class MedianBlurCustom(MedianBlur):
            """
            An ongoing effort to reproduce the median blur function in SciPy.
            """

            def __init__(self, kernel_size: Tuple[int, int]) -> None:
                super().__init__(kernel_size)

                # Half-pad the input so that the output keeps the same shape.
                # * center pixel located lower right
                half_pad = [int(k % 2 == 0) for k in kernel_size]
                if hasattr(self, "padding"):
                    # kornia < 0.5.0
                    padding = self.padding
                else:
                    # kornia >= 0.5.0
                    from kornia.filters.median import _compute_zero_padding

                    padding = _compute_zero_padding(kernel_size)  # type: ignore
                self.p2d = [
                    int(padding[-1]) + half_pad[-1],  # type: ignore
                    int(padding[-1]),  # type: ignore
                    int(padding[-2]) + half_pad[-2],  # type: ignore
                    int(padding[-2]),  # type: ignore
                ]
                # PyTorch requires padding size should be less than the corresponding input dimension.

                if not hasattr(self, "kernel"):
                    # kornia >= 0.5.0
                    from kornia.filters.kernels import get_binary_kernel2d

                    self.kernel = get_binary_kernel2d(kernel_size)

            # pylint: disable=W0622
            def forward(self, input: "torch.Tensor"):  # type: ignore
                import torch  # lgtm [py/repeated-import]
                import torch.nn.functional as F

                if not torch.is_tensor(input):
                    raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(input)))
                if not len(input.shape) == 4:
                    raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}".format(input.shape))
                # prepare kernel
                batch_size, channels, height, width = input.shape
                kernel: torch.Tensor = self.kernel.to(input.device).to(input.dtype)
                # map the local window to single vector

                _input = input.reshape(batch_size * channels, 1, height, width)
                if input.dtype == torch.int64:
                    # "reflection_pad2d" not implemented for 'Long'
                    # "reflect" in scipy.ndimage.median_filter has no equivalence in F.pad.
                    # "reflect" in PyTorch maps to "mirror" in scipy.ndimage.median_filter.
                    _input = _input.to(torch.float32)
                    _input = F.pad(_input, self.p2d, "reflect")
                    _input = _input.to(torch.int64)
                else:
                    _input = F.pad(_input, self.p2d, "reflect")

                features: torch.Tensor = F.conv2d(_input, kernel, stride=1)
                features = features.view(batch_size, channels, -1, height, width)  # BxCx(K_h * K_w)xHxW

                # compute the median along the feature axis
                # * torch.median(), if window size even, use smaller value (e.g. median(4,5)=4)
                median: torch.Tensor = torch.median(features, dim=2)[0]
                return median

        self.median_blur = MedianBlurCustom(kernel_size=(self.window_size, self.window_size))

    def forward(
        self, x: "torch.Tensor", y: Optional["torch.Tensor"] = None
    ) -> Tuple["torch.Tensor", Optional["torch.Tensor"]]:
        """
        Apply local spatial smoothing to sample `x`.
        """
        x_ndim = x.ndim

        # NHWC/NCFHW/NFHWC --> NCHW.
        if x_ndim == 4:
            if self.channels_first:
                x_nchw = x
            else:
                # NHWC --> NCHW
                x_nchw = x.permute(0, 3, 1, 2)
        elif x_ndim == 5:
            if self.channels_first:
                # NCFHW --> NFCHW --> NCHW
                nb_clips, channels, clip_size, height, width = x.shape
                x_nchw = x.permute(0, 2, 1, 3, 4).reshape(nb_clips * clip_size, channels, height, width)
            else:
                # NFHWC --> NHWC --> NCHW
                nb_clips, clip_size, height, width, channels = x.shape
                x_nchw = x.reshape(nb_clips * clip_size, height, width, channels).permute(0, 3, 1, 2)
        else:
            raise ValueError(
                "Unrecognized input dimension. Spatial smoothing can only be applied to image and video data."
            )

        x_nchw = self.median_blur(x_nchw)

        # NHWC/NCFHW/NFHWC <-- NCHW.
        if x_ndim == 4:
            if self.channels_first:
                x = x_nchw
            else:
                #   NHWC <-- NCHW
                x = x_nchw.permute(0, 2, 3, 1)
        elif x_ndim == 5:  # lgtm [py/redundant-comparison]
            if self.channels_first:
                # NCFHW <-- NFCHW <-- NCHW
                x_nfchw = x_nchw.reshape(nb_clips, clip_size, channels, height, width)
                x = x_nfchw.permute(0, 2, 1, 3, 4)
            else:
                # NFHWC <-- NHWC <-- NCHW
                x_nhwc = x_nchw.permute(0, 2, 3, 1)
                x = x_nhwc.reshape(nb_clips, clip_size, height, width, channels)

        if self.clip_values is not None:
            x = x.clamp(min=self.clip_values[0], max=self.clip_values[1])

        return x, y

    def _check_params(self) -> None:
        if not (isinstance(self.window_size, (int, np.int)) and self.window_size > 0):
            raise ValueError("Sliding window size must be a positive integer.")

        if self.clip_values is not None and len(self.clip_values) != 2:
            raise ValueError("'clip_values' should be a tuple of 2 floats or arrays containing the allowed data range.")

        if self.clip_values is not None and np.array(self.clip_values[0] >= self.clip_values[1]).any():
            raise ValueError("Invalid 'clip_values': min >= max.")
