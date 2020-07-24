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
from kornia.filters.kernels import get_binary_kernel2d
import torch.nn.functional as F
import torch.nn as nn

import logging
from typing import Optional, Tuple

import numpy as np
import torch
from kornia.filters import MedianBlur

from art.defences.preprocessor.spatial_smoothing import SpatialSmoothing
from art.utils import Deprecated, deprecated_keyword_arg

logger = logging.getLogger(__name__)


class MedianBlurCustom(MedianBlur):
    """
    An ongoing effort to reproduce the median blur function in SciPy.
    """

    def __init__(self, kernel_size: Tuple[int, int]) -> None:
        super().__init__(kernel_size)

        # Half-pad the input so that the output keeps the same shape.
        # * center pixel located lower right
        half_pad = [k % 2 == 0 for k in kernel_size]
        self.p2d = (self.padding[-1] + half_pad[-1], self.padding[-1],
                    self.padding[-2] + half_pad[-2], self.padding[-2])
        # PyTorch requires Padding size should be less than the corresponding input dimension,

    def forward(self, input: torch.Tensor):  # type: ignore
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}".format(input.shape))
        # prepare kernel
        b, c, h, w = input.shape
        kernel: torch.Tensor = self.kernel.to(input.device).to(input.dtype)
        # map the local window to single vector

        _input = input.reshape(b * c, 1, h, w)
        if input.dtype == torch.int64:
            # "reflection_pad2d" not implemented for 'Long'
            # "reflect" in scipy.ndimage.median_filter has no equivalance in F.pad.
            # "reflect" in PyTorch maps to "mirror" in scipy.ndimage.median_filter.
            _input = _input.to(torch.float32)
            _input = F.pad(_input, self.p2d, "reflect")
            _input = _input.to(torch.int64)
        else:
            _input = F.pad(_input, self.p2d, "reflect")

        features: torch.Tensor = F.conv2d(_input, kernel, stride=1)
        features = features.view(b, c, -1, h, w)  # BxCx(K_h * K_w)xHxW

        # compute the median along the feature axis
        # * torch.median(), if window size even, use smaller value (e.g. median(4,5)=4)
        median: torch.Tensor = torch.median(features, dim=2)[0]
        return median


class SpatialSmoothingPyTorch(SpatialSmoothing):
    """
    Implement the local spatial smoothing defence approach in PyTorch.

    | Paper link: https://arxiv.org/abs/1704.01155

    | Please keep in mind the limitations of defences. For more information on the limitations of this defence,
        see https://arxiv.org/abs/1803.09868 . For details on how to evaluate classifier security in general, see
        https://arxiv.org/abs/1902.06705
    """

    def __init__(
        self,
        device_type: str = "gpu",
        **kwargs,
    ) -> None:
        """
        Create an instance of local spatial smoothing.

        :param device_type: Type of device on which the classifier is run, either `gpu` or `cpu`.
        :param **kwargs: Parameters from the parent.
        """
        super().__init__(**kwargs)

        self.median_blur = MedianBlurCustom(kernel_size=(self.window_size, self.window_size))

        # Set device
        if device_type == "cpu" or not torch.cuda.is_available():
            self._device = torch.device("cpu")
        else:
            cuda_idx = torch.cuda.current_device()
            self._device = torch.device("cuda:{}".format(cuda_idx))

        if self.clip_values is not None:
            self.clip_values = torch.tensor(self.clip_values, device=self._device)

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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
                nb_clips, c, clip_size, h, w = x.shape
                x_nchw = x.permute(0, 2, 1, 3, 4).reshape(nb_clips * clip_size, c, h, w)
            else:
                # NFHWC --> NHWC --> NCHW
                nb_clips, clip_size, h, w, c = x.shape
                x_nchw = x.reshape(nb_clips * clip_size, h, w, c).permute(0, 3, 1, 2)
        else:
            raise ValueError(
                "Unrecognized input dimension. Spatial smoothing can only be applied to image and video data.")

        x_nchw = self.median_blur(x_nchw)

        # NHWC/NCFHW/NFHWC <-- NCHW.
        if x_ndim == 4:
            if self.channels_first:
                x = x_nchw
            else:
                #   NHWC <-- NCHW
                x = x_nchw.permute(0, 2, 3, 1)
        elif x_ndim == 5:
            if self.channels_first:
                # NCFHW <-- NFCHW <-- NCHW
                x_nfchw = x_nchw.reshape(nb_clips, clip_size, c, h, w)
                x = x_nfchw.permute(0, 2, 1, 3, 4)
            else:
                # NFHWC <-- NHWC <-- NCHW
                x_nhwc = x_nchw.permute(0, 2, 3, 1)
                x = x_nhwc.reshape(nb_clips, clip_size, h, w, c)

        if self.clip_values is not None:
            x = x.clamp(min=self.clip_values[0], max=self.clip_values[1])

        return x, y

    def estimate_forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        No need to estimate, since the forward pass is differentiable.
        """
        return self.forward(x, y)[0]

    def __call__(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply local spatial smoothing to sample `x`.

        :param x: Sample to smooth with shape `(batch_size, width, height, depth)`.
        :param y: Labels of the sample `x`. This function does not affect them in any way.
        :return: Smoothed sample.
        """
        x = torch.tensor(x, device=self._device)
        if y is not None:
            y = torch.tensor(y, device=self._device)

        with torch.no_grad():
            x, y = self.forward(x, y)

        result = x.cpu().numpy()
        if y is not None:
            y = y.cpu().numpy()
        return result, y

    # Backward compatibility.
    def estimate_gradient(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        x = torch.tensor(x, device=self._device, requires_grad=True)
        grad = torch.tensor(grad, device=self._device)

        x_prime = self.estimate_forward(x)
        x_prime.backward(grad)
        x_grad = x.grad.detach().cpu().numpy()
        if x_grad.shape != x.shape:
            raise ValueError("The input shape is {} while the gradient shape is {}".format(x.shape, x_grad.shape))
        return x_grad
