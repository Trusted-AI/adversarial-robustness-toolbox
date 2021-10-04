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
This module implements a wrapper for video compression defence with FFmpeg.

| Please keep in mind the limitations of defences. For details on how to evaluate classifier security in general,
    see https://arxiv.org/abs/1902.06705.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np

from art.defences.preprocessor.preprocessor import PreprocessorPyTorch
from art.defences.preprocessor.video_compression import VideoCompression

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    # pylint: disable=C0412
    import torch


class VideoCompressionPyTorch(PreprocessorPyTorch):
    """
    Implement FFmpeg wrapper for video compression defence based on H.264/MPEG-4 AVC.

    Video compression uses H.264 video encoding. The video quality is controlled with the constant rate factor
    parameter. More information on the constant rate factor: https://trac.ffmpeg.org/wiki/Encode/H.264.
    """

    params = ["video_format", "constant_rate_factor", "channels_first", "verbose"]

    def __init__(
        self,
        *,
        video_format: str,
        constant_rate_factor: int = 28,
        channels_first: bool = False,
        apply_fit: bool = False,
        apply_predict: bool = True,
        device_type: str = "gpu",
        verbose: bool = False,
    ):
        """
        Create an instance of VideoCompression.

        :param video_format: Specify one of supported video file extensions, e.g. `avi`, `mp4` or `mkv`.
        :param constant_rate_factor: Specify constant rate factor (range 0 to 51, where 0 is lossless).
        :param channels_first: Set channels first or last.
        :param apply_fit: True if applied during fitting/training.
        :param apply_predict: True if applied during predicting.
        :param device_type: Type of device on which the classifier is run, either `gpu` or `cpu`.
        :param verbose: Show progress bars.
        """
        import torch  # lgtm [py/repeated-import]
        from torch.autograd import Function

        super().__init__(is_fitted=True, apply_fit=apply_fit, apply_predict=apply_predict)
        self.video_format = video_format
        self.constant_rate_factor = constant_rate_factor
        self.channels_first = channels_first
        self.verbose = verbose
        self._check_params()

        # Set device
        if device_type == "cpu" or not torch.cuda.is_available():
            self._device = torch.device("cpu")
        else:  # pragma: no cover
            cuda_idx = torch.cuda.current_device()
            self._device = torch.device("cuda:{}".format(cuda_idx))

        self.compression_numpy = VideoCompression(
            video_format=video_format,
            constant_rate_factor=constant_rate_factor,
            channels_first=channels_first,
            apply_fit=apply_fit,
            apply_predict=apply_predict,
            verbose=verbose,
        )

        class CompressionPyTorchNumpy(Function):
            """
            Function running Preprocessor.
            """

            @staticmethod
            def forward(ctx, input):  # pylint: disable=W0622,W0221
                numpy_input = input.detach().cpu().numpy()
                result, _ = self.compression_numpy(numpy_input)
                return input.new(result)

            @staticmethod
            def backward(ctx, grad_output):  # pylint: disable=W0221
                numpy_go = grad_output.cpu().numpy()
                result = self.compression_numpy.estimate_gradient(None, numpy_go)
                return grad_output.new(result)

        self._compression_pytorch_numpy = CompressionPyTorchNumpy

    def forward(
        self, x: "torch.Tensor", y: Optional["torch.Tensor"] = None
    ) -> Tuple["torch.Tensor", Optional["torch.Tensor"]]:
        """
        Apply video compression to sample `x`.

        :param x: Sample to compress of shape NCFHW or NFHWC. `x` values are expected to be in the data range [0, 255].
        :param y: Labels of the sample `x`. This function does not affect them in any way.
        :return: Compressed sample.
        """
        x_compressed = self._compression_pytorch_numpy.apply(x)
        return x_compressed, y

    def _check_params(self) -> None:
        if not (isinstance(self.constant_rate_factor, (int, np.int)) and 0 <= self.constant_rate_factor < 52):
            raise ValueError("Constant rate factor must be an integer in the range [0, 51].")

        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be of type bool.")
