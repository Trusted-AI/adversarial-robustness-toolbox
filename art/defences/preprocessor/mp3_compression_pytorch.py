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
This module implements a wrapper for MP3 compression defence.

| Please keep in mind the limitations of defences. For details on how to evaluate classifier security in general,
    see https://arxiv.org/abs/1902.06705.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import TYPE_CHECKING, Optional, Tuple

from art.defences.preprocessor.mp3_compression import Mp3Compression
from art.defences.preprocessor.preprocessor import PreprocessorPyTorch

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    # pylint: disable=C0412
    import torch


class Mp3CompressionPyTorch(PreprocessorPyTorch):
    """
    Implement the MP3 compression defense approach.
    """

    params = ["channels_first", "sample_rate", "verbose"]

    def __init__(
        self,
        sample_rate: int,
        channels_first: bool = False,
        apply_fit: bool = False,
        apply_predict: bool = True,
        device_type: str = "gpu",
        verbose: bool = False,
    ):
        """
        Create an instance of MP3 compression.

        :param sample_rate: Specifies the sampling rate of sample.
        :param channels_first: Set channels first or last.
        :param apply_fit: True if applied during fitting/training.
        :param apply_predict: True if applied during predicting.
        :param device_type: Type of device on which the classifier is run, either `gpu` or `cpu`.
        :param verbose: Show progress bars.
        """
        from torch.autograd import Function

        super().__init__(
            device_type=device_type,
            is_fitted=True,
            apply_fit=apply_fit,
            apply_predict=apply_predict,
        )
        self.channels_first = channels_first
        self.sample_rate = sample_rate
        self.verbose = verbose
        self._check_params()

        self.compression_numpy = Mp3Compression(
            sample_rate=sample_rate,
            channels_first=channels_first,
            apply_fit=apply_fit,
            apply_predict=apply_predict,
            verbose=verbose,
        )

        class CompressionPyTorchNumpy(Function):  # pylint: disable=W0223
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
                # np.expand_dims(input, axis=[0, 2])
                result = self.compression_numpy.estimate_gradient(None, numpy_go)
                # result = result.squeeze()
                return grad_output.new(result)

        self._compression_pytorch_numpy = CompressionPyTorchNumpy

    def forward(
        self, x: "torch.Tensor", y: Optional["torch.Tensor"] = None
    ) -> Tuple["torch.Tensor", Optional["torch.Tensor"]]:
        """
        Apply MP3 compression to sample `x`.

        :param x: Sample to compress with shape `(length, channel)` or an array of sample arrays with shape
                  (length,) or (length, channel).
        :param y: Labels of the sample `x`. This function does not affect them in any way.
        :return: Compressed sample.
        """
        import torch  # lgtm [py/repeated-import]

        ndim = x.ndim

        if ndim == 1:
            x = torch.unsqueeze(x, dim=0)
            if self.channels_first:
                dim = 1
            else:
                dim = 2
            x = torch.unsqueeze(x, dim=dim)

        x_compressed = self._compression_pytorch_numpy.apply(x)

        if ndim == 1:
            x_compressed = torch.squeeze(x_compressed)

        return x_compressed, y

    def _check_params(self) -> None:
        if not (isinstance(self.sample_rate, int) and self.sample_rate > 0):
            raise ValueError("Sample rate be must a positive integer.")

        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be of type bool.")
