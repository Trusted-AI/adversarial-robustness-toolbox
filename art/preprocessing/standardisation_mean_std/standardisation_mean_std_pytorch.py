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
This module implements the standardisation with mean and standard deviation.
"""
import logging
from typing import Optional, Tuple, TYPE_CHECKING

from art.preprocessing.preprocessing import PreprocessorPyTorch

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


class StandardisationMeanStdPyTorch(PreprocessorPyTorch):
    """
    Implement the standardisation with mean and standard deviation.
    """

    params = ["mean", "std", "apply_fit", "apply_predict"]

    def __init__(
        self,
        mean: float = 0.0,
        std: float = 1.0,
        apply_fit: bool = True,
        apply_predict: bool = True,
        device_type: str = "gpu",
    ):
        """
        Create an instance of StandardisationMeanStdPyTorch.

        :param mean: Mean.
        :param std: Standard Deviation.
        """
        import torch  # lgtm [py/repeated-import]

        super().__init__(is_fitted=True, apply_fit=apply_fit, apply_predict=apply_predict)
        self.mean = mean
        self.std = std
        self._check_params()

        # Set device
        if device_type == "cpu" or not torch.cuda.is_available():
            self._device = torch.device("cpu")
        else:
            cuda_idx = torch.cuda.current_device()
            self._device = torch.device("cuda:{}".format(cuda_idx))

    def forward(
        self, x: "torch.Tensor", y: Optional["torch.Tensor"] = None
    ) -> Tuple["torch.Tensor", Optional["torch.Tensor"]]:
        """
        Apply standardisation with mean and standard deviation to input `x`.
        """
        import torch  # lgtm [py/repeated-import]

        mean = torch.tensor(self.mean, device=self._device, dtype=torch.float32)
        std = torch.tensor(self.std, device=self._device, dtype=torch.float32)

        x_norm = x - mean
        x_norm = x_norm / std

        return x_norm, y

    def _check_params(self) -> None:
        pass

    def __repr__(self):
        return "StandardisationMeanStdPyTorch(mean={}, std={}, apply_fit={}, apply_predict={}, device={})".format(
            self.mean, self.std, self.apply_fit, self.apply_predict, self._device
        )
