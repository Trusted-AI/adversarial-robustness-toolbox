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
from typing import TYPE_CHECKING, Optional, Tuple, Union

import numpy as np

from art.config import ART_NUMPY_DTYPE
from art.preprocessing.preprocessing import PreprocessorTensorFlowV2
from art.preprocessing.standardisation_mean_std.utils import broadcastable_mean_std

if TYPE_CHECKING:
    import tensorflow as tf

logger = logging.getLogger(__name__)


class StandardisationMeanStdTensorFlow(PreprocessorTensorFlowV2):
    """
    Implement the standardisation with mean and standard deviation.
    """

    params = ["mean", "std", "apply_fit", "apply_predict"]

    def __init__(
        self,
        mean: Union[float, np.ndarray] = 0.0,
        std: Union[float, np.ndarray] = 1.0,
        apply_fit: bool = True,
        apply_predict: bool = True,
    ):
        """
        Create an instance of StandardisationMeanStdTensorFlow.

        :param mean: Mean.
        :param std: Standard Deviation.
        """
        super().__init__(is_fitted=True, apply_fit=apply_fit, apply_predict=apply_predict)
        self.mean = np.asarray(mean, dtype=ART_NUMPY_DTYPE)
        self.std = np.asarray(std, dtype=ART_NUMPY_DTYPE)
        self._check_params()

        # init broadcastable mean and std for lazy loading
        self._broadcastable_mean: Optional[np.ndarray] = None
        self._broadcastable_std: Optional[np.ndarray] = None

    def forward(self, x: "tf.Tensor", y: Optional["tf.Tensor"] = None) -> Tuple["tf.Tensor", Optional["tf.Tensor"]]:
        """
        Apply standardisation with mean and standard deviation to input `x`.

        :param x: Input samples to standardise.
        :param y: Label data, will not be affected by this preprocessing.
        :return: Standardised input samples and unmodified labels.
        """
        import tensorflow as tf  # lgtm [py/repeated-import]

        if self._broadcastable_mean is None:
            self._broadcastable_mean, self._broadcastable_std = broadcastable_mean_std(x, self.mean, self.std)

        x_norm = x - self._broadcastable_mean
        x_norm = x_norm / self._broadcastable_std
        x_norm = tf.cast(x_norm, dtype=ART_NUMPY_DTYPE)  # pylint: disable=E1123,E1120

        return x_norm, y

    def _check_params(self) -> None:
        pass

    def __repr__(self):
        return (
            f"StandardisationMeanStdTensorFlow(mean={self.mean}, std={self.std}, apply_fit={self.apply_fit}, "
            f"apply_predict={self.apply_predict})"
        )
