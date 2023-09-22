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
This module implements the abstract base class for the ablators.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    # pylint: disable=C0412
    import tensorflow as tf
    import torch


class BaseAblator(ABC):
    """
    Base class defining the methods used for the ablators.
    """

    @abstractmethod
    def __call__(
        self, x: np.ndarray, column_pos: Optional[Union[int, list]] = None, row_pos: Optional[Union[int, list]] = None
    ) -> np.ndarray:
        """
        Ablate the image x at location specified by "column_pos" for the case of column ablation or at the location
        specified by "column_pos" and "row_pos" in the case of block ablation.

        :param x: input image.
        :param column_pos: column position to specify where to retain the image
        :param row_pos: row position to specify where to retain the image. Not used for ablation type "column".
        """
        raise NotImplementedError

    @abstractmethod
    def certify(
        self, pred_counts: np.ndarray, size_to_certify: int, label: Union[np.ndarray, "tf.Tensor"]
    ) -> Union[Tuple["tf.Tensor", "tf.Tensor", "tf.Tensor"], Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]]:
        """
        Checks if based on the predictions supplied the classifications over the ablated datapoints result in a
        certified prediction against a patch attack of size size_to_certify.

        :param pred_counts: The cumulative predictions of the classifier over the ablation locations.
        :param size_to_certify: The size of the patch to check against.
        :param label: ground truth labels
        """
        raise NotImplementedError

    @abstractmethod
    def ablate(self, x: np.ndarray, column_pos: int, row_pos: int) -> Union[np.ndarray, "torch.Tensor"]:
        """
        Ablate the image x at location specified by "column_pos" for the case of column ablation or at the location
        specified by "column_pos" and "row_pos" in the case of block ablation.

        :param x: input image.
        :param column_pos: column position to specify where to retain the image
        :param row_pos: row position to specify where to retain the image. Not used for ablation type "column".
        """
        raise NotImplementedError

    @abstractmethod
    def forward(
        self, x: np.ndarray, column_pos: Optional[int] = None, row_pos: Optional[int] = None
    ) -> Union[np.ndarray, "torch.Tensor"]:
        """
        Ablate batch of data at locations specified by column_pos and row_pos

        :param x: input image.
        :param column_pos: column position to specify where to retain the image
        :param row_pos: row position to specify where to retain the image. Not used for ablation type "column".
        """
        raise NotImplementedError
