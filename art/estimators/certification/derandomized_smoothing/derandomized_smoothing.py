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
This module implements (De)Randomized Smoothing for Certifiable Defense against Patch Attacks

| Paper link: https://arxiv.org/abs/2002.10733
"""

from __future__ import absolute_import, division, print_function, unicode_literals

from abc import ABC

from typing import Optional, Union

import numpy as np
import sys
import random


class DeRandomizedSmoothingMixin(ABC):
    """
    Implementation of (De)Randomized Smoothing applied to classifier predictions as introduced
    in Levine et al. (2020).

    | Paper link: https://arxiv.org/abs/2002.10733
    """

    def __init__(
        self,
        ablation_type: str,
        ablation_size: int,
        threshold: float,
        logits: bool,
        channels_first: bool,
        *args,
        **kwargs,
    ) -> None:
        """
        Create a derandomized smoothing wrapper.
        :param ablation_type: Number of samples for smoothing.
        :param ablation_size: Size of the retained image patch.
                              An int specifying the width of the column for column ablation
                              Or an int specifying the height/width of a square for block ablation

        """
        super().__init__(*args, **kwargs)  # type: ignore
        self.ablation_type = ablation_type
        self.logits = logits
        self.threshold = threshold
        self.channels_first = channels_first

        if self.ablation_type == "column":
            self.ablator = ColumnAblator(ablation_size=ablation_size, channels_first=self.channels_first)
        elif self.ablation_type == "block":
            self.ablator = BlockAblator(ablation_size=ablation_size, channels_first=self.channels_first)
        else:
            print("Ablation type not supported!")
            sys.exit()

    def predict(self, x: np.ndarray, batch_size: int = 128, **kwargs) -> np.ndarray:
        """
        :param x: Unablated image
        :param batch_size: the batch size for the prediction
        :return: cumulative predictions after sweeping over all the ablation configurations.
        """

        if self.channels_first:
            columns_in_data = x.shape[-1]
            rows_in_data = x.shape[-2]
        else:
            columns_in_data = x.shape[-2]
            rows_in_data = x.shape[-3]

        if self.ablation_type == "column":
            for ablation_start in range(columns_in_data):  # assumes channels first
                ablated_x = self.ablator.forward(np.copy(x), start_alblation_loc=ablation_start)
                if ablation_start == 0:
                    preds = self._predict_classifier(ablated_x, batch_size=batch_size, training_mode=False)  # type: ignore
                else:
                    preds += self._predict_classifier(ablated_x, batch_size=batch_size, training_mode=False)  # type: ignore
        elif self.ablation_type == "block":
            for xcorner in range(rows_in_data):
                for ycorner in range(columns_in_data):
                    ablated_x = self.ablator.forward(np.copy(x), row_pos=xcorner, column_pos=ycorner)
                    if ycorner == 0 and xcorner == 0:
                        preds = self._predict_classifier(ablated_x, batch_size=batch_size, training_mode=False)  # type: ignore
                    else:
                        preds += self._predict_classifier(ablated_x, batch_size=batch_size, training_mode=False)  # type: ignore
        return preds


class ColumnAblator:
    """
    Implements the functionality for albating the image, and retaining only a column
    """

    def __init__(self, ablation_size, channels_first):
        super().__init__()
        self.ablation_size = ablation_size
        self.channels_first = channels_first

    def __call__(self, x: np.ndarray, start_alblation_loc: int) -> np.ndarray:
        """

        :param x:
        :param start_alblation_loc: int indicating the start column to retain across all samples in the batch
                                    or list of ints of len equal to the number of samples to have a different
                                    column retained per sample.
        :return:
        """
        return self.forward(x=x, start_alblation_loc=start_alblation_loc)

    def certify(self, preds, size_to_certify):
        """
        :param preds:
        :param size_to_certify:
        """
        # values, indices = torch.sort(torch.from_numpy(preds), dim=1, descending=True, stable=True)
        indices = np.argsort(-preds, axis=1, kind="stable")
        values = -np.sort(-preds, axis=1, kind="stable")

        num_affected_classifications = size_to_certify + self.ablation_size - 1

        margin = values[:, 0] - values[:, 1]

        certs = margin > 2 * num_affected_classifications
        tie_break_certs = (margin == 2 * num_affected_classifications) & (indices[:, 0] < indices[:, 1])
        return np.logical_or(certs, tie_break_certs)

    def column_ablate(self, x: np.ndarray, pos: int) -> np.ndarray:
        """
        Ablates the image only retaining a column starting at "pos" of width "self.ablation_size"
        :param x: input image.
        :param pos: location to start the retained column.
        :return: ablated image keeping only a column.
        """
        k = self.ablation_size
        num_of_image_columns = x.shape[-1]

        if pos + k > num_of_image_columns:
            start_of_ablation = pos + k - num_of_image_columns
            x[:, :, :, start_of_ablation:pos] = 0.0
        else:
            x[:, :, :, :pos] = 0.0
            x[:, :, :, pos + k :] = 0.0
        return x

    def forward(self, x: np.ndarray, start_alblation_loc: Optional[Union[int, list]] = None) -> np.ndarray:
        """

        :param x:
        :param start_alblation_loc: int indicating the start column to retain across all samples in the batch
                                    or list of ints of length equal to the number of samples to have a different
                                    column retained per sample.
        :return:
        """
        if not self.channels_first:
            x = np.transpose(x, (0, 3, 1, 2))

        x = np.concatenate([x, 1.0 - x], axis=1)

        if start_alblation_loc is None:
            start_alblation_loc = random.randint(0, x.shape[3])

        if isinstance(start_alblation_loc, list):
            for i, pos in enumerate(start_alblation_loc):
                x[i : i + 1] = self.column_ablate(x[i : i + 1], pos)
        else:
            x = self.column_ablate(x, start_alblation_loc)

        if not self.channels_first:
            x = np.transpose(x, (0, 2, 3, 1))

        return x


class BlockAblator:
    """
    Implements the functionality for albating the image, and retaining only a block
    """

    def __init__(self, ablation_size, channels_first):
        super().__init__()
        self.ablation_size = ablation_size
        self.channels_first = channels_first

    def __call__(
        self, x: np.ndarray, row_pos: Optional[Union[int, list]] = None, column_pos: Optional[Union[int, list]] = None
    ) -> np.ndarray:
        """

        :param x:
        :param start_alblation_loc: int indicating the start column to retain across all samples in the batch
                                    or list of ints of len equal to the number of samples to have a different
                                    column retained per sample.
        :return:
        """
        return self.forward(x=x, row_pos=row_pos, column_pos=column_pos)

    def certify(self, preds, size_to_certify):
        # values, indices = torch.sort(preds, dim=1, descending=True, stable=True)
        indices = np.argsort(-preds, axis=1, kind="stable")
        values = -np.sort(-preds, axis=1, kind="stable")

        margin = values[:, 0] - values[:, 1]

        num_affected_classifications = (size_to_certify + self.ablation_size - 1) ** 2

        certs = margin > 2 * num_affected_classifications
        tie_break_certs = (margin == 2 * num_affected_classifications) & (indices[:, 0] < indices[:, 1])
        return np.logical_or(certs, tie_break_certs)

    def forward(
        self, x: np.ndarray, row_pos: Optional[Union[int, list]] = None, column_pos: Optional[Union[int, list]] = None
    ) -> np.ndarray:
        """

        :param x:
        :param row_pos:
        :param column_pos:
        :return:
        """
        if not self.channels_first:
            x = np.transpose(x, (0, 3, 1, 2))

        if row_pos is None:
            row_pos = random.randint(0, x.shape[2])
        if column_pos is None:
            column_pos = random.randint(0, x.shape[3])

        x = np.concatenate([x, 1.0 - x], axis=1)

        if isinstance(row_pos, list) and isinstance(column_pos, list):
            for i, (r, c) in enumerate(zip(row_pos, column_pos)):
                x[i : i + 1] = self.block_ablate(x[i : i + 1], row_pos=r, column_pos=c)
        elif isinstance(row_pos, int) and isinstance(column_pos, int):
            x = self.block_ablate(x, row_pos=row_pos, column_pos=column_pos)

        if not self.channels_first:
            x = np.transpose(x, (0, 2, 3, 1))
        return x

    def block_ablate(self, data: np.ndarray, row_pos: int, column_pos: int) -> np.ndarray:
        """

        :param data:
        :param row_pos:
        :param column_pos:
        :return:
        """

        k = self.ablation_size
        num_of_image_columns = data.shape[3]
        num_of_image_rows = data.shape[2]

        if row_pos + k > data.shape[2] and column_pos + k > data.shape[3]:
            start_of_ablation = column_pos + k - num_of_image_columns
            data[:, :, :, start_of_ablation:column_pos] = 0.0

            start_of_ablation = row_pos + k - num_of_image_rows
            data[:, :, start_of_ablation:row_pos, :] = 0.0

        # only the row wraps
        elif row_pos + k > data.shape[2] and column_pos + k <= data.shape[3]:
            data[:, :, :, :column_pos] = 0.0
            data[:, :, :, column_pos + k :] = 0.0

            start_of_ablation = row_pos + k - num_of_image_rows
            data[:, :, start_of_ablation:row_pos, :] = 0.0

        # only column wraps
        elif row_pos + k <= data.shape[2] and column_pos + k > data.shape[3]:
            start_of_ablation = column_pos + k - num_of_image_columns
            data[:, :, :, start_of_ablation:column_pos] = 0.0

            data[:, :, :row_pos, :] = 0.0
            data[:, :, row_pos + k :, :] = 0.0

        # neither wraps
        elif row_pos + k <= data.shape[2] and column_pos + k <= data.shape[3]:
            data[:, :, :, :column_pos] = 0.0
            data[:, :, :, column_pos + k :] = 0.0

            data[:, :, :row_pos, :] = 0.0
            data[:, :, row_pos + k :, :] = 0.0
        else:
            print(row_pos, column_pos, k)
            print("no ablation!")
            sys.exit()
        return data
