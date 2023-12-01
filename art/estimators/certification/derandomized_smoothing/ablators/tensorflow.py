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

from typing import Optional, Union, Tuple, TYPE_CHECKING
import random

import numpy as np

from art.estimators.certification.derandomized_smoothing.ablators.ablate import BaseAblator

if TYPE_CHECKING:
    # pylint: disable=C0412
    import tensorflow as tf


class ColumnAblator(BaseAblator):
    """
    Implements the functionality for albating the image, and retaining only a column
    """

    def __init__(self, ablation_size: int, channels_first: bool, row_ablation_mode: bool = False):
        """
        Creates an ablator which will retain columns by default, or rows if operating in row_ablation_mode

        :param ablation_size: Size of the column (or row if running in row_ablation_mode) to retain.
        :param channels_first: If the input data will be in channels first or channels last format.
        :param row_ablation_mode: if True then the ablator will function by retaining rows rather than columns.
        """
        super().__init__()
        self.ablation_size = ablation_size
        self.channels_first = channels_first
        self.row_ablation_mode = row_ablation_mode

    def __call__(
        self, x: np.ndarray, column_pos: Optional[Union[int, list]] = None, row_pos: Optional[Union[int, list]] = None
    ) -> np.ndarray:
        """
        Performs ablation on the input x. If no column_pos is specified a random location will be selected.

        :param x: input image.
        :param column_pos: int indicating the start column to retain across all samples in the batch
                           or list of ints of length equal to the number of samples to have a different
                           column retained per sample. If not supplied a random location will be selected.
                           NB, if row_ablation_mode is true then this will be used to act on the rows through
                           transposing the image.
        :param row_pos: Unused
        :return: ablated image keeping only a column.
        """
        return self.forward(x=x, column_pos=column_pos)

    def certify(
        self, pred_counts: "tf.Tensor", size_to_certify: int, label: Union[np.ndarray, "tf.Tensor"]
    ) -> Tuple["tf.Tensor", "tf.Tensor", "tf.Tensor"]:
        """
        Checks if based on the predictions supplied the classifications over the ablated datapoints result in a
        certified prediction against a patch attack of size size_to_certify.

        :param preds: The cumulative predictions of the classifier over the ablation locations.
        :param size_to_certify: The size of the patch to check against.
        :param label: Ground truth labels
        :return: A tuple consisting of: the certified predictions,
                 the predictions which were certified and also correct,
                 and the most predicted class across the different ablations on the input.
        """
        import tensorflow as tf

        result = tf.math.top_k(pred_counts, k=2)

        top_predicted_class, second_predicted_class = result.indices[:, 0], result.indices[:, 1]
        top_class_counts, second_class_counts = result.values[:, 0], result.values[:, 1]

        certs = (top_class_counts - second_class_counts) > 2 * (size_to_certify + self.ablation_size - 1)

        tie_break_certs = (
            (top_class_counts - second_class_counts) == 2 * (size_to_certify + self.ablation_size - 1)
        ) & (top_predicted_class < second_predicted_class)
        cert = tf.math.logical_or(certs, tie_break_certs)

        # NB, newer versions of pylint do not require the disable.
        if label.ndim > 1:
            cert_and_correct = cert & (
                tf.math.argmax(label, axis=1)
                == tf.cast(  # pylint: disable=E1120, E1123
                    top_predicted_class, dtype=tf.math.argmax(label, axis=1).dtype
                )
            )
        else:
            cert_and_correct = cert & (
                label == tf.cast(top_predicted_class, dtype=label.dtype)  # pylint: disable=E1120, E1123
            )

        return cert, cert_and_correct, top_predicted_class

    def ablate(self, x: np.ndarray, column_pos: int, row_pos: Optional[int] = None) -> np.ndarray:
        """
        Ablates the image only retaining a column starting at "pos" of width "self.ablation_size"

        :param x: input image.
        :param column_pos: location to start the retained column. NB, if row_ablation_mode is true then this will
                           be used to act on the rows through transposing the image.
        :param row_pos: Unused.
        :return: ablated image keeping only a column.
        """
        if self.row_ablation_mode:
            x = np.transpose(x, (0, 1, 3, 2))

        k = self.ablation_size
        num_of_image_columns = x.shape[-1]

        if column_pos + k > num_of_image_columns:
            start_of_ablation = column_pos + k - num_of_image_columns
            x[:, :, :, start_of_ablation:column_pos] = 0.0
        else:
            x[:, :, :, :column_pos] = 0.0
            x[:, :, :, column_pos + k :] = 0.0

        if self.row_ablation_mode:
            x = np.transpose(x, (0, 1, 3, 2))

        return x

    def forward(
        self, x: np.ndarray, column_pos: Optional[Union[int, list]] = None, row_pos: Optional[Union[int, list]] = None
    ) -> np.ndarray:
        """
        Performs ablation on the input x. If no column_pos is specified a random location will be selected.

        :param x: input batch.
        :param column_pos: int indicating the start column to retain across all samples in the batch
                           or list of ints of length equal to the number of samples to have a different
                           column retained per sample. If not supplied a random location will be selected.
                           NB, if row_ablation_mode is true then this will be used to act on the rows through
                           transposing the image.
        :param row_pos: Unused.
        :return: Batch ablated according to the locations in column_pos. Data is channel extended to indicate to a
                 model if a position is ablated.
        """
        if not self.channels_first:
            x = np.transpose(x, (0, 3, 1, 2))

        x = np.concatenate([x, 1.0 - x], axis=1)

        if column_pos is None:
            column_pos = random.randint(0, x.shape[3])

        if isinstance(column_pos, list):
            assert len(column_pos) == len(x)
            for i, pos in enumerate(column_pos):
                x[i : i + 1] = self.ablate(x[i : i + 1], pos)
        else:
            x = self.ablate(x, column_pos)

        if not self.channels_first:
            x = np.transpose(x, (0, 2, 3, 1))

        return x


class BlockAblator(BaseAblator):
    """
    Implements the functionality for albating the image, and retaining only a block
    """

    def __init__(self, ablation_size: int, channels_first: bool):
        """
        Creates an ablator which will retain blocks of the input data.

        :param ablation_size: Size of the column (or row if running in row_ablation_mode) to retain.
        :param channels_first: If the input data will be in channels first or channels last format.
        """
        super().__init__()
        self.ablation_size = ablation_size
        self.channels_first = channels_first

    def __call__(
        self, x: np.ndarray, column_pos: Optional[Union[int, list]] = None, row_pos: Optional[Union[int, list]] = None
    ) -> np.ndarray:
        """
        Performs ablation on the input x. If no row_pos/column_pos is specified a random location will be selected.

        :param x: input data
        :param column_pos: Specifies the column index to retain the image block. Either an int to apply the same
                   position to all images in a batch, or a list of ints to apply a different
                   column position per datapoint.
        :param row_pos: Specifies the row index to retain the image block. Either an int to apply the same position to
                        all images in a batch, or a list of ints to apply a different row position per datapoint.
        :return: Data ablated at all locations aside from the specified block. Data is channel extended to indicate to a
                 model if a position is ablated.
        """
        return self.forward(x=x, row_pos=row_pos, column_pos=column_pos)

    def certify(
        self, pred_counts: Union["tf.Tensor", np.ndarray], size_to_certify: int, label: Union[np.ndarray, "tf.Tensor"]
    ) -> Tuple["tf.Tensor", "tf.Tensor", "tf.Tensor"]:
        """
        Checks if based on the predictions supplied the classifications over the ablated datapoints result in a
        certified prediction against a patch attack of size size_to_certify.

        :param pred_counts: The cumulative predictions of the classifier over the ablation locations.
        :param size_to_certify: The size of the patch to check against.
        :param label: Ground truth labels
        :return: A tuple consisting of: the certified predictions,
                 the predictions which were certified and also correct,
                 and the most predicted class across the different ablations on the input.
        """
        import tensorflow as tf

        result = tf.math.top_k(pred_counts, k=2)

        top_predicted_class, second_predicted_class = result.indices[:, 0], result.indices[:, 1]
        top_class_counts, second_class_counts = result.values[:, 0], result.values[:, 1]

        certs = (top_class_counts - second_class_counts) > 2 * (size_to_certify + self.ablation_size - 1) ** 2
        tie_break_certs = (
            (top_class_counts - second_class_counts) == 2 * (size_to_certify + self.ablation_size - 1) ** 2
        ) & (top_predicted_class < second_predicted_class)
        cert = tf.math.logical_or(certs, tie_break_certs)

        # NB, newer versions of pylint do not require the disable.
        if label.ndim > 1:
            cert_and_correct = cert & (
                tf.math.argmax(label, axis=1)
                == tf.cast(  # pylint: disable=E1120, E1123
                    top_predicted_class, dtype=tf.math.argmax(label, axis=1).dtype
                )
            )
        else:
            cert_and_correct = cert & (
                label == tf.cast(top_predicted_class, dtype=label.dtype)  # pylint: disable=E1120, E1123
            )

        return cert, cert_and_correct, top_predicted_class

    def forward(
        self,
        x: np.ndarray,
        column_pos: Optional[Union[int, list]] = None,
        row_pos: Optional[Union[int, list]] = None,
    ) -> np.ndarray:
        """
        Performs ablation on the input x. If no column_pos/row_pos are specified a random location will be selected.

        :param x: input data
        :param row_pos: Specifies the row index to retain the image block. Either an int to apply the same position to
                        all images in a batch, or a list of ints to apply a different row position per datapoint.
        :param column_pos: Specifies the column index to retain the image block. Either an int to apply the same
                           position to all images in a batch, or a list of ints to apply a different
                           column position per datapoint.
        :return: Data ablated at all locations aside from the specified block. Data is channel extended to indicate to a
                 model if a position is ablated.
        """
        if not self.channels_first:
            x = np.transpose(x, (0, 3, 1, 2))

        if row_pos is None:
            row_pos = random.randint(0, x.shape[2])
        if column_pos is None:
            column_pos = random.randint(0, x.shape[3])

        x = np.concatenate([x, 1.0 - x], axis=1)

        if isinstance(row_pos, list) and isinstance(column_pos, list):
            for i, (row, col) in enumerate(zip(row_pos, column_pos)):
                x[i : i + 1] = self.ablate(x[i : i + 1], row_pos=row, column_pos=col)
        elif isinstance(row_pos, int) and isinstance(column_pos, int):
            x = self.ablate(x, row_pos=row_pos, column_pos=column_pos)

        if not self.channels_first:
            x = np.transpose(x, (0, 2, 3, 1))
        return x

    def ablate(self, x: np.ndarray, column_pos: int, row_pos: int) -> np.ndarray:
        """
        Ablates the image only retaining a block starting at (row_pos, column_pos) of height/width "self.ablation_size"

        :param x: input data
        :param row_pos: Specifies the row index where to retain the image block.
        :param column_pos: Specifies the column index where to retain the image block.
        :return: Data ablated at all locations aside from the specified block.
        """
        k = self.ablation_size
        # Column ablations
        if column_pos + k > x.shape[-1]:
            x[:, :, :, (column_pos + k) % x.shape[-1] : column_pos] = 0.0
        else:
            x[:, :, :, :column_pos] = 0.0
            x[:, :, :, column_pos + k :] = 0.0

        # Row ablations
        if row_pos + k > x.shape[-2]:
            x[:, :, (row_pos + k) % x.shape[-2] : row_pos, :] = 0.0
        else:
            x[:, :, :row_pos, :] = 0.0
            x[:, :, row_pos + k :, :] = 0.0
        return x
