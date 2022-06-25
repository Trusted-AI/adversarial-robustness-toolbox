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

from abc import ABC, abstractmethod
from typing import Optional, Union, TYPE_CHECKING
import random

import numpy as np

if TYPE_CHECKING:
    from art.utils import ABLATOR_TYPE


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

        :param ablation_type: The type of ablations to perform. Currently must be either "column", "row", or "block"
        :param ablation_size: Size of the retained image patch.
                              An int specifying the width of the column for column ablation
                              Or an int specifying the height/width of a square for block ablation
        :param threshold: The minimum threshold to count a prediction.
        :param logits: if the model returns logits or normalized probabilities
        :param channels_first: If the channels are first or last.
        """
        super().__init__(*args, **kwargs)  # type: ignore
        self.ablation_type = ablation_type
        self.logits = logits
        self.threshold = threshold
        self._channels_first = channels_first
        if TYPE_CHECKING:
            self.ablator: ABLATOR_TYPE  # pylint: disable=used-before-assignment

        if self.ablation_type in {"column", "row"}:
            row_ablation_mode = self.ablation_type == "row"
            self.ablator = ColumnAblator(
                ablation_size=ablation_size, channels_first=self._channels_first, row_ablation_mode=row_ablation_mode
            )
        elif self.ablation_type == "block":
            self.ablator = BlockAblator(ablation_size=ablation_size, channels_first=self._channels_first)
        else:
            raise ValueError("Ablation type not supported. Must be either column or block")

    def _predict_classifier(self, x: np.ndarray, batch_size: int, training_mode: bool, **kwargs) -> np.ndarray:
        """
        Perform prediction for a batch of inputs.

        :param x: Input samples.
        :param batch_size: Size of batches.
        :param training_mode: `True` for model set to training mode and `'False` for model set to evaluation mode.
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        """
        raise NotImplementedError

    def predict(self, x: np.ndarray, batch_size: int = 128, training_mode: bool = False, **kwargs) -> np.ndarray:
        """
        Performs cumulative predictions over every ablation location

        :param x: Unablated image
        :param batch_size: the batch size for the prediction
        :param training_mode: if to run the classifier in training mode
        :return: cumulative predictions after sweeping over all the ablation configurations.
        """
        if self._channels_first:
            columns_in_data = x.shape[-1]
            rows_in_data = x.shape[-2]
        else:
            columns_in_data = x.shape[-2]
            rows_in_data = x.shape[-3]

        if self.ablation_type in {"column", "row"}:
            if self.ablation_type == "column":
                ablate_over_range = columns_in_data
            else:
                # image will be transposed, so loop over the number of rows
                ablate_over_range = rows_in_data

            for ablation_start in range(ablate_over_range):
                ablated_x = self.ablator.forward(np.copy(x), column_pos=ablation_start)
                if ablation_start == 0:
                    preds = self._predict_classifier(
                        ablated_x, batch_size=batch_size, training_mode=training_mode, **kwargs
                    )
                else:
                    preds += self._predict_classifier(
                        ablated_x, batch_size=batch_size, training_mode=training_mode, **kwargs
                    )
        elif self.ablation_type == "block":
            for xcorner in range(rows_in_data):
                for ycorner in range(columns_in_data):
                    ablated_x = self.ablator.forward(np.copy(x), row_pos=xcorner, column_pos=ycorner)
                    if ycorner == 0 and xcorner == 0:
                        preds = self._predict_classifier(
                            ablated_x, batch_size=batch_size, training_mode=training_mode, **kwargs
                        )
                    else:
                        preds += self._predict_classifier(
                            ablated_x, batch_size=batch_size, training_mode=training_mode, **kwargs
                        )
        return preds


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
    def certify(self, preds: np.ndarray, size_to_certify: int):
        """
        Checks if based on the predictions supplied the classifications over the ablated datapoints result in a
        certified prediction against a patch attack of size size_to_certify.

        :param preds: The cumulative predictions of the classifier over the ablation locations.
        :param size_to_certify: The size of the patch to check against.
        """
        raise NotImplementedError

    @abstractmethod
    def ablate(self, x: np.ndarray, column_pos: int, row_pos: int) -> np.ndarray:
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
        self, x: np.ndarray, column_pos: Optional[Union[int, list]] = None, row_pos: Optional[Union[int, list]] = None
    ) -> np.ndarray:
        """
        Ablate batch of data at locations specified by column_pos and row_pos

        :param x: input image.
        :param column_pos: column position to specify where to retain the image
        :param row_pos: row position to specify where to retain the image. Not used for ablation type "column".
        """
        raise NotImplementedError


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

    def certify(self, preds: np.ndarray, size_to_certify: int) -> np.ndarray:
        """
        Checks if based on the predictions supplied the classifications over the ablated datapoints result in a
        certified prediction against a patch attack of size size_to_certify.

        :param preds: The cumulative predictions of the classifier over the ablation locations.
        :param size_to_certify: The size of the patch to check against.
        :return: Array of bools indicating if a point is certified against the given patch dimensions.
        """
        indices = np.argsort(-preds, axis=1, kind="stable")
        values = np.take_along_axis(np.copy(preds), indices, axis=1)

        num_affected_classifications = size_to_certify + self.ablation_size - 1

        margin = values[:, 0] - values[:, 1]

        certs = margin > 2 * num_affected_classifications
        tie_break_certs = (margin == 2 * num_affected_classifications) & (indices[:, 0] < indices[:, 1])
        return np.logical_or(certs, tie_break_certs)

    def ablate(self, x: np.ndarray, column_pos: int, row_pos=None) -> np.ndarray:
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

    def certify(self, preds: np.ndarray, size_to_certify: int) -> np.ndarray:
        """
        Checks if based on the predictions supplied the classifications over the ablated datapoints result in a
        certified prediction against a patch attack of size size_to_certify.

        :param preds: The cumulative predictions of the classifier over the ablation locations.
        :param size_to_certify: The size of the patch to check against.
        :return: Array of bools indicating if a point is certified against the given patch dimensions.
        """
        indices = np.argsort(-preds, axis=1, kind="stable")
        values = np.take_along_axis(np.copy(preds), indices, axis=1)
        margin = values[:, 0] - values[:, 1]

        num_affected_classifications = (size_to_certify + self.ablation_size - 1) ** 2

        certs = margin > 2 * num_affected_classifications
        tie_break_certs = (margin == 2 * num_affected_classifications) & (indices[:, 0] < indices[:, 1])
        return np.logical_or(certs, tie_break_certs)

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
        num_of_image_columns = x.shape[3]
        num_of_image_rows = x.shape[2]

        if row_pos + k > x.shape[2] and column_pos + k > x.shape[3]:
            start_of_ablation = column_pos + k - num_of_image_columns
            x[:, :, :, start_of_ablation:column_pos] = 0.0

            start_of_ablation = row_pos + k - num_of_image_rows
            x[:, :, start_of_ablation:row_pos, :] = 0.0

        # only the row wraps
        elif row_pos + k > x.shape[2] and column_pos + k <= x.shape[3]:
            x[:, :, :, :column_pos] = 0.0
            x[:, :, :, column_pos + k :] = 0.0

            start_of_ablation = row_pos + k - num_of_image_rows
            x[:, :, start_of_ablation:row_pos, :] = 0.0

        # only column wraps
        elif row_pos + k <= x.shape[2] and column_pos + k > x.shape[3]:
            start_of_ablation = column_pos + k - num_of_image_columns
            x[:, :, :, start_of_ablation:column_pos] = 0.0

            x[:, :, :row_pos, :] = 0.0
            x[:, :, row_pos + k :, :] = 0.0

        # neither wraps
        elif row_pos + k <= x.shape[2] and column_pos + k <= x.shape[3]:
            x[:, :, :, :column_pos] = 0.0
            x[:, :, :, column_pos + k :] = 0.0

            x[:, :, :row_pos, :] = 0.0
            x[:, :, row_pos + k :, :] = 0.0
        else:
            raise ValueError(f"Ablation failed on row: {row_pos} and column: {column_pos} with size {k}")

        return x
