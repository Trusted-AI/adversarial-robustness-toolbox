from __future__ import absolute_import, division, print_function, unicode_literals

from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
import tensorflow as tf
import torch

if TYPE_CHECKING:
    # pylint: disable=C0412
    import tensorflow as tf
    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE, ABLATOR_TYPE
    from art.defences.preprocessor import Preprocessor
    from art.defences.postprocessor import Postprocessor


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
        self, preds: Union["tf.Tensor", torch.Tensor], size_to_certify: int, label: Union[np.ndarray, "tf.Tensor"]
    ) -> Tuple["tf.Tensor", "tf.Tensor", "tf.Tensor"]:
        """
        Checks if based on the predictions supplied the classifications over the ablated datapoints result in a
        certified prediction against a patch attack of size size_to_certify.

        :param preds: The cumulative predictions of the classifier over the ablation locations.
        :param size_to_certify: The size of the patch to check against.
        :param label: ground truth labels
        """
        raise NotImplementedError

    @abstractmethod
    def ablate(
        self, x: Union[np.ndarray, torch.Tensor], column_pos: int, row_pos: int
    ) -> Union[np.ndarray, torch.Tensor]:
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
