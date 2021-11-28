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
This module implements the classifier `PyTorchEncoder` for PyTorch models.
"""
import logging
from typing import Tuple, Union, Optional, TYPE_CHECKING, List

import torch
import numpy as np

from art.estimators import PyTorchEstimator
from art.estimators.encoding import EncoderMixin

if TYPE_CHECKING:
    # pylint: disable=C0412

    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor import Preprocessor
    from art.defences.postprocessor import Postprocessor

logger = logging.getLogger(__name__)


class PyTorchEncoder(EncoderMixin, PyTorchEstimator):
    """
    This class implements an encoder model using the PyTorch framework.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        input_shape: Tuple[int, ...],
        encoding_length: int,
        channels_first: bool = False,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        preprocessing_defences: Union[
            "Preprocessor", List["Preprocessor"], None
        ] = None,
        postprocessing_defences: Union[
            "Postprocessor", List["Postprocessor"], None
        ] = None,
        preprocessing: "PREPROCESSING_TYPE" = (0.0, 1.0),
    ):
        """
        Initialization specific to encoder estimator implementation in PyTorch.

        :param model: TensorFlow model, neural network or other.
        :param channels_first: Set channels first or last.
        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and
                            maximum values allowed for features. If floats are provided, these will be used as the range
                            of all features. If arrays are provided, each value will be considered the bound for a
                            feature, thus the shape of clip values needs to match the total number of features.
        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
        :param preprocessing: Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be
                              used for data preprocessing. The first value will be subtracted from the input. The input
                              will then be divided by the second one.
        """

        super().__init__(
            model=model,
            clip_values=clip_values,
            channels_first=channels_first,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )

        self._input_shape = input_shape
        self._encoding_length = encoding_length

    def predict(self, x: "np.ndarray", batch_size: int = 128, **kwargs):
        """
        Perform prediction for a batch of inputs.

        :param x: Input samples.
        :param batch_size: Batch size.
        :return: Array of encoding predictions of shape `(num_inputs, encoding_length)`.
        """
        logger.info("Encoding input")
        with torch.no_grad():
            y = self._model(torch.Tensor(x))
        return y

    @property
    def encoding_length(self) -> int:
        """
        Returns the length of the encoding size output.

        :return: The length of the encoding size output.
        """
        return self._encoding_length

    def get_activations(
        self,
        x: np.ndarray,
        layer: Union[int, str],
        batch_size: int,
        framework: bool = False,
    ) -> np.ndarray:
        """
        Do nothing.
        """
        raise NotImplementedError

    def loss_gradient(self, x, y, **kwargs):
        """
        No gradients to compute for this method; do nothing.
        """
        raise NotImplementedError

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """
        Return the shape of one input sample.

        :return: Shape of one input sample.
        """
        return self._input_shape  # type: ignore

    def compute_loss(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:

        """
        Do nothing.
        """
        raise NotImplementedError
