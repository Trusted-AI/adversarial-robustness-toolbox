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
This module implements the classifier `TensorFlowGenerator` for TensorFlow models.
"""
from __future__ import absolute_import, division, print_function, unicode_literals, annotations

import logging
from typing import Any, TYPE_CHECKING

import numpy as np

from art.estimators.generation.generator import GeneratorMixin
from art.estimators.tensorflow import TensorFlowV2Estimator

if TYPE_CHECKING:

    import tensorflow.compat.v1 as tf

    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor import Preprocessor
    from art.defences.postprocessor import Postprocessor

logger = logging.getLogger(__name__)


class TensorFlowV2Generator(GeneratorMixin, TensorFlowV2Estimator):
    """
    This class implements a DGM with the TensorFlow framework.
    """

    estimator_params = TensorFlowV2Estimator.estimator_params + [
        "encoding_length",
    ]

    def __init__(
        self,
        encoding_length: int,
        model: "tf.Tensor",
        channels_first: bool = False,
        clip_values: "CLIP_VALUES_TYPE" | None = None,
        preprocessing_defences: "Preprocessor" | list["Preprocessor"] | None = None,
        postprocessing_defences: "Postprocessor" | list["Postprocessor"] | None = None,
        preprocessing: "PREPROCESSING_TYPE" = (0.0, 1.0),
    ):
        """
        Initialization specific to TensorFlow generator implementations.

        :param encoding_length: length of the input seed
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
        self._encoding_length = encoding_length

    @property
    def model(self) -> "tf.Tensor":
        """
        :return: The generator tensor.
        """
        return self._model

    @property
    def encoding_length(self) -> int:
        """
        :return: The length of the encoding size output.
        """
        return self._encoding_length

    @property
    def input_shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    def predict(self, x: np.ndarray, batch_size: int = 128, training_mode: bool = False, **kwargs) -> np.ndarray:
        """
        Perform projections over a batch of encodings.

        :param x: Encodings.
        :param batch_size: Batch size.
        :param training_mode: `True` for model set to training mode and `False` for model set to evaluation mode.
        :return: Array of prediction projections of shape `(num_inputs, nb_classes)`.
        """
        # Run prediction with batch processing
        results_list = []
        num_batch = int(np.ceil(len(x) / float(batch_size)))
        for m in range(num_batch):
            # Batch indexes
            begin, end = (
                m * batch_size,
                min((m + 1) * batch_size, x.shape[0]),
            )

            # Run prediction
            results_list.append(self._model(x[begin:end], training=training_mode).numpy())

        results = np.vstack(results_list)

        return results

    def loss_gradient(self, x, y, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def fit(self, x, y, batch_size=128, nb_epochs=10, **kwargs):
        """
        Do nothing.
        """
        raise NotImplementedError

    def get_activations(self, x: np.ndarray, layer: int | str, batch_size: int, framework: bool = False) -> np.ndarray:
        """
        Do nothing.
        """
        raise NotImplementedError
