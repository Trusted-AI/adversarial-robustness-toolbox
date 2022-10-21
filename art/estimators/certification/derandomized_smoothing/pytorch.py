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

import logging
from typing import List, Optional, Tuple, Union, Any, TYPE_CHECKING
import random

import numpy as np
from tqdm import tqdm

from art.config import ART_NUMPY_DTYPE
from art.estimators.classification.pytorch import PyTorchClassifier
from art.estimators.certification.derandomized_smoothing.derandomized_smoothing import DeRandomizedSmoothingMixin
from art.utils import check_and_transform_label_format

if TYPE_CHECKING:
    # pylint: disable=C0412
    import torch

    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor import Preprocessor
    from art.defences.postprocessor import Postprocessor

logger = logging.getLogger(__name__)


class PyTorchDeRandomizedSmoothing(DeRandomizedSmoothingMixin, PyTorchClassifier):
    """
    Implementation of (De)Randomized Smoothing applied to classifier predictions as introduced
    in Levine et al. (2020).

    | Paper link: https://arxiv.org/abs/2002.10733
    """

    estimator_params = PyTorchClassifier.estimator_params + ["ablation_type", "ablation_size", "threshold", "logits"]

    def __init__(
        self,
        model: "torch.nn.Module",
        loss: "torch.nn.modules.loss._Loss",
        input_shape: Tuple[int, ...],
        nb_classes: int,
        ablation_type: str,
        ablation_size: int,
        threshold: float,
        logits: bool,
        optimizer: Optional["torch.optim.Optimizer"] = None,  # type: ignore
        channels_first: bool = True,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: "PREPROCESSING_TYPE" = (0.0, 1.0),
        device_type: str = "gpu",
    ):
        """
        Create a derandomized smoothing classifier.

        :param model: PyTorch model. The output of the model can be logits, probabilities or anything else. Logits
               output should be preferred where possible to ensure attack efficiency.
        :param loss: The loss function for which to compute gradients for training. The target label must be raw
               categorical, i.e. not converted to one-hot encoding.
        :param input_shape: The shape of one input instance.
        :param nb_classes: The number of classes of the model.
        :param ablation_type: The type of ablation to perform, must be either "column" or "block"
        :param ablation_size: The size of the data portion to retain after ablation. Will be a column of size N for
                              "column" ablation type or a NxN square for ablation of type "block"
        :param threshold: The minimum threshold to count a prediction.
        :param logits: if the model returns logits or normalized probabilities
        :param optimizer: The optimizer used to train the classifier.
        :param channels_first: Set channels first or last.
        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and
               maximum values allowed for features. If floats are provided, these will be used as the range of all
               features. If arrays are provided, each value will be considered the bound for a feature, thus
               the shape of clip values needs to match the total number of features.
        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
        :param preprocessing: Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :param device_type: Type of device on which the classifier is run, either `gpu` or `cpu`.
        """
        super().__init__(
            model=model,
            loss=loss,
            input_shape=input_shape,
            nb_classes=nb_classes,
            optimizer=optimizer,
            channels_first=channels_first,
            clip_values=clip_values,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
            device_type=device_type,
            ablation_type=ablation_type,
            ablation_size=ablation_size,
            threshold=threshold,
            logits=logits,
        )

    def _predict_classifier(self, x: np.ndarray, batch_size: int, training_mode: bool, **kwargs) -> np.ndarray:
        import torch  # lgtm [py/repeated-import]

        x = x.astype(ART_NUMPY_DTYPE)
        outputs = PyTorchClassifier.predict(self, x=x, batch_size=batch_size, training_mode=training_mode, **kwargs)

        if not self.logits:
            return np.asarray((outputs >= self.threshold))
        return np.asarray(
            (torch.nn.functional.softmax(torch.from_numpy(outputs), dim=1) >= self.threshold).type(torch.int)
        )

    def predict(
        self, x: np.ndarray, batch_size: int = 128, training_mode: bool = False, **kwargs
    ) -> np.ndarray:  # type: ignore
        """
        Perform prediction of the given classifier for a batch of inputs, taking an expectation over transformations.

        :param x: Input samples.
        :param batch_size: Batch size.
        :param training_mode: if to run the classifier in training mode
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        """
        return DeRandomizedSmoothingMixin.predict(self, x, batch_size=batch_size, training_mode=training_mode, **kwargs)

    def _fit_classifier(self, x: np.ndarray, y: np.ndarray, batch_size: int, nb_epochs: int, **kwargs) -> None:
        x = x.astype(ART_NUMPY_DTYPE)
        return PyTorchClassifier.fit(self, x, y, batch_size=batch_size, nb_epochs=nb_epochs, **kwargs)
