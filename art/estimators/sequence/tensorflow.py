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
This module implements task-specific estimators for automatic speech recognition in TensorFlow.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import os
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np

from art.config import ART_DATA_PATH
from art.estimators.sequence.sequence import SequenceNetworkMixin
from art.estimators.tensorflow import TensorFlowV2Estimator
from art.utils import get_file, make_directory

if TYPE_CHECKING:
    from art.config import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor.preprocessor import Preprocessor
    from art.defences.postprocessor.postprocessor import Postprocessor

logger = logging.getLogger(__name__)


class LingvoAsr(SequenceNetworkMixin, TensorFlowV2Estimator):
    """
    This class implements the task-specific Lingvo ASR model of Qin et al. (2019).

    | Paper link: http://proceedings.mlr.press/v97/qin19a.html
    |             https://arxiv.org/abs/1902.08295
    """

    _LINGVO_PATH = os.path.join(ART_DATA_PATH, "lingvo/")
    _LINGVO_PARAMS_URI = (
        "https://raw.githubusercontent.com/tensorflow/lingvo/"
        "3c5ef88b8a9407124afe045a8e5048a9c5013acd/lingvo/tasks/asr/params/librispeech.py"
    )

    def __init__(
        self,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        channels_first: Optional[bool] = None,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: "PREPROCESSING_TYPE" = None,
        device_type: str = "cpu",
    ):
        """
        Initialization.

        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and
               maximum values allowed for features. If floats are provided, these will be used as the range of all
               features. If arrays are provided, each value will be considered the bound for a feature, thus
               the shape of clip values needs to match the total number of features.
        :param channels_first: Set channels first or last.
        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
        :param preprocessing: Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :param device_type: Type of device to be used for model and tensors, if `cpu` run on CPU, if `gpu` run on GPU
                            if available otherwise run on CPU.
        """
        # Super initialization
        super().__init__(
            clip_values=clip_values,
            channels_first=channels_first,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )
        self.device_type = device_type
        if self.preprocessing is not None:
            raise ValueError("This estimator does not support `preprocessing`.")
        if self.preprocessing_defences is not None:
            raise ValueError("This estimator does not support `preprocessing_defences`.")
        if self.postprocessing_defences is not None:
            raise ValueError("This estimator does not support `postprocessing_defences`.")
        if self.device_type != "cpu":
            raise ValueError("This estimator does not yet support running on a GPU.")
        pass

    def _check_and_download_params(self) -> None:
        """Check and download the `params/librispeech.py` file from the official Lingvo repository."""
        params_dir = os.path.join(self._LINGVO_PATH, "asr/")
        params_base = "librispeech.py"
        if not os.path.isdir(params_dir):
            make_directory(params_dir)
        if not os.path.isfile(os.path.join(params_dir, params_base)):
            get_file(params_base, self._LINGVO_PARAMS_URI, path=params_dir)

    def loss_gradient(self, x, y, **kwargs):
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Samples.
        :type x: Format as expected by the `model`
        :param y: Target values.
        :type y: Format as expected by the `model`
        :return: Loss gradients w.r.t. `x` in the same format as `x`.
        :rtype: Format as expected by the `model`
        """
        raise NotImplementedError

    def set_learning_phase(self) -> None:
        raise NotImplementedError

    def get_activations(self) -> np.ndarray:
        raise NotImplementedError
