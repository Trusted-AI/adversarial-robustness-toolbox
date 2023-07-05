# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2023
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
This module implements the ObjectSeeker certifiably robust defense.

| Paper link: https://arxiv.org/abs/2202.01811
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

from tqdm import tqdm
import numpy as np

from art.config import ART_NUMPY_DTYPE
from art.estimators.object_detection import ObjectDetectorMixin
from art.estimators.pytorch import PyTorchEstimator
from art.estimators.certification.randomized_smoothing.randomized_smoothing import RandomizedSmoothingMixin
from art.utils import intersection_over_union, non_maximum_supression

if TYPE_CHECKING:
    # pylint: disable=C0412
    import torch

    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor import Preprocessor
    from art.defences.postprocessor import Postprocessor

logger = logging.getLogger(__name__)

class PyTorchObjectSeeker(ObjectDetectorMixin, PyTorchEstimator):
    """
    Implementation of the ObjectSeeker certifiable robust defense applied to object detection models.

    | Paper link: https://arxiv.org/abs/2202.01811
    """

    estimator_params = PyTorchEstimator.estimator_params + ["input_shape", "optimizer", "attack_losses"]

    def __init__(
        self,
        model: "torch.nn.Module",
        input_shape: Tuple[int, ...] = (3, 416, 416),
        optimizer: Optional["torch.optim.Optimizer"] = None,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        channels_first: Optional[bool] = True,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: "PREPROCESSING_TYPE" = None,
        attack_losses: Tuple[str, ...] = (
            "loss_classifier",
            "loss_box_reg",
            "loss_objectness",
            "loss_rpn_box_reg",
        ),
        device_type: str = "gpu",
    ):
        pass

