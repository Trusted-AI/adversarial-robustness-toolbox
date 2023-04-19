from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import List, Optional, Tuple, Union, Any, TYPE_CHECKING
import random

import numpy as np
from tqdm import tqdm

from art.estimators.classification.pytorch import PyTorchClassifier
from art.estimators.certification.smoothed_vision_transformers.smooth_vit import ColumnAblator


class PyTorchSmoothedViT(PyTorchClassifier):
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
        Create a smoothed ViT classifier.

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
            logits=logits,
        )

        self.ablation_type = ablation_type
        self.ablation_size = ablation_size,
        self.threshold = threshold