# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
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
This module loads and provides configuration parameters for ART.
"""
import json
import logging
import os
from typing import Tuple, Union, Optional

import numpy as np

from art.estimators.classification.classifier import (
    Classifier,
    ClassifierLossGradients,
    ClassifierClassLossGradients,
    ClassifierNeuralNetwork,
    ClassifierDecisionTree,
)
from art.estimators.classification.blackbox import BlackBoxClassifier
from art.estimators.classification.catboost import CatBoostARTClassifier
from art.estimators.classification.detector_classifier import DetectorClassifier
from art.estimators.classification.ensemble import EnsembleClassifier
from art.estimators.classification.GPy import GPyGaussianProcessClassifier
from art.estimators.classification.keras import KerasClassifier
from art.estimators.classification.lightgbm import LightGBMClassifier
from art.estimators.classification.mxnet import MXClassifier
from art.estimators.classification.pytorch import PyTorchClassifier
from art.estimators.classification.scikitlearn import (
    ScikitlearnClassifier,
    ScikitlearnDecisionTreeClassifier,
    ScikitlearnDecisionTreeRegressor,
    ScikitlearnExtraTreeClassifier,
    ScikitlearnAdaBoostClassifier,
    ScikitlearnBaggingClassifier,
    ScikitlearnExtraTreesClassifier,
    ScikitlearnGradientBoostingClassifier,
    ScikitlearnRandomForestClassifier,
    ScikitlearnLogisticRegression,
    ScikitlearnSVC,
)
from art.estimators.classification.tensorflow import TensorFlowClassifier, TensorFlowV2Classifier
from art.estimators.classification.xgboost import XGBoostClassifier

from art.estimators.object_detection.object_detector import ObjectDetector
from art.estimators.object_detection.pytorch_faster_rcnn import PyTorchFasterRCNN
from art.estimators.object_detection.tensorflow_faster_rcnn import TensorFlowFasterRCNN


logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------------------------- CONSTANTS AND TYPES


ART_NUMPY_DTYPE = np.float32
DATASET_TYPE = Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], float, float]
CLIP_VALUES_TYPE = Tuple[Union[int, float, np.ndarray], Union[int, float, np.ndarray]]
PREPROCESSING_TYPE = Optional[Tuple[Union[int, float, np.ndarray], Union[int, float, np.ndarray]]]

CLASSIFIER_TYPE = Union[
    Classifier,
    BlackBoxClassifier,
    CatBoostARTClassifier,
    DetectorClassifier,
    EnsembleClassifier,
    GPyGaussianProcessClassifier,
    KerasClassifier,
    LightGBMClassifier,
    MXClassifier,
    PyTorchClassifier,
    ScikitlearnClassifier,
    ScikitlearnDecisionTreeClassifier,
    ScikitlearnDecisionTreeRegressor,
    ScikitlearnExtraTreeClassifier,
    ScikitlearnAdaBoostClassifier,
    ScikitlearnBaggingClassifier,
    ScikitlearnExtraTreesClassifier,
    ScikitlearnGradientBoostingClassifier,
    ScikitlearnRandomForestClassifier,
    ScikitlearnLogisticRegression,
    ScikitlearnSVC,
    TensorFlowClassifier,
    TensorFlowV2Classifier,
    XGBoostClassifier,
]

CLASSIFIER_LOSS_GRADIENTS_TYPE = Union[
    ClassifierLossGradients,
    EnsembleClassifier,
    GPyGaussianProcessClassifier,
    KerasClassifier,
    MXClassifier,
    PyTorchClassifier,
    ScikitlearnLogisticRegression,
    ScikitlearnSVC,
    TensorFlowClassifier,
    TensorFlowV2Classifier,
]


CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE = Union[
    ClassifierClassLossGradients,
    EnsembleClassifier,
    GPyGaussianProcessClassifier,
    KerasClassifier,
    MXClassifier,
    PyTorchClassifier,
    ScikitlearnLogisticRegression,
    ScikitlearnSVC,
    TensorFlowClassifier,
    TensorFlowV2Classifier,
]

CLASSIFIER_NEURALNETWORK_TYPE = Union[
    ClassifierNeuralNetwork,
    DetectorClassifier,
    EnsembleClassifier,
    KerasClassifier,
    MXClassifier,
    PyTorchClassifier,
    TensorFlowClassifier,
    TensorFlowV2Classifier,
]

CLASSIFIER_DECISION_TREE_TYPE = Union[
    ClassifierDecisionTree,
    LightGBMClassifier,
    ScikitlearnDecisionTreeClassifier,
    ScikitlearnDecisionTreeRegressor,
    ScikitlearnExtraTreesClassifier,
    ScikitlearnGradientBoostingClassifier,
    ScikitlearnRandomForestClassifier,
    XGBoostClassifier,
]

OBJECT_DETECTOR_TYPE = Union[
    ObjectDetector, PyTorchFasterRCNN, TensorFlowFasterRCNN,
]

# --------------------------------------------------------------------------------------------- DEFAULT PACKAGE CONFIGS

_folder = os.path.expanduser("~")
if not os.access(_folder, os.W_OK):
    _folder = "/tmp"
_folder = os.path.join(_folder, ".art")

# Load data from configuration file if it exists. Otherwise create one.
_config_path = os.path.expanduser(os.path.join(_folder, "config.json"))
if os.path.exists(_config_path):
    try:
        with open(_config_path) as f:
            _config = json.load(f)

            # Since renaming this variable we must update existing config files
            if "DATA_PATH" in _config:
                _config["ART_DATA_PATH"] = _config.pop("DATA_PATH")
                try:
                    with open(_config_path, "w") as f:
                        f.write(json.dumps(_config, indent=4))
                except IOError:
                    logger.warning("Unable to update configuration file", exc_info=True)

    except ValueError:
        _config = {}

if not os.path.exists(_folder):
    try:
        os.makedirs(_folder)
    except OSError:
        logger.warning("Unable to create folder for configuration file.", exc_info=True)

if not os.path.exists(_config_path):
    # Generate default config
    _config = {"ART_DATA_PATH": os.path.join(_folder, "data")}

    try:
        with open(_config_path, "w") as f:
            f.write(json.dumps(_config, indent=4))
    except IOError:
        logger.warning("Unable to create configuration file", exc_info=True)

if not os.path.exists(_config["ART_DATA_PATH"]):
    try:
        os.makedirs(_config["ART_DATA_PATH"])
    except OSError:
        logger.warning("Unable to create folder for ART_DATA_PATH dir.", exc_info=True)

if "ART_DATA_PATH" in _config:
    ART_DATA_PATH = _config["ART_DATA_PATH"]
