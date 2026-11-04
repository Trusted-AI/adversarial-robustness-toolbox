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
This module defines type aliases used across ART.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple, Union

import numpy as np

# ------------------------------------------------------------------------------------------------- CONSTANTS AND TYPES

DATASET_TYPE = Tuple[  # pylint: disable=invalid-name
    Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], float, float
]
CLIP_VALUES_TYPE = Tuple[Union[int, float, np.ndarray], Union[int, float, np.ndarray]]  # pylint: disable=invalid-name

if TYPE_CHECKING:
    from art.defences.preprocessor.preprocessor import Preprocessor
    from art.estimators.estimator import BaseEstimator
    from art.estimators.keras import KerasEstimator
    from art.estimators.pytorch import PyTorchEstimator
    from art.estimators.scikitlearn import ScikitlearnEstimator
    from art.estimators.tensorflow import TensorFlowV2Estimator
    from art.estimators.classification.classifier import (
        Classifier,
        ClassifierClassLossGradients,
        ClassifierDecisionTree,
        ClassifierLossGradients,
        ClassifierNeuralNetwork,
    )
    from art.estimators.object_detection.object_detector import ObjectDetector
    from art.estimators.regression.regressor import Regressor
    from art.estimators.speech_recognition.speech_recognizer import SpeechRecognizerMixin
    from art.estimators.certification.derandomized_smoothing.ablators import BlockAblator, ColumnAblator

    PREPROCESSING_TYPE = Optional[  # pylint: disable=invalid-name
        Union[
            Tuple[Union[int, float, np.ndarray], Union[int, float, np.ndarray]], "Preprocessor", Tuple["Preprocessor", ...]
        ]
    ]

    CLASSIFIER_LOSS_GRADIENTS_TYPE = Union["ClassifierLossGradients", "Classifier"]  # pylint: disable=invalid-name
    CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE = Union["ClassifierClassLossGradients", "Classifier"]  # pylint: disable=invalid-name
    CLASSIFIER_NEURALNETWORK_TYPE = Union["ClassifierNeuralNetwork", "Classifier"]  # pylint: disable=invalid-name
    CLASSIFIER_DECISION_TREE_TYPE = Union["ClassifierDecisionTree", "Classifier"]  # pylint: disable=invalid-name
    CLASSIFIER_TYPE = Union["Classifier", "BaseEstimator"]  # pylint: disable=invalid-name

    GENERATOR_TYPE = "BaseEstimator"  # pylint: disable=invalid-name
    REGRESSOR_TYPE = Union["Regressor", "BaseEstimator"]  # pylint: disable=invalid-name
    OBJECT_DETECTOR_TYPE = Union["ObjectDetector", "BaseEstimator"]  # pylint: disable=invalid-name
    SPEECH_RECOGNIZER_TYPE = Union["SpeechRecognizerMixin", "BaseEstimator"]  # pylint: disable=invalid-name
    PYTORCH_ESTIMATOR_TYPE = Union["PyTorchEstimator", "BaseEstimator"]  # pylint: disable=invalid-name
    PYTORCH_OBJECT_DETECTOR_TYPE = Union["ObjectDetector", "BaseEstimator"]  # pylint: disable=invalid-name
    KERAS_ESTIMATOR_TYPE = Union["KerasEstimator", "BaseEstimator"]  # pylint: disable=invalid-name
    TENSORFLOWV2_ESTIMATOR_TYPE = Union["TensorFlowV2Estimator", "BaseEstimator"]  # pylint: disable=invalid-name
    ESTIMATOR_TYPE = "BaseEstimator"  # pylint: disable=invalid-name
    CLONABLE = Union["ScikitlearnEstimator", "PyTorchEstimator", "TensorFlowV2Estimator", "KerasEstimator", "BaseEstimator"]  # pylint: disable=invalid-name

    ABLATOR_TYPE = Union["BlockAblator", "ColumnAblator"]  # pylint: disable=invalid-name
    CERTIFIER_TYPE = "BaseEstimator"  # pylint: disable=invalid-name
    IBP_CERTIFIER_TYPE = "BaseEstimator"  # pylint: disable=invalid-name
