# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2026
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

    PREPROCESSING_TYPE = Optional[  # pylint: disable=invalid-name
        Union[
            Tuple[Union[int, float, np.ndarray], Union[int, float, np.ndarray]], Preprocessor, Tuple[Preprocessor, ...]
        ]
    ]

    from art.estimators.classification.blackbox import BlackBoxClassifier
    from art.estimators.classification.catboost import CatBoostARTClassifier
    from art.estimators.classification.classifier import (
        Classifier,
        ClassifierClassLossGradients,
        ClassifierDecisionTree,
        ClassifierLossGradients,
        ClassifierNeuralNetwork,
    )
    from art.estimators.classification.detector_classifier import DetectorClassifier
    from art.estimators.classification.ensemble import EnsembleClassifier
    from art.estimators.classification.GPy import GPyGaussianProcessClassifier
    from art.estimators.classification.keras import KerasClassifier
    from art.experimental.estimators.classification.jax import JaxClassifier
    from art.estimators.classification.lightgbm import LightGBMClassifier
    from art.estimators.classification.pytorch import PyTorchClassifier
    from art.estimators.classification.query_efficient_bb import QueryEfficientGradientEstimationClassifier
    from art.estimators.classification.scikitlearn import (
        ScikitlearnAdaBoostClassifier,
        ScikitlearnBaggingClassifier,
        ScikitlearnClassifier,
        ScikitlearnDecisionTreeClassifier,
        ScikitlearnDecisionTreeRegressor,
        ScikitlearnExtraTreeClassifier,
        ScikitlearnExtraTreesClassifier,
        ScikitlearnGradientBoostingClassifier,
        ScikitlearnLogisticRegression,
        ScikitlearnRandomForestClassifier,
        ScikitlearnSVC,
    )
    from art.estimators.classification.tensorflow import TensorFlowV2Classifier
    from art.estimators.classification.xgboost import XGBoostClassifier
    from art.estimators.certification.deep_z import PytorchDeepZ
    from art.estimators.certification.interval import PyTorchIBPClassifier
    from art.estimators.certification.derandomized_smoothing.ablators import BlockAblator, ColumnAblator
    from art.estimators.generation.tensorflow import TensorFlowV2Generator
    from art.estimators.object_detection.object_detector import ObjectDetector
    from art.estimators.object_detection.pytorch_object_detector import PyTorchObjectDetector
    from art.estimators.object_detection.pytorch_faster_rcnn import PyTorchFasterRCNN
    from art.estimators.object_detection.pytorch_yolo import PyTorchYolo
    from art.estimators.object_detection.tensorflow_v2_faster_rcnn import TensorFlowV2FasterRCNN
    from art.estimators.pytorch import PyTorchEstimator
    from art.estimators.keras import KerasEstimator
    from art.estimators.regression.scikitlearn import ScikitlearnRegressor
    from art.estimators.regression.pytorch import PyTorchRegressor
    from art.estimators.regression.keras import KerasRegressor
    from art.estimators.regression.blackbox import BlackBoxRegressor
    from art.estimators.speech_recognition.pytorch_deep_speech import PyTorchDeepSpeech
    from art.estimators.tensorflow import TensorFlowV2Estimator

    CLASSIFIER_LOSS_GRADIENTS_TYPE = Union[  # pylint: disable=invalid-name
        ClassifierLossGradients,
        EnsembleClassifier,
        GPyGaussianProcessClassifier,
        KerasClassifier,
        JaxClassifier,
        PyTorchClassifier,
        ScikitlearnLogisticRegression,
        ScikitlearnSVC,
        TensorFlowV2Classifier,
        QueryEfficientGradientEstimationClassifier,
    ]

    CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE = Union[  # pylint: disable=invalid-name
        ClassifierClassLossGradients,
        EnsembleClassifier,
        GPyGaussianProcessClassifier,
        KerasClassifier,
        PyTorchClassifier,
        ScikitlearnLogisticRegression,
        ScikitlearnSVC,
        TensorFlowV2Classifier,
    ]

    CLASSIFIER_NEURALNETWORK_TYPE = Union[  # pylint: disable=invalid-name
        ClassifierNeuralNetwork,
        DetectorClassifier,
        EnsembleClassifier,
        KerasClassifier,
        PyTorchClassifier,
        TensorFlowV2Classifier,
    ]

    CLASSIFIER_DECISION_TREE_TYPE = Union[  # pylint: disable=invalid-name
        ClassifierDecisionTree,
        LightGBMClassifier,
        ScikitlearnDecisionTreeClassifier,
        ScikitlearnExtraTreesClassifier,
        ScikitlearnGradientBoostingClassifier,
        ScikitlearnRandomForestClassifier,
        XGBoostClassifier,
    ]

    CLASSIFIER_TYPE = Union[  # pylint: disable=invalid-name
        Classifier,
        BlackBoxClassifier,
        CatBoostARTClassifier,
        DetectorClassifier,
        EnsembleClassifier,
        GPyGaussianProcessClassifier,
        KerasClassifier,
        JaxClassifier,
        LightGBMClassifier,
        PyTorchClassifier,
        ScikitlearnClassifier,
        ScikitlearnDecisionTreeClassifier,
        ScikitlearnExtraTreeClassifier,
        ScikitlearnAdaBoostClassifier,
        ScikitlearnBaggingClassifier,
        ScikitlearnExtraTreesClassifier,
        ScikitlearnGradientBoostingClassifier,
        ScikitlearnRandomForestClassifier,
        ScikitlearnLogisticRegression,
        ScikitlearnSVC,
        TensorFlowV2Classifier,
        XGBoostClassifier,
        CLASSIFIER_NEURALNETWORK_TYPE,
    ]

    GENERATOR_TYPE = Union[TensorFlowV2Generator]  # pylint: disable=invalid-name

    REGRESSOR_TYPE = Union[  # pylint: disable=invalid-name
        ScikitlearnRegressor, ScikitlearnDecisionTreeRegressor, PyTorchRegressor, KerasRegressor, BlackBoxRegressor
    ]

    OBJECT_DETECTOR_TYPE = Union[  # pylint: disable=invalid-name
        ObjectDetector,
        PyTorchObjectDetector,
        PyTorchFasterRCNN,
        PyTorchYolo,
        TensorFlowV2FasterRCNN,
    ]

    SPEECH_RECOGNIZER_TYPE = Union[PyTorchDeepSpeech,]  # pylint: disable=invalid-name

    PYTORCH_ESTIMATOR_TYPE = Union[  # pylint: disable=invalid-name
        PyTorchClassifier,
        PyTorchDeepSpeech,
        PyTorchEstimator,
        PyTorchObjectDetector,
        PyTorchFasterRCNN,
        PyTorchYolo,
        PyTorchRegressor,
    ]

    PYTORCH_OBJECT_DETECTOR_TYPE = Union[PyTorchObjectDetector]  # pylint: disable=invalid-name

    KERAS_ESTIMATOR_TYPE = Union[  # pylint: disable=invalid-name
        KerasClassifier,
        KerasEstimator,
        KerasRegressor,
    ]

    TENSORFLOWV2_ESTIMATOR_TYPE = Union[  # pylint: disable=invalid-name
        TensorFlowV2Classifier,
        TensorFlowV2Estimator,
        TensorFlowV2FasterRCNN,
    ]

    ESTIMATOR_TYPE = Union[  # pylint: disable=invalid-name
        CLASSIFIER_TYPE,
        REGRESSOR_TYPE,
        OBJECT_DETECTOR_TYPE,
        SPEECH_RECOGNIZER_TYPE,
        PYTORCH_ESTIMATOR_TYPE,
        KERAS_ESTIMATOR_TYPE,
        TENSORFLOWV2_ESTIMATOR_TYPE,
    ]

    CLONABLE = Union[  # pylint: disable=invalid-name
        ScikitlearnClassifier,
        PyTorchClassifier,
        TensorFlowV2Classifier,
        KerasEstimator,
        PyTorchRegressor,
        ScikitlearnRegressor,
        XGBoostClassifier,
    ]

    ABLATOR_TYPE = Union[BlockAblator, ColumnAblator]  # pylint: disable=invalid-name

    CERTIFIER_TYPE = Union[PytorchDeepZ]  # pylint: disable=invalid-name
    IBP_CERTIFIER_TYPE = Union[PyTorchIBPClassifier]  # pylint: disable=invalid-name
