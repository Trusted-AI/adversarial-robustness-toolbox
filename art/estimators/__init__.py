"""
This module contains the Estimator API.
"""

from art.estimators import (
    certification,
    classification,
    generation,
    object_detection,
    poison_mitigation,
    regression,
    speech_recognition,
)
from art.estimators.estimator import (
    BaseEstimator,
    DecisionTreeMixin,
    LossGradientsMixin,
    NeuralNetworkMixin,
)
from art.estimators.keras import KerasEstimator
from art.estimators.pytorch import PyTorchEstimator
from art.estimators.scikitlearn import ScikitlearnEstimator
from art.estimators.tensorflow import TensorFlowV2Estimator
