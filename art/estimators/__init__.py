"""
This module contains the Estimator API.
"""

from art.estimators.estimator import (
    BaseEstimator,
    LossGradientsMixin,
    NeuralNetworkMixin,
    DecisionTreeMixin,
)

from art.estimators.keras import KerasEstimator
from art.estimators.pytorch import PyTorchEstimator
from art.estimators.scikitlearn import ScikitlearnEstimator
from art.estimators.tensorflow import TensorFlowV2Estimator

from art.estimators import certification
from art.estimators import classification
from art.estimators import generation
from art.estimators import object_detection
from art.estimators import poison_mitigation
from art.estimators import regression
from art.estimators import speech_recognition
