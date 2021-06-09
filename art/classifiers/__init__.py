"""
Classifier API for applying all attacks. Use the :class:`.Classifier` wrapper to be able to apply an attack to a
existing model.
"""
import warnings

warnings.simplefilter("always", category=DeprecationWarning)
warnings.warn(
    "The module art.classifiers will be removed in ART 1.8.0 and replaced with art.estimators.classification",
    DeprecationWarning,
    stacklevel=2,
)
warnings.simplefilter("default", category=DeprecationWarning)
from art.estimators.classification.blackbox import BlackBoxClassifier
from art.estimators.classification.catboost import CatBoostARTClassifier
from art.estimators.classification.detector_classifier import DetectorClassifier
from art.estimators.classification.ensemble import EnsembleClassifier
from art.estimators.classification.GPy import GPyGaussianProcessClassifier
from art.estimators.classification.keras import KerasClassifier
from art.estimators.classification.lightgbm import LightGBMClassifier
from art.estimators.classification.mxnet import MXClassifier
from art.estimators.classification.pytorch import PyTorchClassifier
from art.estimators.classification.scikitlearn import SklearnClassifier
from art.estimators.classification.tensorflow import (
    TFClassifier,
    TensorFlowClassifier,
    TensorFlowV2Classifier,
)
from art.estimators.classification.xgboost import XGBoostClassifier
