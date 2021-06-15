"""
Classifier API for applying all attacks. Use the :class:`.Classifier` wrapper to be able to apply an attack to a
existing model.
"""
# pylint: disable=C0413
import warnings

warnings.simplefilter("always", category=DeprecationWarning)
warnings.warn(
    "The module art.classifiers will be removed in ART 1.8.0 and replaced with art.estimators.classification",
    DeprecationWarning,
    stacklevel=2,
)
warnings.simplefilter("default", category=DeprecationWarning)
from art.estimators.classification.blackbox import BlackBoxClassifier  # noqa
from art.estimators.classification.catboost import CatBoostARTClassifier  # noqa
from art.estimators.classification.detector_classifier import DetectorClassifier  # noqa
from art.estimators.classification.ensemble import EnsembleClassifier  # noqa
from art.estimators.classification.GPy import GPyGaussianProcessClassifier  # noqa
from art.estimators.classification.keras import KerasClassifier  # noqa
from art.estimators.classification.lightgbm import LightGBMClassifier  # noqa
from art.estimators.classification.mxnet import MXClassifier  # noqa
from art.estimators.classification.pytorch import PyTorchClassifier  # noqa
from art.estimators.classification.scikitlearn import SklearnClassifier  # noqa
from art.estimators.classification.tensorflow import (  # noqa
    TFClassifier,
    TensorFlowClassifier,
    TensorFlowV2Classifier,
)
from art.estimators.classification.xgboost import XGBoostClassifier  # noqa
