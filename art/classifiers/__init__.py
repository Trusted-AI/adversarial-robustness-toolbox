"""
Classifier API for applying all attacks. Use the :class:`.Classifier` wrapper to be able to apply an attack to a
existing model.
"""
# pylint: disable=C0413
import warnings  # pragma: no cover

warnings.simplefilter("always", category=DeprecationWarning)  # pragma: no cover
warnings.warn(  # pragma: no cover
    "The module art.classifiers will be removed in ART 1.8.0 and replaced with art.estimators.classification",
    DeprecationWarning,
    stacklevel=2,
)
warnings.simplefilter("default", category=DeprecationWarning)  # pragma: no cover
from art.estimators.classification.blackbox import BlackBoxClassifier  # noqa; pragma: no cover
from art.estimators.classification.catboost import CatBoostARTClassifier  # noqa; pragma: no cover
from art.estimators.classification.detector_classifier import DetectorClassifier  # noqa; pragma: no cover
from art.estimators.classification.ensemble import EnsembleClassifier  # noqa; pragma: no cover
from art.estimators.classification.GPy import GPyGaussianProcessClassifier  # noqa; pragma: no cover
from art.estimators.classification.keras import KerasClassifier  # noqa; pragma: no cover
from art.estimators.classification.lightgbm import LightGBMClassifier  # noqa; pragma: no cover
from art.estimators.classification.mxnet import MXClassifier  # noqa; pragma: no cover
from art.estimators.classification.pytorch import PyTorchClassifier  # noqa; pragma: no cover
from art.estimators.classification.scikitlearn import SklearnClassifier  # noqa; pragma: no cover
from art.estimators.classification.tensorflow import (  # noqa; pragma: no cover
    TFClassifier,
    TensorFlowClassifier,
    TensorFlowV2Classifier,
)
from art.estimators.classification.xgboost import XGBoostClassifier  # noqa; pragma: no cover
