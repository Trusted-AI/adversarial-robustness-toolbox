"""
Classifier API for applying all attacks. Use the :class:`.Classifier` wrapper to be able to apply an attack to a
preexisting model.
"""
from art.classifiers.classifier import Classifier
from art.classifiers.keras import KerasClassifier
from art.classifiers.mxnet import MXClassifier
from art.classifiers.pytorch import PyTorchClassifier
from art.classifiers.tensorflow import TFClassifier
from art.classifiers.ensemble import EnsembleClassifier
from art.classifiers.sklearn_logistic_regression import SklearnLogisticRegression
from art.classifiers.sklearn_svm import SklearnSVC
from art.classifiers.scikitklearn import ScikitlearnDecisionTreeClassifier
from art.classifiers.scikitklearn import ScikitlearnExtraTreeClassifier
from art.classifiers.scikitklearn import ScikitlearnAdaBoostClassifier
from art.classifiers.scikitklearn import ScikitlearnBaggingClassifier
from art.classifiers.scikitklearn import ScikitlearnExtraTreesClassifier
from art.classifiers.scikitklearn import ScikitlearnGradientBoostingClassifier
from art.classifiers.scikitklearn import ScikitlearnRandomForestClassifier
from art.classifiers.lightgbm import LightGBMClassifier
from art.classifiers.xgboost import XGBoostClassifier
from art.classifiers.catboost import CatBoostARTClassifier
