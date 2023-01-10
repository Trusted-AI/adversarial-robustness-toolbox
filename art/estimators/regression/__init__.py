"""
This module implements all regressors in ART.
"""
from art.estimators.regression.regressor import RegressorMixin, Regressor

from art.estimators.regression.scikitlearn import ScikitlearnRegressor

from art.estimators.regression.keras import KerasRegressor

from art.estimators.regression.pytorch import PyTorchRegressor

from art.estimators.regression.blackbox import BlackBoxRegressor
