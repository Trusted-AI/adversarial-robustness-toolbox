"""
Classifier API for abstracting classification models for testing purposes. Soon to be replaced by Estimator API.
Use `art.estimators.classification instead.
"""
from art.estimators.classification.scikitlearn import SklearnClassifier
from art.estimators.classification.scikitlearn import ScikitlearnClassifier
from art.estimators.classification.scikitlearn import ScikitlearnDecisionTreeClassifier
from art.estimators.classification.scikitlearn import ScikitlearnDecisionTreeRegressor
from art.estimators.classification.scikitlearn import ScikitlearnExtraTreeClassifier
from art.estimators.classification.scikitlearn import ScikitlearnAdaBoostClassifier
from art.estimators.classification.scikitlearn import ScikitlearnBaggingClassifier
from art.estimators.classification.scikitlearn import ScikitlearnExtraTreesClassifier
from art.estimators.classification.scikitlearn import ScikitlearnGradientBoostingClassifier
from art.estimators.classification.scikitlearn import ScikitlearnRandomForestClassifier
from art.estimators.classification.scikitlearn import ScikitlearnLogisticRegression
from art.estimators.classification.scikitlearn import ScikitlearnSVC
from art.estimators.classification.scikitlearn import ScikitlearnLinearSVC
