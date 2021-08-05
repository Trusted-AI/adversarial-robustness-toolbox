"""
Classifier API for abstracting classification models for testing purposes. Soon to be replaced by Estimator API.
Use `art.estimators.classification instead.
"""
from art.estimators.classification.scikitlearn import SklearnClassifier  # pragma: no_cover
from art.estimators.classification.scikitlearn import ScikitlearnClassifier  # pragma: no_cover
from art.estimators.classification.scikitlearn import ScikitlearnDecisionTreeClassifier  # pragma: no_cover
from art.estimators.classification.scikitlearn import ScikitlearnDecisionTreeRegressor  # pragma: no_cover
from art.estimators.classification.scikitlearn import ScikitlearnExtraTreeClassifier  # pragma: no_cover
from art.estimators.classification.scikitlearn import ScikitlearnAdaBoostClassifier  # pragma: no_cover
from art.estimators.classification.scikitlearn import ScikitlearnBaggingClassifier  # pragma: no_cover
from art.estimators.classification.scikitlearn import ScikitlearnExtraTreesClassifier  # pragma: no_cover
from art.estimators.classification.scikitlearn import ScikitlearnGradientBoostingClassifier  # pragma: no_cover
from art.estimators.classification.scikitlearn import ScikitlearnRandomForestClassifier  # pragma: no_cover
from art.estimators.classification.scikitlearn import ScikitlearnLogisticRegression  # pragma: no_cover
from art.estimators.classification.scikitlearn import ScikitlearnSVC  # pragma: no_cover
from art.estimators.classification.scikitlearn import ScikitlearnLinearSVC  # pragma: no_cover
