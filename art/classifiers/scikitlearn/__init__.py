"""
Classifier API for abstracting classification models for testing purposes. Soon to be replaced by Estimator API.
Use `art.estimators.classification instead.
"""
from art.estimators.classification.scikitlearn import SklearnClassifier  # pragma: no cover
from art.estimators.classification.scikitlearn import ScikitlearnClassifier  # pragma: no cover
from art.estimators.classification.scikitlearn import ScikitlearnDecisionTreeClassifier  # pragma: no cover
from art.estimators.classification.scikitlearn import ScikitlearnDecisionTreeRegressor  # pragma: no cover
from art.estimators.classification.scikitlearn import ScikitlearnExtraTreeClassifier  # pragma: no cover
from art.estimators.classification.scikitlearn import ScikitlearnAdaBoostClassifier  # pragma: no cover
from art.estimators.classification.scikitlearn import ScikitlearnBaggingClassifier  # pragma: no cover
from art.estimators.classification.scikitlearn import ScikitlearnExtraTreesClassifier  # pragma: no cover
from art.estimators.classification.scikitlearn import ScikitlearnGradientBoostingClassifier  # pragma: no cover
from art.estimators.classification.scikitlearn import ScikitlearnRandomForestClassifier  # pragma: no cover
from art.estimators.classification.scikitlearn import ScikitlearnLogisticRegression  # pragma: no cover
from art.estimators.classification.scikitlearn import ScikitlearnSVC  # pragma: no cover
from art.estimators.classification.scikitlearn import ScikitlearnLinearSVC  # pragma: no cover
