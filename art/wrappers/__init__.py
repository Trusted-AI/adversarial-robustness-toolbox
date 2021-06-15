"""
Module providing wrappers for :class:`.Classifier` instances to simulate different capacities and behaviours, like
black-box gradient estimation.
"""
# pylint: disable=C0413
import warnings

warnings.simplefilter("always", category=DeprecationWarning)
warnings.warn(
    "The module art.wrappers will be removed in ART 1.8.0 and replaced with tools in art.estimators and "
    "art.preprocessing.expectation_over_transformation",
    DeprecationWarning,
    stacklevel=2,
)
warnings.simplefilter("default", category=DeprecationWarning)
from art.wrappers.wrapper import ClassifierWrapper  # noqa
from art.wrappers.expectation import ExpectationOverTransformations  # noqa
from art.wrappers.query_efficient_bb import QueryEfficientBBGradientEstimation  # noqa
