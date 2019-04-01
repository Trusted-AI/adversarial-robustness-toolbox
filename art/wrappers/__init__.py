"""
Module providing wrappers for :class:`.Classifier` instances to simulate different capacities and behaviours, like
black-box gradient estimation.
"""
from art.wrappers.wrapper import ClassifierWrapper
from art.wrappers.expectation import ExpectationOverTransformations
from art.wrappers.query_efficient_bb import QueryEfficientBBGradientEstimation
