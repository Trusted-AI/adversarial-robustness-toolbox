"""
Module providing wrappers for :class:`.Classifier` instances to simulate different capacities and behaviours, like
black-box gradient estimation.
"""
from art.wrappers.wrapper import ClassifierWrapper
from art.wrappers.expectation import ExpectationOverTransformations
from art.wrappers.randomized_smoothing import RandomizedSmoothing
from art.wrappers.query_efficient_bb import QueryEfficientBBGradientEstimation
from art.wrappers.output_add_random_noise import OutputRandomNoise
from art.wrappers.output_class_labels import OutputClassLabels
from art.wrappers.output_high_confidence import OutputHighConfidence
from art.wrappers.output_reverse_sigmoid import OutputReverseSigmoid
from art.wrappers.output_rounded import OutputRounded
