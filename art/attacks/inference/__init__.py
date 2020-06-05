"""
Module providing extraction attacks under a common interface.
"""
from art.attacks.inference.model_inversion import MIFace
from art.attacks.inference.attribute_inference import (
    AttributeInferenceBlackBox,
    AttributeInferenceWhiteBoxLifestyleDecisionTree,
    AttributeInferenceWhiteBoxDecisionTree,
)
