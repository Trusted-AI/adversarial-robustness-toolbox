"""
Module providing attribute inference attacks.
"""
from art.attacks.inference.attribute_inference.black_box import AttributeInferenceBlackBox
from art.attacks.inference.attribute_inference.baseline import AttributeInferenceBaseline
from art.attacks.inference.attribute_inference.white_box_decision_tree import AttributeInferenceWhiteBoxDecisionTree
from art.attacks.inference.attribute_inference.white_box_lifestyle_decision_tree import (
    AttributeInferenceWhiteBoxLifestyleDecisionTree,
)
from art.attacks.inference.attribute_inference.meminf_based import AttributeInferenceMembership
