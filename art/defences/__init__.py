"""
Module implementing multiple types of defences against adversarial attacks.
"""
from art.defences.adversarial_trainer import AdversarialTrainer
from art.defences.feature_squeezing import FeatureSqueezing
from art.defences.gaussian_augmentation import GaussianAugmentation
from art.defences.label_smoothing import LabelSmoothing
from art.defences.spatial_smoothing import SpatialSmoothing
