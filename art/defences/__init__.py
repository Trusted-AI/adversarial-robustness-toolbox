"""
Module implementing multiple types of defences against adversarial attacks.
"""
from art.defences.adversarial_trainer import AdversarialTrainer, StaticAdversarialTrainer
from art.defences.feature_squeezing import FeatureSqueezing
from art.defences.gaussian_augmentation import GaussianAugmentation
from art.defences.jpeg_compression import JpegCompression
from art.defences.label_smoothing import LabelSmoothing
from art.defences.spatial_smoothing import SpatialSmoothing
from art.defences.thermometer_encoding import ThermometerEncoding
from art.defences.variance_minimization import TotalVarMin
