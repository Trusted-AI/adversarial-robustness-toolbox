"""
Module implementing multiple types of defences against adversarial attacks.
"""
from art.defences.train.adversarial_trainer import AdversarialTrainer

from art.defences.preprocess.feature_squeezing import FeatureSqueezing
from art.defences.preprocess.gaussian_augmentation import GaussianAugmentation
from art.defences.preprocess.jpeg_compression import JpegCompression
from art.defences.preprocess.label_smoothing import LabelSmoothing
from art.defences.preprocess.pixel_defend import PixelDefend
from art.defences.preprocess.preprocessor import Preprocessor
from art.defences.preprocess.spatial_smoothing import SpatialSmoothing
from art.defences.preprocess.thermometer_encoding import ThermometerEncoding
from art.defences.preprocess.variance_minimization import TotalVarMin
