"""
Module implementing preprocessing defences against adversarial attacks.
"""
from art.defences.preprocessor.feature_squeezing import FeatureSqueezing
from art.defences.preprocessor.gaussian_augmentation import GaussianAugmentation
from art.defences.preprocessor.jpeg_compression import JpegCompression
from art.defences.preprocessor.label_smoothing import LabelSmoothing
from art.defences.preprocessor.pixel_defend import PixelDefend
from art.defences.preprocessor.preprocessor import Preprocessor
from art.defences.preprocessor.spatial_smoothing import SpatialSmoothing
from art.defences.preprocessor.thermometer_encoding import ThermometerEncoding
from art.defences.preprocessor.variance_minimization import TotalVarMin
