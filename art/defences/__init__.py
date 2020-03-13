"""
Module implementing multiple types of defences against adversarial attacks.
"""
from art.defences.postprocessor.class_labels import ClassLabels
from art.defences.postprocessor.gaussian_noise import GaussianNoise
from art.defences.postprocessor.high_confidence import HighConfidence
from art.defences.postprocessor.postprocessor import Postprocessor
from art.defences.postprocessor.reverse_sigmoid import ReverseSigmoid
from art.defences.postprocessor.rounded import Rounded

from art.defences.preprocessor.feature_squeezing import FeatureSqueezing
from art.defences.preprocessor.gaussian_augmentation import GaussianAugmentation
from art.defences.preprocessor.jpeg_compression import JpegCompression
from art.defences.preprocessor.label_smoothing import LabelSmoothing
from art.defences.preprocessor.pixel_defend import PixelDefend
from art.defences.preprocessor.preprocessor import Preprocessor
from art.defences.preprocessor.spatial_smoothing import SpatialSmoothing
from art.defences.preprocessor.thermometer_encoding import ThermometerEncoding
from art.defences.preprocessor.variance_minimization import TotalVarMin

from art.defences.trainer.adversarial_trainer import AdversarialTrainer
from art.defences.trainer.adversarial_trainer_madry_pgd import AdversarialTrainerMadryPGD
from art.defences.trainer.trainer import Trainer

from art.defences.transformer.defensive_distillation import DefensiveDistillation
from art.defences.transformer.transformer import Transformer
