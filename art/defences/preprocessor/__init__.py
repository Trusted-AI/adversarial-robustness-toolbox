"""
Module implementing preprocessing defences against adversarial attacks.
"""
from art.defences.preprocessor.cutmix.cutmix import CutMix
from art.defences.preprocessor.cutmix.cutmix_pytorch import CutMixPyTorch
from art.defences.preprocessor.cutmix.cutmix_tensorflow import CutMixTensorFlowV2
from art.defences.preprocessor.cutout.cutout import Cutout
from art.defences.preprocessor.cutout.cutout_pytorch import CutoutPyTorch
from art.defences.preprocessor.cutout.cutout_tensorflow import CutoutTensorFlowV2
from art.defences.preprocessor.feature_squeezing import FeatureSqueezing
from art.defences.preprocessor.gaussian_augmentation import GaussianAugmentation
from art.defences.preprocessor.inverse_gan import DefenseGAN, InverseGAN
from art.defences.preprocessor.jpeg_compression import JpegCompression
from art.defences.preprocessor.label_smoothing import LabelSmoothing
from art.defences.preprocessor.mixup.mixup import Mixup
from art.defences.preprocessor.mixup.mixup_pytorch import MixupPyTorch
from art.defences.preprocessor.mixup.mixup_tensorflow import MixupTensorFlowV2
from art.defences.preprocessor.mp3_compression import Mp3Compression
from art.defences.preprocessor.mp3_compression_pytorch import Mp3CompressionPyTorch
from art.defences.preprocessor.pixel_defend import PixelDefend
from art.defences.preprocessor.preprocessor import Preprocessor
from art.defences.preprocessor.resample import Resample
from art.defences.preprocessor.spatial_smoothing import SpatialSmoothing
from art.defences.preprocessor.spatial_smoothing_pytorch import SpatialSmoothingPyTorch
from art.defences.preprocessor.spatial_smoothing_tensorflow import SpatialSmoothingTensorFlowV2
from art.defences.preprocessor.thermometer_encoding import ThermometerEncoding
from art.defences.preprocessor.variance_minimization import TotalVarMin
from art.defences.preprocessor.video_compression import VideoCompression
from art.defences.preprocessor.video_compression_pytorch import VideoCompressionPyTorch
