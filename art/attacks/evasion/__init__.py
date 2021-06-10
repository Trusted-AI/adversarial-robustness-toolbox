"""
Module providing evasion attacks under a common interface.
"""
from art.attacks.evasion.adversarial_patch.adversarial_patch import AdversarialPatch
from art.attacks.evasion.adversarial_patch.adversarial_patch_numpy import AdversarialPatchNumpy
from art.attacks.evasion.adversarial_patch.adversarial_patch_tensorflow import AdversarialPatchTensorFlowV2
from art.attacks.evasion.adversarial_patch.adversarial_patch_pytorch import AdversarialPatchPyTorch
from art.attacks.evasion.adversarial_asr import CarliniWagnerASR
from art.attacks.evasion.auto_attack import AutoAttack
from art.attacks.evasion.auto_projected_gradient_descent import AutoProjectedGradientDescent
from art.attacks.evasion.brendel_bethge import BrendelBethgeAttack
from art.attacks.evasion.boundary import BoundaryAttack
from art.attacks.evasion.carlini import CarliniL2Method, CarliniLInfMethod, CarliniL0Method
from art.attacks.evasion.decision_tree_attack import DecisionTreeAttack
from art.attacks.evasion.deepfool import DeepFool
from art.attacks.evasion.dpatch import DPatch
from art.attacks.evasion.dpatch_robust import RobustDPatch
from art.attacks.evasion.elastic_net import ElasticNet
from art.attacks.evasion.fast_gradient import FastGradientMethod
from art.attacks.evasion.frame_saliency import FrameSaliencyAttack
from art.attacks.evasion.feature_adversaries.feature_adversaries_numpy import FeatureAdversariesNumpy
from art.attacks.evasion.feature_adversaries.feature_adversaries_pytorch import FeatureAdversariesPyTorch
from art.attacks.evasion.feature_adversaries.feature_adversaries_tensorflow import FeatureAdversariesTensorFlowV2
from art.attacks.evasion.geometric_decision_based_attack import GeoDA
from art.attacks.evasion.hclu import HighConfidenceLowUncertainty
from art.attacks.evasion.hop_skip_jump import HopSkipJump
from art.attacks.evasion.imperceptible_asr.imperceptible_asr import ImperceptibleASR
from art.attacks.evasion.imperceptible_asr.imperceptible_asr_pytorch import ImperceptibleASRPyTorch
from art.attacks.evasion.iterative_method import BasicIterativeMethod
from art.attacks.evasion.lowprofool import LowProFool
from art.attacks.evasion.newtonfool import NewtonFool
from art.attacks.evasion.pe_malware_attack import MalwareGDTensorFlow
from art.attacks.evasion.pixel_threshold import PixelAttack
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent_numpy import (
    ProjectedGradientDescentNumpy,
)
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent_pytorch import (
    ProjectedGradientDescentPyTorch,
)
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent_tensorflow_v2 import (
    ProjectedGradientDescentTensorFlowV2,
)
from art.attacks.evasion.over_the_air_flickering.over_the_air_flickering_pytorch import OverTheAirFlickeringPyTorch
from art.attacks.evasion.saliency_map import SaliencyMapMethod
from art.attacks.evasion.shadow_attack import ShadowAttack
from art.attacks.evasion.shapeshifter import ShapeShifter
from art.attacks.evasion.simba import SimBA
from art.attacks.evasion.spatial_transformation import SpatialTransformation
from art.attacks.evasion.square_attack import SquareAttack
from art.attacks.evasion.pixel_threshold import ThresholdAttack
from art.attacks.evasion.universal_perturbation import UniversalPerturbation
from art.attacks.evasion.targeted_universal_perturbation import TargetedUniversalPerturbation
from art.attacks.evasion.virtual_adversarial import VirtualAdversarialMethod
from art.attacks.evasion.wasserstein import Wasserstein
from art.attacks.evasion.zoo import ZooAttack
