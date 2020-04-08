"""
Module providing evasion attacks under a common interface.
"""
from art.attacks.evasion.adversarial_patch.adversarial_patch import AdversarialPatch
from art.attacks.evasion.boundary import BoundaryAttack
from art.attacks.evasion.carlini import CarliniL2Method, CarliniLInfMethod
from art.attacks.evasion.decision_tree_attack import DecisionTreeAttack
from art.attacks.evasion.deepfool import DeepFool
from art.attacks.evasion.elastic_net import ElasticNet
from art.attacks.evasion.fast_gradient import FastGradientMethod
from art.attacks.evasion.hclu import HighConfidenceLowUncertainty
from art.attacks.evasion.hop_skip_jump import HopSkipJump
from art.attacks.evasion.iterative_method import BasicIterativeMethod
from art.attacks.evasion.newtonfool import NewtonFool
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent
from art.attacks.evasion.saliency_map import SaliencyMapMethod
from art.attacks.evasion.spatial_transformation import SpatialTransformation
from art.attacks.evasion.universal_perturbation import UniversalPerturbation
from art.attacks.evasion.virtual_adversarial import VirtualAdversarialMethod
from art.attacks.evasion.zoo import ZooAttack
from art.attacks.evasion.pixel_threshold import PixelAttack
from art.attacks.evasion.pixel_threshold import ThresholdAttack
