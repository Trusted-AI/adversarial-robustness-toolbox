"""
Module providing adversarial attacks under a common interface.
"""
from src.attacks.attack import Attack
from src.attacks.carlini import CarliniL2Method
from src.attacks.deepfool import DeepFool
from src.attacks.fast_gradient import FastGradientMethod
from src.attacks.newtonfool import NewtonFool
from src.attacks.saliency_map import SaliencyMapMethod
from src.attacks.universal_perturbation import UniversalPerturbation
from src.attacks.virtual_adversarial import VirtualAdversarialMethod
