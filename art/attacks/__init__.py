"""
Module providing adversarial attacks under a common interface.
"""
from art.attacks.attack import Attack
from art.attacks.carlini import CarliniL2Method
from art.attacks.deepfool import DeepFool
from art.attacks.fast_gradient import FastGradientMethod
from art.attacks.iterative_method import BasicIterativeMethod
from art.attacks.newtonfool import NewtonFool
from art.attacks.projected_gradient_descent import ProjectedGradientDescent
from art.attacks.saliency_map import SaliencyMapMethod
from art.attacks.universal_perturbation import UniversalPerturbation
from art.attacks.virtual_adversarial import VirtualAdversarialMethod
