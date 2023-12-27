"""
Module providing membership inference attacks.
"""
from art.attacks.inference.membership_inference.black_box import MembershipInferenceBlackBox
from art.attacks.inference.membership_inference.black_box_rule_based import MembershipInferenceBlackBoxRuleBased
from art.attacks.inference.membership_inference.white_box import MembershipInferenceWhiteBox
from art.attacks.inference.membership_inference.label_only_gap_attack import LabelOnlyGapAttack
from art.attacks.inference.membership_inference.label_only_boundary_distance import LabelOnlyDecisionBoundary
from art.attacks.inference.membership_inference.self_influence_function_attack import SelfInfluenceFunctionAttack
from art.attacks.inference.membership_inference.influence_functions import calc_s_test, calc_grad_z, calc_all_influences
from art.attacks.inference.membership_inference.shadow_models import ShadowModels
from art.attacks.inference.membership_inference.blindMI_attack import MembershipInferenceBlindMI
from art.attacks.inference.membership_inference.utils import (
    compute_pairwise_distances,
    gaussian_kernel_matrix,
    maximum_mean_discrepancy,
    sobel,
    mmd_loss,
)
