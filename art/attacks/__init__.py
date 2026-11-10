"""
Module providing adversarial attacks under a common interface.
"""

from art.attacks import evasion, extraction, inference, poisoning
from art.attacks.attack import (
    Attack,
    AttributeInferenceAttack,
    EvasionAttack,
    ExtractionAttack,
    InferenceAttack,
    PoisoningAttack,
    PoisoningAttackBlackBox,
    PoisoningAttackGenerator,
    PoisoningAttackObjectDetector,
    PoisoningAttackTransformer,
    PoisoningAttackWhiteBox,
    ReconstructionAttack,
)
