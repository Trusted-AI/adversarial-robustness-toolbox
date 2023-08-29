"""
This module contains the ablators for the certified smoothing approaches.
"""
from art.estimators.certification.derandomized_smoothing.ablators.pytorch import (
    ColumnAblatorPyTorch,
    BlockAblatorPyTorch,
)
from art.estimators.certification.derandomized_smoothing.ablators.tensorflow import ColumnAblator, BlockAblator
