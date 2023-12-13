"""
This module contains the ablators for the certified smoothing approaches.
"""
import importlib

from art.estimators.certification.derandomized_smoothing.ablators.tensorflow import ColumnAblator, BlockAblator

if importlib.util.find_spec("torch") is not None:
    from art.estimators.certification.derandomized_smoothing.ablators.pytorch import (
        ColumnAblatorPyTorch,
        BlockAblatorPyTorch,
    )
