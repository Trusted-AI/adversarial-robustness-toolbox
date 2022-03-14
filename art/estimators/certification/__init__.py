"""
This module contains certified classifiers.
"""
import importlib
from art.estimators.certification import randomized_smoothing

if importlib.util.find_spec("torch") is not None:
    from art.estimators.certification import deep_z
else:
    import warnings

    warnings.warn("PyTorch not found. Not importing DeepZ functionality")
