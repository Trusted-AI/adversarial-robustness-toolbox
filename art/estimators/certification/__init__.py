"""
This module contains certified classifiers.
"""
import importlib
from art.estimators.certification import randomized_smoothing
from art.estimators.certification import derandomized_smoothing

if importlib.util.find_spec("torch") is not None:
    from art.estimators.certification import deep_z
    from art.estimators.certification import interval
else:
    import warnings

    warnings.warn("PyTorch not found. Not importing DeepZ or Interval Bound Propagation functionality")
