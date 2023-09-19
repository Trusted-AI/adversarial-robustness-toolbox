"""
This module contains certified classifiers.
"""
import importlib
from art.estimators.certification.randomized_smoothing.randomized_smoothing import RandomizedSmoothingMixin
from art.estimators.certification.randomized_smoothing.numpy import NumpyRandomizedSmoothing
from art.estimators.certification.randomized_smoothing.tensorflow import TensorFlowV2RandomizedSmoothing
from art.estimators.certification.randomized_smoothing.pytorch import PyTorchRandomizedSmoothing
from art.estimators.certification.derandomized_smoothing.pytorch import PyTorchDeRandomizedSmoothing
from art.estimators.certification.derandomized_smoothing.tensorflow import TensorFlowV2DeRandomizedSmoothing
from art.estimators.certification.object_seeker.object_seeker import ObjectSeekerMixin
from art.estimators.certification.object_seeker.pytorch import PyTorchObjectSeeker

if importlib.util.find_spec("torch") is not None:
    from art.estimators.certification.deep_z.deep_z import ZonoDenseLayer
    from art.estimators.certification.deep_z.deep_z import ZonoBounds
    from art.estimators.certification.deep_z.deep_z import ZonoConv
    from art.estimators.certification.deep_z.deep_z import ZonoReLU
    from art.estimators.certification.deep_z.pytorch import PytorchDeepZ
    from art.estimators.certification.interval.interval import PyTorchIntervalDense
    from art.estimators.certification.interval.interval import PyTorchIntervalConv2D
    from art.estimators.certification.interval.interval import PyTorchIntervalReLU
    from art.estimators.certification.interval.interval import PyTorchIntervalFlatten
    from art.estimators.certification.interval.interval import PyTorchIntervalBounds
    from art.estimators.certification.interval.pytorch import PyTorchIBPClassifier
else:
    import warnings

    warnings.warn("PyTorch not found. Not importing DeepZ or Interval Bound Propagation functionality")
