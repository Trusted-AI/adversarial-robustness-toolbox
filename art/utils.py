# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Module providing convenience functions.
"""
# pylint: disable=C0302
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import math
import os
import shutil
import sys
import tarfile
import warnings
import zipfile
from functools import wraps
from inspect import signature
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union

import numpy as np
import six
from scipy.special import gammainc  # pylint: disable=E0611
from tqdm.auto import tqdm

from art import config

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------------------------- CONSTANTS AND TYPES


DATASET_TYPE = Tuple[  # pylint: disable=C0103
    Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], float, float
]
CLIP_VALUES_TYPE = Tuple[Union[int, float, np.ndarray], Union[int, float, np.ndarray]]  # pylint: disable=C0103

if TYPE_CHECKING:
    # pylint: disable=R0401,C0412
    from art.defences.preprocessor.preprocessor import Preprocessor

    PREPROCESSING_TYPE = Optional[  # pylint: disable=C0103
        Union[
            Tuple[Union[int, float, np.ndarray], Union[int, float, np.ndarray]], Preprocessor, Tuple[Preprocessor, ...]
        ]
    ]

    from art.estimators.classification.blackbox import BlackBoxClassifier
    from art.estimators.classification.catboost import CatBoostARTClassifier
    from art.estimators.classification.classifier import (
        Classifier,
        ClassifierClassLossGradients,
        ClassifierDecisionTree,
        ClassifierLossGradients,
        ClassifierNeuralNetwork,
    )
    from art.estimators.classification.detector_classifier import DetectorClassifier
    from art.estimators.classification.ensemble import EnsembleClassifier
    from art.estimators.classification.GPy import GPyGaussianProcessClassifier
    from art.estimators.classification.keras import KerasClassifier
    from art.experimental.estimators.classification.jax import JaxClassifier
    from art.estimators.classification.lightgbm import LightGBMClassifier
    from art.estimators.classification.mxnet import MXClassifier
    from art.estimators.classification.pytorch import PyTorchClassifier
    from art.estimators.classification.query_efficient_bb import QueryEfficientGradientEstimationClassifier
    from art.estimators.classification.scikitlearn import (
        ScikitlearnAdaBoostClassifier,
        ScikitlearnBaggingClassifier,
        ScikitlearnClassifier,
        ScikitlearnDecisionTreeClassifier,
        ScikitlearnDecisionTreeRegressor,
        ScikitlearnExtraTreeClassifier,
        ScikitlearnExtraTreesClassifier,
        ScikitlearnGradientBoostingClassifier,
        ScikitlearnLogisticRegression,
        ScikitlearnRandomForestClassifier,
        ScikitlearnSVC,
    )
    from art.estimators.classification.tensorflow import TensorFlowClassifier, TensorFlowV2Classifier
    from art.estimators.classification.xgboost import XGBoostClassifier
    from art.estimators.certification.deep_z import PytorchDeepZ
    from art.estimators.certification.interval import PyTorchIBPClassifier
    from art.estimators.certification.derandomized_smoothing.derandomized_smoothing import BlockAblator, ColumnAblator
    from art.estimators.generation import TensorFlowGenerator
    from art.estimators.generation.tensorflow import TensorFlowV2Generator
    from art.estimators.object_detection.object_detector import ObjectDetector
    from art.estimators.object_detection.pytorch_object_detector import PyTorchObjectDetector
    from art.estimators.object_detection.pytorch_faster_rcnn import PyTorchFasterRCNN
    from art.estimators.object_detection.pytorch_yolo import PyTorchYolo
    from art.estimators.object_detection.tensorflow_faster_rcnn import TensorFlowFasterRCNN
    from art.estimators.object_detection.tensorflow_v2_faster_rcnn import TensorFlowV2FasterRCNN
    from art.estimators.pytorch import PyTorchEstimator
    from art.estimators.keras import KerasEstimator
    from art.estimators.regression.scikitlearn import ScikitlearnRegressor
    from art.estimators.regression.pytorch import PyTorchRegressor
    from art.estimators.regression.keras import KerasRegressor
    from art.estimators.regression.blackbox import BlackBoxRegressor
    from art.estimators.speech_recognition.pytorch_deep_speech import PyTorchDeepSpeech
    from art.estimators.speech_recognition.tensorflow_lingvo import TensorFlowLingvoASR
    from art.estimators.tensorflow import TensorFlowV2Estimator

    CLASSIFIER_LOSS_GRADIENTS_TYPE = Union[  # pylint: disable=C0103
        ClassifierLossGradients,
        EnsembleClassifier,
        GPyGaussianProcessClassifier,
        KerasClassifier,
        JaxClassifier,
        MXClassifier,
        PyTorchClassifier,
        ScikitlearnLogisticRegression,
        ScikitlearnSVC,
        TensorFlowClassifier,
        TensorFlowV2Classifier,
        QueryEfficientGradientEstimationClassifier,
    ]

    CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE = Union[  # pylint: disable=C0103
        ClassifierClassLossGradients,
        EnsembleClassifier,
        GPyGaussianProcessClassifier,
        KerasClassifier,
        MXClassifier,
        PyTorchClassifier,
        ScikitlearnLogisticRegression,
        ScikitlearnSVC,
        TensorFlowClassifier,
        TensorFlowV2Classifier,
    ]

    CLASSIFIER_NEURALNETWORK_TYPE = Union[  # pylint: disable=C0103
        ClassifierNeuralNetwork,
        DetectorClassifier,
        EnsembleClassifier,
        KerasClassifier,
        MXClassifier,
        PyTorchClassifier,
        TensorFlowClassifier,
        TensorFlowV2Classifier,
    ]

    CLASSIFIER_DECISION_TREE_TYPE = Union[  # pylint: disable=C0103
        ClassifierDecisionTree,
        LightGBMClassifier,
        ScikitlearnDecisionTreeClassifier,
        ScikitlearnExtraTreesClassifier,
        ScikitlearnGradientBoostingClassifier,
        ScikitlearnRandomForestClassifier,
        XGBoostClassifier,
    ]

    CLASSIFIER_TYPE = Union[  # pylint: disable=C0103
        Classifier,
        BlackBoxClassifier,
        CatBoostARTClassifier,
        DetectorClassifier,
        EnsembleClassifier,
        GPyGaussianProcessClassifier,
        KerasClassifier,
        JaxClassifier,
        LightGBMClassifier,
        MXClassifier,
        PyTorchClassifier,
        ScikitlearnClassifier,
        ScikitlearnDecisionTreeClassifier,
        ScikitlearnExtraTreeClassifier,
        ScikitlearnAdaBoostClassifier,
        ScikitlearnBaggingClassifier,
        ScikitlearnExtraTreesClassifier,
        ScikitlearnGradientBoostingClassifier,
        ScikitlearnRandomForestClassifier,
        ScikitlearnLogisticRegression,
        ScikitlearnSVC,
        TensorFlowClassifier,
        TensorFlowV2Classifier,
        XGBoostClassifier,
        CLASSIFIER_NEURALNETWORK_TYPE,
    ]

    GENERATOR_TYPE = Union[TensorFlowGenerator, TensorFlowV2Generator]  # pylint: disable=C0103

    REGRESSOR_TYPE = Union[  # pylint: disable=C0103
        ScikitlearnRegressor, ScikitlearnDecisionTreeRegressor, PyTorchRegressor, KerasRegressor, BlackBoxRegressor
    ]

    OBJECT_DETECTOR_TYPE = Union[  # pylint: disable=C0103
        ObjectDetector,
        PyTorchObjectDetector,
        PyTorchFasterRCNN,
        PyTorchYolo,
        TensorFlowFasterRCNN,
        TensorFlowV2FasterRCNN,
    ]

    SPEECH_RECOGNIZER_TYPE = Union[  # pylint: disable=C0103
        PyTorchDeepSpeech,
        TensorFlowLingvoASR,
    ]

    PYTORCH_ESTIMATOR_TYPE = Union[  # pylint: disable=C0103
        PyTorchClassifier,
        PyTorchDeepSpeech,
        PyTorchEstimator,
        PyTorchObjectDetector,
        PyTorchFasterRCNN,
        PyTorchYolo,
        PyTorchRegressor,
    ]

    KERAS_ESTIMATOR_TYPE = Union[  # pylint: disable=C0103
        KerasClassifier,
        KerasEstimator,
        KerasRegressor,
    ]

    TENSORFLOWV2_ESTIMATOR_TYPE = Union[  # pylint: disable=C0103
        TensorFlowV2Classifier,
        TensorFlowV2Estimator,
        TensorFlowV2FasterRCNN,
    ]

    ESTIMATOR_TYPE = Union[  # pylint: disable=C0103
        CLASSIFIER_TYPE,
        REGRESSOR_TYPE,
        OBJECT_DETECTOR_TYPE,
        SPEECH_RECOGNIZER_TYPE,
        PYTORCH_ESTIMATOR_TYPE,
        KERAS_ESTIMATOR_TYPE,
        TENSORFLOWV2_ESTIMATOR_TYPE,
    ]

    CLONABLE = Union[  # pylint: disable=C0103
        ScikitlearnClassifier,
        PyTorchClassifier,
        TensorFlowV2Classifier,
        KerasEstimator,
        PyTorchRegressor,
        ScikitlearnRegressor,
        XGBoostClassifier,
    ]

    ABLATOR_TYPE = Union[BlockAblator, ColumnAblator]  # pylint: disable=C0103

    CERTIFIER_TYPE = Union[PytorchDeepZ]  # pylint: disable=C0103
    IBP_CERTIFIER_TYPE = Union[PyTorchIBPClassifier]  # pylint: disable=C0103

# --------------------------------------------------------------------------------------------------------- DEPRECATION


class _Deprecated:
    """
    Create Deprecated() singleton object.
    """

    _instance = None

    def __new__(cls):
        if _Deprecated._instance is None:
            _Deprecated._instance = object.__new__(cls)
        return _Deprecated._instance


Deprecated = _Deprecated()


def deprecated(end_version: str, *, reason: str = "", replaced_by: str = "") -> Callable:
    """
    Deprecate a function or method and raise a `DeprecationWarning`.

    The `@deprecated` decorator is used to deprecate functions and methods. Several cases are supported. For example
    one can use it to deprecate a function that has become redundant or rename a function. The following code examples
    provide different use cases of how to use decorator.

    .. code-block:: python

      @deprecated("0.1.5", replaced_by="sum")
      def simple_addition(a, b):
          return a + b

    :param end_version: Release version of removal.
    :param reason: Additional deprecation reason.
    :param replaced_by: Function that replaces deprecated function.
    """

    def decorator(function):
        reason_msg = "\n" + reason if reason else reason
        replaced_msg = f" It will be replaced by '{replaced_by}'." if replaced_by else replaced_by
        deprecated_msg = (
            f"Function '{function.__name__}' is deprecated and will be removed in future release {end_version}."
        )

        @wraps(function)
        def wrapper(*args, **kwargs):
            warnings.simplefilter("always", category=DeprecationWarning)
            warnings.warn(
                deprecated_msg + replaced_msg + reason_msg,
                category=DeprecationWarning,
                stacklevel=2,
            )
            warnings.simplefilter("default", category=DeprecationWarning)
            return function(*args, **kwargs)

        return wrapper

    return decorator


def deprecated_keyword_arg(identifier: str, end_version: str, *, reason: str = "", replaced_by: str = "") -> Callable:
    """
    Deprecate a keyword argument and raise a `DeprecationWarning`.

    The `@deprecated_keyword_arg` decorator is used to deprecate keyword arguments. The deprecated keyword argument must
    default to `Deprecated`. Several use cases are supported. For example one can use it to to rename a keyword
    identifier. The following code examples provide different use cases of how to use the decorator.

    .. code-block:: python

      @deprecated_keyword_arg("print", "1.1.0", replaced_by="verbose")
      def simple_addition(a, b, print=Deprecated, verbose=False):
          if verbose:
              print(a + b)
          return a + b

      @deprecated_keyword_arg("verbose", "1.1.0")
      def simple_addition(a, b, verbose=Deprecated):
          return a + b

    :param identifier: Keyword identifier.
    :param end_version: Release version of removal.
    :param reason: Additional deprecation reason.
    :param replaced_by: Function that replaces deprecated function.
    """

    def decorator(function):
        reason_msg = "\n" + reason if reason else reason
        replaced_msg = f" It will be replaced by '{replaced_by}'." if replaced_by else replaced_by
        deprecated_msg = (
            f"Keyword argument '{identifier}' in '{function.__name__}' is deprecated and will be removed in"
            f" future release {end_version}."
        )

        @wraps(function)
        def wrapper(*args, **kwargs):
            params = signature(function).bind(*args, **kwargs)
            params.apply_defaults()

            if params.signature.parameters[identifier].default is not Deprecated:
                raise ValueError("Deprecated keyword argument must default to the Decorator singleton.")
            if replaced_by != "" and replaced_by not in params.arguments:
                raise ValueError("Deprecated keyword replacement not found in function signature.")

            if params.arguments[identifier] is not Deprecated:
                warnings.simplefilter("always", category=DeprecationWarning)
                warnings.warn(deprecated_msg + replaced_msg + reason_msg, category=DeprecationWarning, stacklevel=2)
                warnings.simplefilter("default", category=DeprecationWarning)
            return function(*args, **kwargs)

        return wrapper

    return decorator


# ----------------------------------------------------------------------------------------------------- MATH OPERATIONS


def projection_l1_1(values: np.ndarray, eps: Union[int, float, np.ndarray]) -> np.ndarray:
    """
    This function computes the orthogonal projections of a batch of points on L1-balls of given radii
    The batch size is  m = values.shape[0].  The points are flattened to dimension
    n = np.prod(value.shape[1:]).  This is required to facilitate sorting.

    If a[0] <= ... <= a[n-1], then the projection can be characterized using the largest  j  such that
    a[j+1] +...+ a[n-1] - a[j]*(n-j-1) >= eps. The  ith  coordinate of projection is equal to  0
    if i=0,...,j.

    :param values:  A batch of  m  points, each an ndarray
    :param eps:  The radii of the respective L1-balls
    :return: projections
    """
    # pylint: disable=C0103

    shp = values.shape
    a = values.copy()
    n = np.prod(a.shape[1:])
    m = a.shape[0]
    a = a.reshape((m, n))
    sgns = np.sign(a)
    a = np.abs(a)

    a_argsort = a.argsort(axis=1)
    a_sorted = np.zeros((m, n))
    for i in range(m):
        a_sorted[i, :] = a[i, a_argsort[i, :]]
    a_argsort_inv = a.argsort(axis=1).argsort(axis=1)
    mat = np.zeros((m, 2))

    #   if  a_sorted[i, n-1]  >= a_sorted[i, n-2] + eps,  then the projection is  [0,...,0,eps]
    done = early_done = False
    active = np.array([1] * m)
    after_vec = np.zeros((m, n))
    proj = a_sorted.copy()
    j = n - 2
    while j >= 0:
        mat[:, 0] += a_sorted[:, j + 1]  # =  sum(a_sorted[: i] :  i = j + 1,...,n-1
        mat[:, 1] = a_sorted[:, j] * (n - j - 1) + eps
        #  Find the max in each problem  max{ sum{a_sorted[:, i] : i=j+1,..,n-1} , a_sorted[:, j] * (n-j-1) + eps }
        row_maxes = np.max(mat, axis=1)
        #  Set to  1  if  max >  a_sorted[:, j] * (n-j-1) + eps  >  sum ;  otherwise, set to  0
        ind_set = np.sign(np.sign(row_maxes - mat[:, 0]))
        #  ind_set = ind_set.reshape((m, 1))
        #   Multiplier for activation
        act_multiplier = (1 - ind_set) * active
        act_multiplier = np.transpose([np.transpose(act_multiplier)] * n)
        #  if done, the projection is supported by the current indices  j+1,..,n-1   and the amount by which each
        #  has to be reduced is  delta
        delta = (mat[:, 0] - eps) / (n - j - 1)
        #    The vector of reductions
        delta_vec = np.transpose(np.array([delta] * (n - j - 1)))
        #   The sub-vectors:  a_sorted[:, (j+1):]
        a_sub = a_sorted[:, (j + 1) :]
        #   After reduction by delta_vec
        a_after = a_sub - delta_vec
        after_vec[:, (j + 1) :] = a_after
        proj += act_multiplier * (after_vec - proj)
        active = active * ind_set
        if sum(active) == 0:
            done = early_done = True
            break
        j -= 1
    if not early_done:
        delta = (mat[:, 0] + a_sorted[:, 0] - eps) / n
        ind_set = np.sign(np.maximum(delta, 0))
        act_multiplier = ind_set * active
        act_multiplier = np.transpose([np.transpose(act_multiplier)] * n)
        delta_vec = np.transpose(np.array([delta] * n))
        a_after = a_sorted - delta_vec
        proj += act_multiplier * (a_after - proj)
        done = True
    if not done:
        proj = active * (a_sorted - proj)

    for i in range(m):
        proj[i, :] = proj[i, a_argsort_inv[i, :]]

    proj = sgns * proj
    proj = proj.reshape(shp)

    return proj


def projection_l1_2(values: np.ndarray, eps: Union[int, float, np.ndarray]) -> np.ndarray:
    """
    This function computes the orthogonal projections of a batch of points on L1-balls of given radii
    The batch size is  m = values.shape[0].  The points are flattened to dimension
    n = np.prod(value.shape[1:]).  This is required to facilitate sorting.

    Starting from a vector  a = (a1,...,an)  such that  a1 >= ... >= an >= 0,  a1 + ... + an > 1,
    we first move to  a' = a - (t,...,t)  such that either a1 + ... + an >= 1 ,  an >= 0,
    and  min( a1 + ... + an  - nt - 1, an -t ) = 0.  This means  t = min( (a1 + ... + an - 1)/n, an).
    If  t = (a1 + ... + an - 1)/n , then  a' is the desired projection.  Otherwise, the problem is reduced to
    finding the projection of  (a1 - t, ... , a{n-1} - t ).

    :param values:  A batch of  m  points, each an ndarray
    :param eps:  The radii of the respective L1-balls
    :return: projections
    """
    # pylint: disable=C0103
    shp = values.shape
    a = values.copy()
    n = np.prod(a.shape[1:])
    m = a.shape[0]
    a = a.reshape((m, n))
    sgns = np.sign(a)
    a = np.abs(a)
    a_argsort = a.argsort(axis=1)
    a_sorted = np.zeros((m, n))
    for i in range(m):
        a_sorted[i, :] = a[i, a_argsort[i, :]]

    a_argsort_inv = a.argsort(axis=1).argsort(axis=1)
    row_sums = np.sum(a, axis=1)
    mat = np.zeros((m, 2))
    mat0 = np.zeros((m, 2))
    a_var = a_sorted.copy()
    for j in range(n):
        mat[:, 0] = (row_sums - eps) / (n - j)
        mat[:, 1] = a_var[:, j]
        mat0[:, 1] = np.min(mat, axis=1)
        min_t = np.max(mat0, axis=1)
        if np.max(min_t) < 1e-8:
            continue
        row_sums = row_sums - a_var[:, j] * (n - j)
        a_var[:, (j + 1) :] = a_var[:, (j + 1) :] - np.matmul(min_t.reshape((m, 1)), np.ones((1, n - j - 1)))
        a_var[:, j] = a_var[:, j] - min_t
    proj = np.zeros((m, n))
    for i in range(m):
        proj[i, :] = a_var[i, a_argsort_inv[i, :]]

    proj = sgns * proj
    proj = proj.reshape(shp)
    return proj


def projection(values: np.ndarray, eps: Union[int, float, np.ndarray], norm_p: Union[int, float, str]) -> np.ndarray:
    """
    Project `values` on the L_p norm ball of size `eps`.

    :param values: Array of perturbations to clip.
    :param eps: Maximum norm allowed.
    :param norm_p: L_p norm to use for clipping.
            Only 1, 2 , `np.Inf` 1.1 and 1.2 supported for now.
            1.1 and 1.2 compute orthogonal projections on l1-ball, using two different algorithms
    :return: Values of `values` after projection.
    """
    # Pick a small scalar to avoid division by 0
    tol = 10e-8
    values_tmp = values.reshape((values.shape[0], -1))

    if norm_p == 2:
        if isinstance(eps, np.ndarray):
            raise NotImplementedError("The parameter `eps` of type `np.ndarray` is not supported to use with norm 2.")

        values_tmp = values_tmp * np.expand_dims(
            np.minimum(1.0, eps / (np.linalg.norm(values_tmp, axis=1) + tol)), axis=1
        )

    elif norm_p == 1:
        if isinstance(eps, np.ndarray):
            raise NotImplementedError("The parameter `eps` of type `np.ndarray` is not supported to use with norm 1.")

        values_tmp = values_tmp * np.expand_dims(
            np.minimum(1.0, eps / (np.linalg.norm(values_tmp, axis=1, ord=1) + tol)),
            axis=1,
        )
    elif norm_p == 1.1:
        values_tmp = projection_l1_1(values_tmp, eps)
    elif norm_p == 1.2:
        values_tmp = projection_l1_2(values_tmp, eps)

    elif norm_p in [np.inf, "inf"]:
        if isinstance(eps, np.ndarray):
            eps = eps * np.ones_like(values)
            eps = eps.reshape([eps.shape[0], -1])  # type: ignore

        values_tmp = np.sign(values_tmp) * np.minimum(abs(values_tmp), eps)

    else:
        raise NotImplementedError(
            'Values of `norm_p` different from 1, 2, `np.inf` and "inf" are currently not ' "supported."
        )

    values = values_tmp.reshape(values.shape)

    return values


def random_sphere(
    nb_points: int,
    nb_dims: int,
    radius: Union[int, float, np.ndarray],
    norm: Union[int, float, str],
) -> np.ndarray:
    """
    Generate uniformly at random `m x n`-dimension points in the `norm`-norm ball with radius `radius` and centered
    around 0.

    :param nb_points: Number of random data points.
    :param nb_dims: Dimensionality of the sphere.
    :param radius: Radius of the sphere.
    :param norm: Current support: 1, 2, np.inf, "inf".
    :return: The generated random sphere.
    """
    if norm == 1:
        if isinstance(radius, np.ndarray):
            raise NotImplementedError(
                "The parameter `radius` of type `np.ndarray` is not supported to use with norm 1."
            )

        var_u = np.random.uniform(size=(nb_points, nb_dims))
        var_v = np.sort(var_u)
        v_pre = np.concatenate((np.zeros((nb_points, 1)), var_v[:, : nb_dims - 1]), axis=-1)
        x = var_v - v_pre
        res = radius * x * np.random.choice([-1, 1], (nb_points, nb_dims))

    elif norm == 2:
        if isinstance(radius, np.ndarray):
            raise NotImplementedError(
                "The parameter `radius` of type `np.ndarray` is not supported to use with norm 2."
            )

        a_tmp = np.random.randn(nb_points, nb_dims)
        s_2 = np.sum(a_tmp ** 2, axis=1)
        base = gammainc(nb_dims / 2.0, s_2 / 2.0) ** (1 / nb_dims) * radius / np.sqrt(s_2)
        res = a_tmp * (np.tile(base, (nb_dims, 1))).T

    elif norm in [np.inf, "inf"]:
        if isinstance(radius, np.ndarray):
            radius = radius * np.ones(shape=(nb_points, nb_dims))

        res = np.random.uniform(-radius, radius, (nb_points, nb_dims))

    else:
        raise NotImplementedError(f"Norm {norm} not supported")

    return res


def uniform_sample_from_sphere_or_ball(
    nb_points: int,
    nb_dims: int,
    radius: Union[int, float, np.ndarray],
    sample_space: str = "ball",
    norm: Union[int, float, str] = 2,
) -> np.ndarray:
    """
    Generate a sample of  <nb_points>  distributed independently and uniformly on the sphere (with respect to the given
    norm) in dimension <nb_dims> with radius <radius> and centered at the origin.  Note that the sphere is the boundary
    of the ball, i.e., every point on the sphere has the same distance to the origin.

    :param nb_points: Number of random data points
    :param nb_dims: Dimensionality of the sphere
    :param radius: Radius of the sphere
    :param sample_space: One of 'b', 's', 'sphere', 'ball'
    :param norm: Current support: 1, 2, np.inf, "inf"
    :return: The sampled points from the sphere (i.e., boundary of the ball)
    """
    assert sample_space in ["b", "s", "sphere", "ball"]

    if norm == 1:
        if sample_space in ["s", "sphere"]:
            y = np.random.exponential(1, (nb_points, nb_dims))
            sums = np.sum(y, axis=1)
            scal = np.outer(sums, np.ones(nb_dims))
            y = y / scal
        else:
            y = np.random.exponential(1, (nb_points, nb_dims + 1))
            sums = np.sum(y, axis=1)
            scal = np.outer(sums, np.ones(nb_dims + 1))
            y = y / scal
            y = y[:, :nb_dims]

        y = y * np.random.choice([-1, 1], (nb_points, nb_dims))
        if type(radius) in [int, float]:
            res = y * radius
        else:
            radii = np.outer(radius, np.ones(nb_dims))
            res = y * radii

    elif norm == 2:
        x = np.random.normal(0.0, 1.0, (nb_points, nb_dims))
        scal = radius / np.sqrt(np.sum(x * x, axis=1))
        scal = np.transpose(np.array(list(scal) * nb_dims).reshape((nb_dims, nb_points)))
        res = x * scal
        if sample_space in ["b", "ball"]:
            rnd = np.random.rand(nb_points)
            scal = np.float_power(rnd, 1 / nb_dims)
            scal = np.outer(scal, np.ones(nb_dims))
            res = res * scal

    elif norm in [np.inf, "inf"]:
        if sample_space in ["b", "ball"]:
            x = np.random.uniform(-1.0, 1.0, (nb_points, nb_dims))
        else:
            x = np.random.uniform(0, 1.0, (nb_points, nb_dims))
            rnd = np.random.random((nb_points, nb_dims))
            maxes = np.max(rnd, axis=1)
            maxes = np.outer(maxes, np.ones(nb_dims))
            x = np.maximum((rnd >= maxes), x) * np.random.choice([-1, 1], (nb_points, nb_dims))
        if type(radius) in [int, float]:
            res = x * radius
        else:
            radii = np.outer(radius, np.ones(nb_dims))
            res = x * radii
    else:
        raise NotImplementedError(f"Norm {norm} not supported")

    return res


def original_to_tanh(
    x_original: np.ndarray,
    clip_min: Union[float, np.ndarray],
    clip_max: Union[float, np.ndarray],
    tanh_smoother: float = 0.999999,
) -> np.ndarray:
    """
    Transform input from original to tanh space.

    :param x_original: An array with the input to be transformed.
    :param clip_min: Minimum clipping value.
    :param clip_max: Maximum clipping value.
    :param tanh_smoother: Scalar for multiplying arguments of arctanh to avoid division by zero.
    :return: An array holding the transformed input.
    """
    x_tanh = np.clip(x_original, clip_min, clip_max)
    x_tanh = (x_tanh - clip_min) / (clip_max - clip_min)
    x_tanh = np.arctanh(((x_tanh * 2) - 1) * tanh_smoother)
    return x_tanh


def tanh_to_original(
    x_tanh: np.ndarray,
    clip_min: Union[float, np.ndarray],
    clip_max: Union[float, np.ndarray],
) -> np.ndarray:
    """
    Transform input from tanh to original space.

    :param x_tanh: An array with the input to be transformed.
    :param clip_min: Minimum clipping value.
    :param clip_max: Maximum clipping value.
    :return: An array holding the transformed input.
    """
    return (np.tanh(x_tanh) + 1.0) / 2.0 * (clip_max - clip_min) + clip_min


# --------------------------------------------------------------------------------------------------- LABELS OPERATIONS


def to_categorical(labels: Union[np.ndarray, List[float]], nb_classes: Optional[int] = None) -> np.ndarray:
    """
    Convert an array of labels to binary class matrix.

    :param labels: An array of integer labels of shape `(nb_samples,)`.
    :param nb_classes: The number of classes (possible labels).
    :return: A binary matrix representation of `y` in the shape `(nb_samples, nb_classes)`.
    """
    labels = np.array(labels, dtype=int)
    if nb_classes is None:
        nb_classes = np.max(labels) + 1
    categorical = np.zeros((labels.shape[0], nb_classes), dtype=np.float32)
    categorical[np.arange(labels.shape[0]), np.squeeze(labels)] = 1
    return categorical


def float_to_categorical(labels: np.ndarray, nb_classes: Optional[int] = None):
    """
    Convert an array of floating point labels to binary class matrix.

    :param labels: An array of floating point labels of shape `(nb_samples,)`
    :param nb_classes: The number of classes (possible labels)
    :return: A binary matrix representation of `labels` in the shape `(nb_samples, nb_classes)`
    :rtype: `np.ndarray`
    """
    labels = np.array(labels)
    unique = np.unique(labels)
    unique.sort()
    indexes = [np.where(unique == value)[0] for value in labels]
    if nb_classes is None:
        nb_classes = len(unique)
    categorical = np.zeros((labels.shape[0], nb_classes), dtype=np.float32)
    categorical[np.arange(labels.shape[0]), np.squeeze(indexes)] = 1
    return categorical


def floats_to_one_hot(labels: np.ndarray):
    """
    Convert a 2D-array of floating point labels to binary class matrix.

    :param labels: A 2D-array of floating point labels of shape `(nb_samples, nb_classes)`
    :return: A binary matrix representation of `labels` in the shape `(nb_samples, nb_classes)`
    :rtype: `np.ndarray`
    """
    labels = np.array(labels)
    for feature in labels.T:  # pylint: disable=E1133
        unique = np.unique(feature)
        unique.sort()
        for index, value in enumerate(unique):
            feature[feature == value] = index
    return labels.astype(np.float32)


def check_and_transform_label_format(
    labels: np.ndarray, nb_classes: Optional[int], return_one_hot: bool = True
) -> np.ndarray:
    """
    Check label format and transform to one-hot-encoded labels if necessary

    :param labels: An array of integer labels of shape `(nb_samples,)`, `(nb_samples, 1)` or `(nb_samples, nb_classes)`.
    :param nb_classes: The number of classes. If None the number of classes is determined automatically.
    :param return_one_hot: True if returning one-hot encoded labels, False if returning index labels.
    :return: Labels with shape `(nb_samples, nb_classes)` (one-hot) or `(nb_samples,)` (index).
    """
    labels_return = labels

    if len(labels.shape) == 2 and labels.shape[1] > 1:  # multi-class, one-hot encoded
        if not return_one_hot:
            labels_return = np.argmax(labels, axis=1)
            labels_return = np.expand_dims(labels_return, axis=1)
    elif len(labels.shape) == 2 and labels.shape[1] == 1:
        if nb_classes is None:
            nb_classes = int(np.max(labels) + 1)
        if nb_classes > 2:  # multi-class, index labels
            if return_one_hot:
                labels_return = to_categorical(labels, nb_classes)
            else:
                labels_return = np.expand_dims(labels, axis=1)
        elif nb_classes == 2:  # binary, index labels
            if return_one_hot:
                labels_return = to_categorical(labels, nb_classes)
        else:
            raise ValueError(
                "Shape of labels not recognised."
                "Please provide labels in shape (nb_samples,) or (nb_samples, "
                "nb_classes)"
            )
    elif len(labels.shape) == 1:  # index labels
        if return_one_hot:
            labels_return = to_categorical(labels, nb_classes)
        else:
            labels_return = np.expand_dims(labels, axis=1)
    else:
        raise ValueError(
            "Shape of labels not recognised."
            "Please provide labels in shape (nb_samples,) or (nb_samples, "
            "nb_classes)"
        )

    return labels_return


def random_targets(labels: np.ndarray, nb_classes: int) -> np.ndarray:
    """
    Given a set of correct labels, randomly changes some correct labels to target labels different from the original
    ones. These can be one-hot encoded or integers.

    :param labels: The correct labels.
    :param nb_classes: The number of classes for this model.
    :return: An array holding the randomly-selected target classes, one-hot encoded.
    """
    if len(labels.shape) > 1:
        labels = np.argmax(labels, axis=1)

    result = np.zeros(labels.shape)

    for class_ind in range(nb_classes):
        other_classes = list(range(nb_classes))
        other_classes.remove(class_ind)
        in_cl = labels == class_ind
        result[in_cl] = np.random.choice(other_classes)

    return to_categorical(result, nb_classes)


def least_likely_class(x: np.ndarray, classifier: "CLASSIFIER_TYPE") -> np.ndarray:
    """
    Compute the least likely class predictions for sample `x`. This strategy for choosing attack targets was used in
    (Kurakin et al., 2016).

    | Paper link: https://arxiv.org/abs/1607.02533

    :param x: A data sample of shape accepted by `classifier`.
    :param classifier: The classifier used for computing predictions.
    :return: Least-likely class predicted by `classifier` for sample `x` in one-hot encoding.
    """
    return to_categorical(np.argmin(classifier.predict(x), axis=1), nb_classes=classifier.nb_classes)


def second_most_likely_class(x: np.ndarray, classifier: "CLASSIFIER_TYPE") -> np.ndarray:
    """
    Compute the second most likely class predictions for sample `x`. This strategy can be used for choosing target
    labels for an attack to improve its chances to succeed.

    :param x: A data sample of shape accepted by `classifier`.
    :param classifier: The classifier used for computing predictions.
    :return: Second most likely class predicted by `classifier` for sample `x` in one-hot encoding.
    """
    return to_categorical(
        np.argpartition(classifier.predict(x), -2, axis=1)[:, -2],
        nb_classes=classifier.nb_classes,
    )


def get_label_conf(y_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the confidence and the label of the most probable class given a vector of class confidences

    :param y_vec: Vector of class confidences, no. of instances as first dimension.
    :return: Confidences and labels.
    """
    assert len(y_vec.shape) == 2

    confs, labels = np.amax(y_vec, axis=1), np.argmax(y_vec, axis=1)
    return confs, labels


def get_labels_np_array(preds: np.ndarray) -> np.ndarray:
    """
    Returns the label of the most probable class given a array of class confidences.

    :param preds: Array of class confidences, nb of instances as first dimension.
    :return: Labels.
    """
    if len(preds.shape) >= 2:
        preds_max = np.amax(preds, axis=1, keepdims=True)
    else:
        preds_max = np.round(preds)
    y = preds == preds_max
    y = y.astype(np.uint8)
    return y


def get_feature_values(x: np.ndarray, single_index_feature: bool) -> list:
    """
    Returns a list of unique values of a given feature.

    :param x: The feature column(s).
    :param single_index_feature: Bool representing whether this is a single-column or multiple-column feature (for
                                 example 1-hot encoded and then scaled).
    :return: For a single-column feature, a simple list containing all possible values, in increasing order.
             For a multi-column feature, a list of lists, where each internal list represents a column and the values
             represent the possible values for that column (in increasing order).
    """
    values = []
    if single_index_feature:
        values = np.unique(x).tolist()
    else:
        for column in x.T:
            values.append(np.unique(column).tolist())
    return values


def get_feature_index(feature: Union[int, slice]) -> Union[int, slice]:
    """
    Returns a modified feature index: in case of a slice of size 1, returns the corresponding integer. In case
    of a slice with missing params, tries to fill them. Otherwise, returns the same value (integer or slice) as passed.

    :param feature: The index or slice representing a feature to attack
    :return: An integer representing a single column index or a slice representing a multi-column index
    """
    if isinstance(feature, int):
        return feature

    start = feature.start
    stop = feature.stop
    step = feature.step
    if start is None:
        start = 0
    if step is None:
        step = 1
    if stop is not None and ((stop - start) // step) == 1:
        return start

    return slice(start, stop, step)


def remove_attacked_feature(attack_feature: Union[int, slice], non_numerical_features: Optional[List[int]]):
    """
    Removes the attacked feature from the list of non-numeric features to encode.

    :param attack_feature: The index or slice representing a feature to attack
    :param non_numerical_features: a list of feature indexes that require encoding in order to feed into an ML model
                                    (i.e., strings)
    :return:
    """
    if non_numerical_features is not None and isinstance(attack_feature, int):
        if attack_feature in non_numerical_features:
            non_numerical_features.remove(attack_feature)


def compute_success_array(
    classifier: "CLASSIFIER_TYPE",
    x_clean: np.ndarray,
    labels: np.ndarray,
    x_adv: np.ndarray,
    targeted: bool = False,
    batch_size: int = 1,
) -> float:
    """
    Compute the success rate of an attack based on clean samples, adversarial samples and targets or correct labels.

    :param classifier: Classifier used for prediction.
    :param x_clean: Original clean samples.
    :param labels: Correct labels of `x_clean` if the attack is untargeted, or target labels of the attack otherwise.
    :param x_adv: Adversarial samples to be evaluated.
    :param targeted: `True` if the attack is targeted. In that case, `labels` are treated as target classes instead of
           correct labels of the clean samples.
    :param batch_size: Batch size.
    :return: Percentage of successful adversarial samples.
    """
    adv_preds = classifier.predict(x_adv, batch_size=batch_size)
    if len(adv_preds.shape) >= 2:
        adv_preds = np.argmax(adv_preds, axis=1)
    else:
        adv_preds = np.round(adv_preds)
    if targeted:
        attack_success = adv_preds == np.argmax(labels, axis=1)
    else:
        preds = classifier.predict(x_clean, batch_size=batch_size)
        if len(preds.shape) >= 2:
            preds = np.argmax(preds, axis=1)
        else:
            preds = np.round(preds)
        attack_success = adv_preds != preds

    return attack_success


def compute_success(
    classifier: "CLASSIFIER_TYPE",
    x_clean: np.ndarray,
    labels: np.ndarray,
    x_adv: np.ndarray,
    targeted: bool = False,
    batch_size: int = 1,
) -> float:
    """
    Compute the success rate of an attack based on clean samples, adversarial samples and targets or correct labels.

    :param classifier: Classifier used for prediction.
    :param x_clean: Original clean samples.
    :param labels: Correct labels of `x_clean` if the attack is untargeted, or target labels of the attack otherwise.
    :param x_adv: Adversarial samples to be evaluated.
    :param targeted: `True` if the attack is targeted. In that case, `labels` are treated as target classes instead of
           correct labels of the clean samples.
    :param batch_size: Batch size.
    :return: Percentage of successful adversarial samples.
    """
    attack_success = compute_success_array(classifier, x_clean, labels, x_adv, targeted, batch_size)
    return np.sum(attack_success) / x_adv.shape[0]


def compute_accuracy(preds: np.ndarray, labels: np.ndarray, abstain: bool = True) -> Tuple[float, float]:
    """
    Compute the accuracy rate and coverage rate of predictions
    In the case where predictions are abstained, those samples are ignored.

    :param preds: Predictions.
    :param labels: Correct labels of `x`.
    :param abstain: True if ignore abstained prediction, False if count them as incorrect.
    :return: Tuple of accuracy rate and coverage rate.
    """
    has_pred = np.sum(preds, axis=1)
    idx_pred = np.where(has_pred)[0]
    labels = np.argmax(labels[idx_pred], axis=1)
    num_correct = np.sum(np.argmax(preds[idx_pred], axis=1) == labels)
    coverage_rate = len(idx_pred) / preds.shape[0]

    if abstain:
        acc_rate = num_correct / preds[idx_pred].shape[0]
    else:
        acc_rate = num_correct / preds.shape[0]

    return acc_rate, coverage_rate


# -------------------------------------------------------------------------------------------------- DATASET OPERATIONS


def load_cifar10(
    raw: bool = False,
) -> DATASET_TYPE:
    """
    Loads CIFAR10 dataset from config.CIFAR10_PATH or downloads it if necessary.

    :param raw: `True` if no preprocessing should be applied to the data. Otherwise, data is normalized to 1.
    :return: `(x_train, y_train), (x_test, y_test), min, max`
    """

    def load_batch(fpath: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Utility function for loading CIFAR batches, as written in Keras.

        :param fpath: Full path to the batch file.
        :return: `(data, labels)`
        """
        with open(fpath, "rb") as file_:
            if sys.version_info < (3,):
                content = six.moves.cPickle.load(file_)
            else:
                content = six.moves.cPickle.load(file_, encoding="bytes")
                content_decoded = {}
                for key, value in content.items():
                    content_decoded[key.decode("utf8")] = value
                content = content_decoded
        data = content["data"]
        labels = content["labels"]

        data = data.reshape(data.shape[0], 3, 32, 32)
        return data, labels

    path = get_file(
        "cifar-10-batches-py",
        extract=True,
        path=config.ART_DATA_PATH,
        url="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    )

    num_train_samples = 50000

    x_train = np.zeros((num_train_samples, 3, 32, 32), dtype=np.uint8)
    y_train = np.zeros((num_train_samples,), dtype=np.uint8)

    for i in range(1, 6):
        fpath = os.path.join(path, "data_batch_" + str(i))
        data, labels = load_batch(fpath)
        x_train[(i - 1) * 10000 : i * 10000, :, :, :] = data
        y_train[(i - 1) * 10000 : i * 10000] = labels

    fpath = os.path.join(path, "test_batch")
    x_test, y_test = load_batch(fpath)

    # Set channels last
    x_train = x_train.transpose((0, 2, 3, 1))
    x_test = x_test.transpose((0, 2, 3, 1))

    min_, max_ = 0.0, 255.0
    if not raw:
        min_, max_ = 0.0, 1.0
        x_train, y_train = preprocess(x_train, y_train, clip_values=(0, 255))
        x_test, y_test = preprocess(x_test, y_test, clip_values=(0, 255))

    return (x_train, y_train), (x_test, y_test), min_, max_


def load_mnist(
    raw: bool = False,
) -> DATASET_TYPE:
    """
    Loads MNIST dataset from `config.ART_DATA_PATH` or downloads it if necessary.

    :param raw: `True` if no preprocessing should be applied to the data. Otherwise, data is normalized to 1.
    :return: `(x_train, y_train), (x_test, y_test), min, max`.
    """
    path = get_file(
        "mnist.npz",
        path=config.ART_DATA_PATH,
        url="https://s3.amazonaws.com/img-datasets/mnist.npz",
    )

    dict_mnist = np.load(path)
    x_train = dict_mnist["x_train"]
    y_train = dict_mnist["y_train"]
    x_test = dict_mnist["x_test"]
    y_test = dict_mnist["y_test"]
    dict_mnist.close()

    # Add channel axis
    min_, max_ = 0.0, 255.0
    if not raw:
        min_, max_ = 0.0, 1.0
        x_train = np.expand_dims(x_train, axis=3)
        x_test = np.expand_dims(x_test, axis=3)
        x_train, y_train = preprocess(x_train, y_train)
        x_test, y_test = preprocess(x_test, y_test)

    return (x_train, y_train), (x_test, y_test), min_, max_


def load_stl() -> DATASET_TYPE:
    """
    Loads the STL-10 dataset from `config.ART_DATA_PATH` or downloads it if necessary.

    :return: `(x_train, y_train), (x_test, y_test), min, max`.
    """
    min_, max_ = 0.0, 1.0

    # Download and extract data if needed

    path = get_file(
        "stl10_binary",
        path=config.ART_DATA_PATH,
        extract=True,
        url="https://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz",
    )

    with open(os.path.join(path, "train_X.bin"), "rb") as f_numpy:
        x_train = np.fromfile(f_numpy, dtype=np.uint8)
        x_train = np.reshape(x_train, (-1, 3, 96, 96))

    with open(os.path.join(path, "test_X.bin"), "rb") as f_numpy:
        x_test = np.fromfile(f_numpy, dtype=np.uint8)
        x_test = np.reshape(x_test, (-1, 3, 96, 96))

    # Set channel last
    x_train = x_train.transpose((0, 2, 3, 1))
    x_test = x_test.transpose((0, 2, 3, 1))

    with open(os.path.join(path, "train_y.bin"), "rb") as f_numpy:
        y_train = np.fromfile(f_numpy, dtype=np.uint8)
        y_train -= 1

    with open(os.path.join(path, "test_y.bin"), "rb") as f_numpy:
        y_test = np.fromfile(f_numpy, dtype=np.uint8)
        y_test -= 1

    x_train, y_train = preprocess(x_train, y_train)
    x_test, y_test = preprocess(x_test, y_test)

    return (x_train, y_train), (x_test, y_test), min_, max_


def load_iris(raw: bool = False, test_set: float = 0.3) -> DATASET_TYPE:
    """
    Loads the UCI Iris dataset from `config.ART_DATA_PATH` or downloads it if necessary.

    :param raw: `True` if no preprocessing should be applied to the data. Otherwise, data is normalized to 1.
    :param test_set: Proportion of the data to use as validation split. The value should be between 0 and 1.
    :return: Entire dataset and labels.
    """
    from sklearn.datasets import load_iris as load_iris_sk

    iris = load_iris_sk()
    data = iris.data
    if not raw:
        data /= np.amax(data)
    labels = to_categorical(iris.target, 3)

    min_, max_ = float(np.amin(data)), float(np.amax(data))

    # Split training and test sets
    split_index = int((1 - test_set) * len(data) / 3)
    x_train = np.vstack((data[:split_index], data[50 : 50 + split_index], data[100 : 100 + split_index]))
    y_train = np.vstack(
        (
            labels[:split_index],
            labels[50 : 50 + split_index],
            labels[100 : 100 + split_index],
        )
    )

    if split_index >= 49:
        x_test, y_test = None, None
    else:

        x_test = np.vstack(
            (
                data[split_index:50],
                data[50 + split_index : 100],
                data[100 + split_index :],
            )
        ).astype(np.float32)
        y_test = np.vstack(
            (
                labels[split_index:50],
                labels[50 + split_index : 100],
                labels[100 + split_index :],
            )
        )
        assert len(x_train) + len(x_test) == 150

        # Shuffle test set
        random_indices = np.random.permutation(len(y_test))
        x_test, y_test = x_test[random_indices], y_test[random_indices]

    # Shuffle training set
    random_indices = np.random.permutation(len(y_train))
    x_train, y_train = x_train[random_indices].astype(np.float32), y_train[random_indices]

    return (x_train, y_train), (x_test, y_test), min_, max_  # type: ignore


def load_diabetes(raw: bool = False, test_set: float = 0.3) -> DATASET_TYPE:
    """
    Loads the Diabetes Regression dataset from sklearn.

    :param raw: `True` if no preprocessing should be applied to the data. Otherwise, data is normalized to 1.
    :param test_set: Proportion of the data to use as validation split. The value should be between 0 and 1.
    :return: Entire dataset and labels.
    """
    from sklearn.datasets import load_diabetes as load_diabetes_sk

    diabetes = load_diabetes_sk()
    data = diabetes.data
    if not raw:
        data /= np.amax(data, axis=0)
    targets = diabetes.target

    min_, max_ = np.amin(data), np.amax(data)

    # Shuffle data set
    random_indices = np.random.permutation(len(data))
    data, targets = data[random_indices], targets[random_indices]

    # Split training and test sets
    split_index = int((1 - test_set) * len(data))
    x_train = data[:split_index]
    y_train = targets[:split_index]
    x_test = data[split_index:]
    y_test = targets[split_index:]

    return (x_train, y_train), (x_test, y_test), min_, max_


def load_nursery(
    raw: bool = False, scaled: bool = True, test_set: float = 0.2, transform_social: bool = False
) -> DATASET_TYPE:
    """
    Loads the UCI Nursery dataset from `config.ART_DATA_PATH` or downloads it if necessary.

    :param raw: `True` if no preprocessing should be applied to the data. Otherwise, categorical data is one-hot
                encoded and data is scaled using sklearn's StandardScaler according to the value of `scaled`.
    :param scaled: `True` if data should be scaled.
    :param test_set: Proportion of the data to use as validation split. The value should be between 0 and 1.
    :param transform_social: If `True`, transforms the social feature to be binary for the purpose of attribute
                             inference. This is done by assigning the original value 'problematic' the new value 1, and
                             the other original values are assigned the new value 0.
    :return: Entire dataset and labels as numpy array.
    """
    import pandas as pd
    import sklearn.preprocessing

    # Download data if needed
    path = get_file(
        "nursery.data",
        path=config.ART_DATA_PATH,
        extract=False,
        url=[
            "https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.data",
            "https://www.dropbox.com/s/l24hwvkuueor6lp/nursery.data?dl=1",
        ],
    )

    # load data
    features_names = ["parents", "has_nurs", "form", "children", "housing", "finance", "social", "health", "label"]
    categorical_features = ["parents", "has_nurs", "form", "housing", "finance", "social", "health"]
    data = pd.read_csv(path, sep=",", names=features_names, engine="python")
    # remove rows with missing label or too sparse label
    data = data.dropna(subset=["label"])
    data.drop(data.loc[data["label"] == "recommend"].index, axis=0, inplace=True)

    # fill missing values
    data["children"] = data["children"].fillna(0)

    for col in ["parents", "has_nurs", "form", "housing", "finance", "social", "health"]:
        data[col] = data[col].fillna("other")

    # make categorical label
    def modify_label(value):  # 5 classes
        if value == "not_recom":
            return 0
        if value == "very_recom":
            return 1
        if value == "priority":
            return 2
        if value == "spec_prior":
            return 3
        raise Exception(f"Bad label value: {value}")

    data["label"] = data["label"].apply(modify_label)
    data["children"] = data["children"].apply(lambda x: 4 if x == "more" else x)

    if transform_social:

        def modify_social(value):
            if value == "problematic":
                return 1
            return 0

        data["social"] = data["social"].apply(modify_social)
        categorical_features.remove("social")

    if not raw:
        # one-hot-encode categorical features
        features_to_remove = []
        for feature in categorical_features:
            all_values = data.loc[:, feature]
            values = list(all_values.unique())
            data[feature] = pd.Categorical(data.loc[:, feature], categories=values, ordered=False)
            one_hot_vector = pd.get_dummies(data[feature], prefix=feature)
            data = pd.concat([data, one_hot_vector], axis=1)
            features_to_remove.append(feature)
        data = data.drop(features_to_remove, axis=1)

        # normalize data
        if scaled:
            label = data.loc[:, "label"]
            features = data.drop(["label"], axis=1)
            scaler = sklearn.preprocessing.StandardScaler()
            scaler.fit(features)
            scaled_features = pd.DataFrame(scaler.transform(features), columns=features.columns)
            data = pd.concat([label, scaled_features], axis=1, join="inner")

    features = data.drop(["label"], axis=1)
    if raw:
        numeric_features = features.drop(categorical_features, axis=1).to_numpy().astype(np.int32)
        min_, max_ = np.amin(numeric_features), np.amax(numeric_features)
    else:
        min_, max_ = np.amin(features.to_numpy().astype(np.float64)), np.amax(features.to_numpy().astype(np.float64))

    # Split training and test sets
    stratified = sklearn.model_selection.StratifiedShuffleSplit(n_splits=1, test_size=test_set, random_state=18)
    for train_set_i, test_set_i in stratified.split(data, data["label"]):
        train = data.iloc[train_set_i]
        test = data.iloc[test_set_i]
    x_train = train.drop(["label"], axis=1).to_numpy()
    y_train = train.loc[:, "label"].to_numpy()
    x_test = test.drop(["label"], axis=1).to_numpy()
    y_test = test.loc[:, "label"].to_numpy()

    if not raw and not scaled:
        x_train = x_train.astype(np.float64)
        x_test = x_test.astype(np.float64)

    return (x_train, y_train), (x_test, y_test), min_, max_


def load_dataset(
    name: str,
) -> DATASET_TYPE:
    """
    Loads or downloads the dataset corresponding to `name`. Options are: `mnist`, `cifar10`, `stl10`, `iris`, `nursery`
    and `diabetes`.

    :param name: Name of the dataset.
    :return: The dataset separated in training and test sets as `(x_train, y_train), (x_test, y_test), min, max`.
    :raises NotImplementedError: If the dataset is unknown.
    """
    if "mnist" in name:
        return load_mnist()
    if "cifar10" in name:
        return load_cifar10()
    if "stl10" in name:
        return load_stl()
    if "iris" in name:
        return load_iris()
    if "nursery" in name:
        return load_nursery()
    if "diabetes" in name:
        return load_diabetes()

    raise NotImplementedError(f"There is no loader for dataset '{name}'.")


def _extract(full_path: str, path: str) -> bool:
    archive: Union[zipfile.ZipFile, tarfile.TarFile]
    if full_path.endswith("tar"):  # pragma: no cover
        if tarfile.is_tarfile(full_path):
            archive = tarfile.open(full_path, "r:")
    elif full_path.endswith("tar.gz"):  # pragma: no cover
        if tarfile.is_tarfile(full_path):
            archive = tarfile.open(full_path, "r:gz")
    elif full_path.endswith("zip"):  # pragma: no cover
        if zipfile.is_zipfile(full_path):
            archive = zipfile.ZipFile(full_path)  # pylint: disable=R1732
        else:
            return False
    else:
        return False

    try:
        archive.extractall(path)
    except (tarfile.TarError, RuntimeError, KeyboardInterrupt):  # pragma: no cover
        if os.path.exists(path):
            if os.path.isfile(path):
                os.remove(path)
            else:
                shutil.rmtree(path)
        raise
    return True


def get_file(
    filename: str, url: Union[str, List[str]], path: Optional[str] = None, extract: bool = False, verbose: bool = False
) -> str:
    """
    Downloads a file from a URL if it not already in the cache. The file at indicated by `url` is downloaded to the
    path `path` (default is ~/.art/data). and given the name `filename`. Files in tar, tar.gz, tar.bz, and zip formats
    can also be extracted. This is a simplified version of the function with the same name in Keras.

    :param filename: Name of the file.
    :param url: Download URL.
    :param path: Folder to store the download. If not specified, `~/.art/data` is used instead.
    :param extract: If true, tries to extract the archive.
    :param verbose: If true, print download progress bar.
    :return: Path to the downloaded file.
    """
    if isinstance(url, str):
        url_list = [url]
    else:
        url_list = url

    if path is None:
        path_ = os.path.expanduser(config.ART_DATA_PATH)
    else:
        path_ = os.path.expanduser(path)
    if not os.access(path_, os.W_OK):
        path_ = os.path.join("/tmp", ".art")

    if not os.path.exists(path_):
        os.makedirs(path_)

    if extract:
        extract_path = os.path.join(path_, filename)
        full_path = extract_path + ".tar.gz"
    else:
        full_path = os.path.join(path_, filename)

    # Determine if dataset needs downloading
    download = not os.path.exists(full_path)

    if download:
        logger.info("Downloading data from %s", url)
        error_msg = "URL fetch failure on {}: {} -- {}"
        try:
            for url_i in url_list:
                try:
                    from six.moves.urllib.error import HTTPError, URLError
                    from six.moves.urllib.request import urlretrieve

                    # The following two lines should prevent occasionally occurring
                    # [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed (_ssl.c:847)
                    import ssl

                    ssl._create_default_https_context = ssl._create_unverified_context  # pylint: disable=W0212

                    if verbose:
                        with tqdm() as t_bar:
                            # pylint: disable=W0640
                            last_block = [0]

                            def progress_bar(blocks: int = 1, block_size: int = 1, total_size: Optional[int] = None):
                                """
                                :param blocks: Number of blocks transferred so far [default: 1].
                                :param block_size: Size of each block (in tqdm units) [default: 1].
                                :param total_size: Total size (in tqdm units). If [default: None] or -1, remains
                                                   unchanged.
                                """
                                if total_size not in (None, -1):
                                    t_bar.total = total_size
                                displayed = t_bar.update((blocks - last_block[0]) * block_size)
                                last_block[0] = blocks
                                return displayed

                            urlretrieve(url_i, full_path, reporthook=progress_bar)
                    else:
                        urlretrieve(url_i, full_path)

                except HTTPError as exception:  # pragma: no cover
                    raise Exception(
                        error_msg.format(url_i, exception.code, exception.msg)  # type: ignore
                    ) from HTTPError  # type: ignore
                except URLError as exception:  # pragma: no cover
                    raise Exception(
                        error_msg.format(url_i, exception.errno, exception.reason)  # type: ignore
                    ) from HTTPError  # type: ignore
        except (Exception, KeyboardInterrupt):  # pragma: no cover
            if os.path.exists(full_path):
                os.remove(full_path)
            raise

    if extract:
        if not os.path.exists(extract_path):
            _extract(full_path, path_)
        return extract_path

    return full_path


def make_directory(dir_path: str) -> None:
    """
    Creates the specified tree of directories if needed.

    :param dir_path: Folder or file path.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def clip_and_round(x: np.ndarray, clip_values: Optional["CLIP_VALUES_TYPE"], round_samples: float) -> np.ndarray:
    """
    Rounds the input to the correct level of granularity.
    Useful to ensure data passed to classifier can be represented
    in the correct domain, e.g., [0, 255] integers verses [0,1]
    or [0, 255] floating points.

    :param x: Sample input with shape as expected by the model.
    :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
           for features, or `None` if no clipping should be performed.
    :param round_samples: The resolution of the input domain to round the data to, e.g., 1.0, or 1/255. Set to 0 to
           disable.
    """
    if round_samples == 0.0:
        return x
    if clip_values is not None:
        np.clip(x, clip_values[0], clip_values[1], out=x)
    x = np.around(x / round_samples) * round_samples
    return x


def preprocess(
    x: np.ndarray,
    y: np.ndarray,
    nb_classes: int = 10,
    clip_values: Optional["CLIP_VALUES_TYPE"] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scales `x` to [0, 1] and converts `y` to class categorical confidences.

    :param x: Data instances.
    :param y: Labels.
    :param nb_classes: Number of classes in dataset.
    :param clip_values: Original data range allowed value for features, either one respective scalar or one value per
           feature.
    :return: Rescaled values of `x`, `y`.
    """
    if clip_values is None:
        min_, max_ = np.amin(x), np.amax(x)
    else:
        min_, max_ = clip_values

    normalized_x = (x - min_) / (max_ - min_)
    categorical_y = to_categorical(y, nb_classes)

    return normalized_x, categorical_y


def segment_by_class(data: Union[np.ndarray, List[int]], classes: np.ndarray, num_classes: int) -> List[np.ndarray]:
    """
    Returns segmented data according to specified features.

    :param data: Data to be segmented.
    :param classes: Classes used to segment data, e.g., segment according to predicted label or to `y_train` or other
                    array of one hot encodings the same length as data.
    :param num_classes: How many features.
    :return: Segmented data according to specified features.
    """
    by_class: List[List[int]] = [[] for _ in range(num_classes)]
    for indx, feature in enumerate(classes):
        if len(classes.shape) == 2 and classes.shape[1] > 1:
            assigned = int(np.argmax(feature))
        else:
            assigned = int(feature)
        by_class[assigned].append(data[indx])

    return [np.asarray(i) for i in by_class]


def performance_diff(
    model1: "CLASSIFIER_TYPE",
    model2: "CLASSIFIER_TYPE",
    test_data: np.ndarray,
    test_labels: np.ndarray,
    perf_function: Union[str, Callable] = "accuracy",
    **kwargs,
) -> float:
    """
    Calculates the difference in performance between two models on the test_data with a performance function.

    Note: For multi-label classification, f1 scores will use 'micro' averaging unless otherwise specified.

    :param model1: A trained ART classifier.
    :param model2: Another trained ART classifier.
    :param test_data: The data to test both model's performance.
    :param test_labels: The labels to the testing data.
    :param perf_function: The performance metric to be used. One of ['accuracy', 'f1'] or a callable function
           `(true_labels, model_labels[, kwargs]) -> float`.
    :param kwargs: Arguments to add to performance function.
    :return: The difference in performance performance(model1) - performance(model2).
    :raises `ValueError`: If an unsupported performance function is requested.
    """
    from sklearn.metrics import accuracy_score, f1_score

    model1_labels = model1.predict(test_data)
    model2_labels = model2.predict(test_data)

    if perf_function == "accuracy":
        model1_acc = accuracy_score(test_labels, model1_labels, **kwargs)
        model2_acc = accuracy_score(test_labels, model2_labels, **kwargs)
        return model1_acc - model2_acc

    if perf_function == "f1":
        n_classes = test_labels.shape[1]
        if n_classes > 2 and "average" not in kwargs:
            kwargs["average"] = "micro"
        model1_f1 = f1_score(test_labels, model1_labels, **kwargs)
        model2_f1 = f1_score(test_labels, model2_labels, **kwargs)
        return model1_f1 - model2_f1

    if callable(perf_function):
        return perf_function(test_labels, model1_labels, **kwargs) - perf_function(test_labels, model2_labels, **kwargs)

    raise ValueError(f"Performance function '{perf_function}' not supported")


def is_probability(vector: np.ndarray) -> bool:
    """
    Check if an 1D-array is a probability vector.

    :param vector: An 1D-array.
    :return: True if it is a probability vector.
    """
    is_sum_1 = math.isclose(np.sum(vector), 1.0, rel_tol=1e-03)
    is_smaller_1 = np.amax(vector) <= 1.0
    is_larger_0 = np.amin(vector) >= 0.0

    return is_sum_1 and is_smaller_1 and is_larger_0


def is_probability_array(array: np.ndarray) -> bool:
    """
    Check if a multi-dimensional array is an array of probabilities.

    :param vector: A numpy array.
    :return: True if it is an array of probabilities.
    """
    if len(array.shape) == 1:
        return is_probability(array)
    sum_array = np.sum(array, axis=1)
    ones = np.ones_like(sum_array)
    is_sum_1 = np.allclose(sum_array, ones, rtol=1e-03)
    is_smaller_1 = np.amax(array) <= 1.0
    is_larger_0 = np.amin(array) >= 0.0

    return is_sum_1 and is_smaller_1 and is_larger_0


def pad_sequence_input(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply padding to a batch of 1-dimensional samples such that it has shape of (batch_size, max_length).

    :param x: A batch of 1-dimensional input data, e.g. `np.array([np.array([1,2,3]), np.array([4,5,6,7])])`.
    :return: The padded input batch and its corresponding mask.
    """
    max_length = max(map(len, x))
    batch_size = x.shape[0]

    # note: use dtype of inner elements
    x_padded = np.zeros((batch_size, max_length), dtype=x[0].dtype)
    x_mask = np.zeros((batch_size, max_length), dtype=bool)

    for i, x_i in enumerate(x):
        x_padded[i, : len(x_i)] = x_i
        x_mask[i, : len(x_i)] = 1
    return x_padded, x_mask


# -------------------------------------------------------------------------------------------------------- CUDA SUPPORT


def to_cuda(x: "torch.Tensor") -> "torch.Tensor":
    """
    Move the tensor from the CPU to the GPU if a GPU is available.

    :param x: CPU Tensor to move to GPU if available.
    :return: The CPU Tensor moved to a GPU Tensor.
    """
    from torch.cuda import is_available

    use_cuda = is_available()
    if use_cuda:  # pragma: no cover
        x = x.cuda()
    return x


def from_cuda(x: "torch.Tensor") -> "torch.Tensor":
    """
    Move the tensor from the GPU to the CPU if a GPU is available.

    :param x: GPU Tensor to move to CPU if available.
    :return: The GPU Tensor moved to a CPU Tensor.
    """
    from torch.cuda import is_available

    use_cuda = is_available()
    if use_cuda:  # pragma: no cover
        x = x.cpu()
    return x
