# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
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
This module implements membership leakage metrics.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from typing import TYPE_CHECKING, Optional, Tuple
from enum import Enum, auto

import numpy as np
import scipy

from sklearn.neighbors import KNeighborsClassifier

from art.utils import check_and_transform_label_format, is_probability_array

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE


class ComparisonType(Enum):
    """
    An Enum type for different kinds of comparisons between model outputs.
    """

    RATIO = auto()
    DIFFERENCE = auto()


def PDTP(  # pylint: disable=C0103
    target_estimator: "CLASSIFIER_TYPE",
    extra_estimator: "CLASSIFIER_TYPE",
    x: np.ndarray,
    y: np.ndarray,
    indexes: Optional[np.ndarray] = None,
    num_iter: int = 10,
    comparison_type: Optional[ComparisonType] = ComparisonType.RATIO,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the pointwise differential training privacy metric for the given classifier and training set.

    | Paper link: https://arxiv.org/abs/1712.09136

    :param target_estimator: The classifier to be analyzed.
    :param extra_estimator: Another classifier of the same type as the target classifier, but not yet fit.
    :param x: The training data of the classifier.
    :param y: Target values (class labels) of `x`, one-hot-encoded of shape (nb_samples, nb_classes) or indices of
              shape (nb_samples,).
    :param indexes: the subset of indexes of `x` to compute the PDTP metric on. If not supplied, PDTP will be
                    computed for all samples in `x`.
    :param num_iter: the number of iterations of PDTP computation to run for each sample. If not supplied,
                     defaults to 10. The result is the average across iterations.
    :param comparison_type: the way in which to compare the model outputs between models trained with and without
                            a certain sample. Default is to compute the ratio.
    :return: A tuple of three arrays, containing the average (worse, standard deviation) PDTP value for each sample in
             the training set respectively. The higher the value, the higher the privacy leakage for that sample.
    """
    from art.estimators.classification.pytorch import PyTorchClassifier
    from art.estimators.classification.tensorflow import TensorFlowV2Classifier
    from art.estimators.classification.scikitlearn import ScikitlearnClassifier

    supported_classifiers = (PyTorchClassifier, TensorFlowV2Classifier, ScikitlearnClassifier)

    if not isinstance(target_estimator, supported_classifiers) or not isinstance(
        extra_estimator, supported_classifiers
    ):
        raise ValueError("PDTP metric only supports classifiers of type PyTorch, TensorFlowV2 and ScikitLearn.")
    if target_estimator.input_shape[0] != x.shape[1]:
        raise ValueError("Shape of x does not match input_shape of classifier")
    y = check_and_transform_label_format(y, nb_classes=target_estimator.nb_classes)
    if y.shape[0] != x.shape[0]:
        raise ValueError("Number of rows in x and y do not match")

    results = []

    for _ in range(num_iter):
        iter_results = []
        # get probabilities from original model
        pred = target_estimator.predict(x)
        if not is_probability_array(pred):
            try:
                pred = scipy.special.softmax(pred, axis=1)
            except Exception as exc:  # pragma: no cover
                raise ValueError("PDTP metric only supports classifiers that output logits or probabilities.") from exc
        # divide into 100 bins and return center of bin
        bins = np.array(np.arange(0.0, 1.01, 0.01).round(decimals=2))
        pred_bin_indexes = np.digitize(pred, bins)
        pred_bin_indexes[pred_bin_indexes == 101] = 100
        pred_bin = bins[pred_bin_indexes] - 0.005

        if indexes is None:
            indexes = np.array(range(x.shape[0]))
        if indexes is not None:
            for row in indexes:
                # create new model without sample in training data
                alt_x = np.delete(x, row, 0)
                alt_y = np.delete(y, row, 0)
                try:
                    extra_estimator.reset()
                except NotImplementedError as exc:  # pragma: no cover
                    raise ValueError(
                        "PDTP metric can only be applied to classifiers that implement the reset method."
                    ) from exc
                extra_estimator.fit(alt_x, alt_y)
                # get probabilities from new model
                alt_pred = extra_estimator.predict(x)
                if not is_probability_array(alt_pred):
                    alt_pred = scipy.special.softmax(alt_pred, axis=1)
                # divide into 100 bins and return center of bin
                alt_pred_bin_indexes = np.digitize(alt_pred, bins)
                alt_pred_bin_indexes[alt_pred_bin_indexes == 101] = 100
                alt_pred_bin = bins[alt_pred_bin_indexes] - 0.005
                if comparison_type == ComparisonType.RATIO:
                    ratio_1 = pred_bin / alt_pred_bin
                    ratio_2 = alt_pred_bin / pred_bin
                    # get max value
                    max_value: float = max(ratio_1.max(), ratio_2.max())
                elif comparison_type == ComparisonType.DIFFERENCE:
                    max_value = np.max(abs(pred_bin - alt_pred_bin))
                else:
                    raise ValueError("Unsupported comparison type.")
                iter_results.append(max_value)
            results.append(iter_results)

    # get average of iterations for each sample
    # We now have a list of lists, internal lists represent an iteration. We need to transpose and get averages.
    per_sample: list[list[float]] = list(map(list, zip(*results)))
    avg_per_sample = np.array([sum(val) / len(val) for val in per_sample])
    worse_per_sample = np.max(per_sample, axis=1)
    std_dev_per_sample = np.std(per_sample, axis=1)

    # return avg+worse leakage + standard deviation per sample
    return avg_per_sample, worse_per_sample, std_dev_per_sample


def SHAPr(  # pylint: disable=C0103
    target_estimator: "CLASSIFIER_TYPE",
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    knn_metric: Optional[str] = None,
) -> np.ndarray:
    """
    Compute the SHAPr membership privacy risk metric for the given classifier and training set.

    | Paper link: http://arxiv.org/abs/2112.02230

    :param target_estimator: The classifier to be analyzed.
    :param x_train: The training data of the classifier.
    :param y_train: Target values (class labels) of `x_train`, one-hot-encoded of shape (nb_samples, nb_classes) or
                    indices of shape (nb_samples,).
    :param x_test: The test data of the classifier.
    :param y_test: Target values (class labels) of `x_test`, one-hot-encoded of shape (nb_samples, nb_classes) or
                    indices of shape (nb_samples,).
    :param knn_metric: The distance metric to use for the KNN classifier (default is 'minkowski', which represents
                       Euclidean distance).
    :return: an array containing the SHAPr scores for each sample in the training set. The higher the value,
             the higher the privacy leakage for that sample. Any value above 0 should be considered a privacy leak.
    """
    if target_estimator.input_shape[0] != x_train.shape[1]:
        raise ValueError("Shape of x_train does not match input_shape of classifier")

    if x_test.shape[1] != x_train.shape[1]:
        raise ValueError("Shape of x_train does not match the shape of x_test")

    y_train = check_and_transform_label_format(y_train, target_estimator.nb_classes)
    if y_train.shape[0] != x_train.shape[0]:
        raise ValueError("Number of rows in x_train and y_train do not match")

    y_test = check_and_transform_label_format(y_test, target_estimator.nb_classes)
    if y_test.shape[0] != x_test.shape[0]:
        raise ValueError("Number of rows in x_test and y_test do not match")

    n_train_samples = x_train.shape[0]
    pred_train = target_estimator.predict(x_train)
    pred_test = target_estimator.predict(x_test)

    if knn_metric:
        knn = KNeighborsClassifier(metric=knn_metric)
    else:
        knn = KNeighborsClassifier()
    knn.fit(pred_train, y_train)

    results = []

    n_test = pred_test.shape[0]
    for i_test in range(n_test):
        results_test = []
        pred = pred_test[i_test]
        y_0 = y_test[i_test]
        # returns sorted indexes, from closest to farthest
        n_indexes = knn.kneighbors([pred], n_neighbors=n_train_samples, return_distance=False)
        # reverse - from farthest to closest
        n_indexes = n_indexes.reshape(-1)[::-1]
        sorted_y_train = y_train[n_indexes]
        sorted_indexes = np.argsort(n_indexes)
        # compute partial contribution incrementally
        first = True
        phi_y_prev: float = 0.0
        y_indicator_prev = 0
        for i_train in range(sorted_y_train.shape[0]):
            y = sorted_y_train[i_train]
            y_indicator = 1 if np.all(y == y_0) else 0
            if first:
                phi_y = y_indicator / n_train_samples
                first = False
            else:
                phi_y = phi_y_prev + ((y_indicator - y_indicator_prev) / (n_train_samples - i_train))
            results_test.append(phi_y)
            phi_y_prev = phi_y
            y_indicator_prev = y_indicator
        # return to original order of training samples
        results_test_sorted = np.array(results_test)[sorted_indexes]
        results.append(results_test_sorted.tolist())

    # need to sum across test samples (outer list) for each train sample (inner list)
    per_sample = list(map(list, zip(*results)))
    # normalize so it's comparable across different sizes of train and test datasets
    sum_per_sample = np.array([sum(val) for val in per_sample], dtype=np.float32) * n_train_samples / n_test

    return sum_per_sample
