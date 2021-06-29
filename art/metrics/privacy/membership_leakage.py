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
from typing import TYPE_CHECKING, Optional

import numpy as np
import scipy

from art.utils import check_and_transform_label_format, is_probability

if TYPE_CHECKING:
    from art.estimators.classification.classifier import Classifier


def PDTP(  # pylint: disable=C0103
    target_estimator: "Classifier",
    extra_estimator: "Classifier",
    x: np.ndarray,
    y: np.ndarray,
    indexes: Optional[np.ndarray] = None,
    num_iter: Optional[int] = 10,
) -> np.ndarray:
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
    :return: an array containing the average PDTP value for each sample in the training set. The higher the value,
             the higher the privacy leakage for that sample.
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
    y = check_and_transform_label_format(y, target_estimator.nb_classes)
    if y.shape[0] != x.shape[0]:
        raise ValueError("Number of rows in x and y do not match")

    results = []

    for _ in range(num_iter):
        iter_results = []
        # get probabilities from original model
        pred = target_estimator.predict(x)
        if not is_probability(pred):
            try:
                pred = scipy.special.softmax(pred, axis=1)
            except Exception as exc:
                raise ValueError("PDTP metric only supports classifiers that output logits or probabilities.") from exc
        # divide into 100 bins and return center of bin
        bins = np.array(np.arange(0.0, 1.01, 0.01).round(decimals=2))
        pred_bin_indexes = np.digitize(pred, bins)
        pred_bin = bins[pred_bin_indexes] - 0.005

        if not indexes:
            indexes = range(x.shape[0])
        for row in indexes:
            # create new model without sample in training data
            alt_x = np.delete(x, row, 0)
            alt_y = np.delete(y, row, 0)
            try:
                extra_estimator.reset()
            except NotImplementedError as exc:
                raise ValueError(
                    "PDTP metric can only be applied to classifiers that implement the reset method."
                ) from exc
            extra_estimator.fit(alt_x, alt_y)
            # get probabilities from new model
            alt_pred = extra_estimator.predict(x)
            if not is_probability(alt_pred):
                alt_pred = scipy.special.softmax(alt_pred, axis=1)
            # divide into 100 bins and return center of bin
            alt_pred_bin_indexes = np.digitize(alt_pred, bins)
            alt_pred_bin = bins[alt_pred_bin_indexes] - 0.005
            ratio_1 = pred_bin / alt_pred_bin
            ratio_2 = alt_pred_bin / pred_bin
            # get max value
            max_value = max(ratio_1.max(), ratio_2.max())
            iter_results.append(max_value)
        results.append(iter_results)

    # get average of iterations for each sample
    # We now have a list of list, internal lists represent an iteration. We need to transpose and get averages.
    per_sample = list(map(list, zip(*results)))
    avg_per_sample = np.array([sum(val) / len(val) for val in per_sample])

    # return leakage per sample
    return avg_per_sample
