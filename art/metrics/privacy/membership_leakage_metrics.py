from __future__ import absolute_import, division, print_function, unicode_literals
from typing import TYPE_CHECKING, Optional

import logging

import numpy as np
import scipy

from art.utils import check_and_transform_label_format, is_probability
from art.estimators.classification.classifier import ClassifierMixin

if TYPE_CHECKING:
    from art.estimators.classification import Classifier

def PDTP(target_estimator: "Classifier", extra_estimator: "Classifier", x: np.ndarray, y: np.ndarray,
         indexes: Optional[np.ndarray] = None) -> np.ndarray:
    """
        Compute the pointwise differential training privacy metric for the given classifier and training set.
        Taken from: https://arxiv.org/abs/1712.09136

        :param target_estimator: The classifier to be analyzed.
        :param extra_estimator: Another classifier of the same type as the target classifier, but not yet fit.
        :param x: The training data of the classifier.
        :param y: Target values (class labels) of `x`, one-hot-encoded of shape (nb_samples, nb_classes) or indices of
                  shape (nb_samples,).
        :param indexes: the subset of indexes of `x` to compute the PDTP metric on. If not supplied, PDTP will be
                        computed for all samples in `x`.
        :return: an array containing the average PDTP value for each sample in the training set.
    """
    if ClassifierMixin not in type(target_estimator).__mro__ or ClassifierMixin not in type(extra_estimator).__mro__:
        raise ValueError("PDTP metric only supports classifiers")
    if target_estimator.input_shape[0] != x.shape[1]:
        raise ValueError("Shape of x does not match input_shape of classifier")
    y = check_and_transform_label_format(y, target_estimator.nb_classes)
    if y.shape[0] != x.shape[0]:
        raise ValueError("Number of rows in x and y do not match")

    num_iter = 1
    results = []

    for i in range(num_iter):
        iter_results = []
        # get probabilities from original model
        pred = target_estimator.predict(x)
        if not is_probability(pred):
            try:
                pred = scipy.special.softmax(pred, axis=1)
            except:
                raise ValueError(
                    "PDTP metric only supports classifiers that output logits or probabilities."
                )
        if not indexes:
            indexes = range(x.shape[0])
        for row in indexes:
            # create new model without sample in training data
            alt_x = np.delete(x, row, 0)
            alt_y = np.delete(y, row, 0)
            extra_estimator.reset()
            extra_estimator.fit(alt_x, alt_y)
            # get probabilities from new model
            alt_pred = extra_estimator.predict(x)
            if not is_probability(alt_pred):
                alt_pred = scipy.special.softmax(alt_pred, axis=1)
            # divide into 100 bins and return center of bin
            pred_bin = (np.floor(pred * 100) / 100).round(decimals=2) + 0.005
            pred_bin[pred_bin > 1] = 0.995
            alt_pred_bin = (np.floor(alt_pred * 100) / 100).round(decimals=2) + 0.005
            alt_pred_bin[alt_pred_bin > 1] = 0.995
            ratio_1 = pred_bin / alt_pred_bin
            ratio_2 = alt_pred_bin / pred_bin
            # get max value
            max_value = max(ratio_1.max(), ratio_2.max())
            iter_results.append(max_value)
        results.append(iter_results)

    # get average of iterations for each sample
    # We now have a list of list, internal lists represent an iteration. We need to transpose and get averages.
    per_sample = list(map(list, zip(*results)))
    avg_per_sample = np.array([sum(l)/len(l) for l in per_sample])

    # return leakage per sample
    return avg_per_sample
