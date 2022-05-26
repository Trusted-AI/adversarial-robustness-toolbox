# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2022
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
This module implements metric for inference attack worst case accuracy measurement.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from typing import Optional, List, Tuple, Union

import logging
import numpy as np
from sklearn.metrics import roc_curve


TPR = float
FPR = float
THR = float


def _calculate_roc_for_fpr(y_true, y_proba, targeted_fpr):
    """Get FPR, TPR and, THRESHOLD based on the targeted_fpr (such that FPR <= targeted_fpr)"""
    fpr, tpr, thr = roc_curve(y_true=y_true, y_score=y_proba)
    # take the highest fpr and an appropriated threshold that achieve at least FPR=fpr
    if np.isnan(fpr).all() or np.isnan(tpr).all():
        logging.warning("TPR or FPR values are NaN")
        return None, None, None

    targeted_fpr_idx = np.where(fpr <= targeted_fpr)[0][-1]
    return fpr[targeted_fpr_idx], tpr[targeted_fpr_idx], thr[targeted_fpr_idx]


def get_roc_for_fpr(  # pylint: disable=C0103
    attack_proba: np.ndarray,
    attack_true: np.ndarray,
    target_model_labels: Optional[np.ndarray] = None,
    targeted_fpr: Optional[float] = 0.001,
) -> Union[List[Tuple[FPR, TPR, THR]], List[Tuple[int, FPR, TPR, THR]]]:
    """
    Compute the attack TPR, THRESHOLD and achieved FPR based on the targeted FPR. This implementation supports only
    binary attack prediction labels {0,1}. The returned THRESHOLD defines the decision threshold on the attack
    probabilities (meaning if p < THRESHOLD predict 0, otherwise predict 1)
    | Related paper link: https://arxiv.org/abs/2112.03570

    :param attack_proba: Predicted attack probabilities.
    :param attack_true: True attack labels.
    :param targeted_fpr: the targeted False Positive Rate, attack accuracy will be calculated based on this FPRs.
     If not supplied, get_roc_for_fpr will be computed for `0.001` FPR.
    :param target_model_labels: Original labels, if provided the Accuracy and threshold will be calculated per each
     class separately.
    :return: list of tuples the contains (original label (if target_model_labels is provided),
    Achieved FPR, TPR, Threshold).
    """

    if attack_proba.shape[0] != attack_true.shape[0]:
        raise ValueError("Number of rows in attack_pred and attack_true do not match")
    if target_model_labels is not None and attack_proba.shape[0] != target_model_labels.shape[0]:
        raise ValueError("Number of rows in target_model_labels and attack_pred do not match")

    results = []

    if target_model_labels is not None:
        values, _ = np.unique(target_model_labels, return_counts=True)
        for v in values:
            idxs = np.where(target_model_labels == v)[0]
            fpr, tpr, thr = _calculate_roc_for_fpr(
                y_proba=attack_proba[idxs], y_true=attack_true[idxs], targeted_fpr=targeted_fpr
            )
            results.append((v, fpr, tpr, thr))
        return results

    fpr, tpr, thr = _calculate_roc_for_fpr(y_proba=attack_proba, y_true=attack_true, targeted_fpr=targeted_fpr)
    return [(fpr, tpr, thr)]


def get_roc_for_multi_fprs(
    attack_proba: np.ndarray,
    attack_true: np.ndarray,
    targeted_fprs: np.ndarray,
) -> Tuple[List[FPR], List[TPR], List[THR]]:
    """
    Compute the attack ROC based on the targeted FPRs. This implementation supports only binary
    attack prediction labels. The returned list of THRESHOLDs defines the decision threshold on the attack
    probabilities (meaning if p < THRESHOLD predict 0, otherwise predict 1) for each provided fpr

    | Related paper link: https://arxiv.org/abs/2112.03570

    :param attack_proba: Predicted attack probabilities.
    :param attack_true: True attack labels.
    :param targeted_fprs: the set of targeted FPR (False Positive Rates), attack accuracy will be calculated based on
    these FPRs.
    :return: list of tuples that  (TPR, Threshold, Achieved FPR).
    """

    if attack_proba.shape[0] != attack_true.shape[0]:
        raise ValueError("Number of rows in attack_pred and attack_true do not match")

    tpr = []
    thr = []
    fpr = []

    for t_fpr in targeted_fprs:
        res = _calculate_roc_for_fpr(y_proba=attack_proba, y_true=attack_true, targeted_fpr=t_fpr)

        fpr.append(res[0])
        tpr.append(res[1])
        thr.append(res[2])

    return fpr, tpr, thr
