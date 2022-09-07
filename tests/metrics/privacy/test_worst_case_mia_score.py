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
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np
import pytest

from art.metrics.privacy import get_roc_for_fpr, get_roc_for_multi_fprs
from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.mark.framework_agnostic
def test_worst_case_accuracy(art_warning):
    try:
        tpr = 1.0
        thr = 0.33
        fpr = 0.0
        y_true = np.array([1, 0, 1, 1])
        y_proba = np.array([0.35, 0.3, 0.33, 0.6])
        res = get_roc_for_fpr(attack_proba=y_proba, attack_true=y_true)[0]

        assert res[0] == fpr
        assert res[1] == tpr
        assert res[2] == thr
        print(res)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_worst_case_targeted_fpr_1(art_warning):
    try:
        tpr = 1.0
        thr = 0.32
        fpr = 0.5
        y_true = np.array([1, 0, 1, 1, 1, 0])
        y_proba = np.array([0.35, 0.33, 0.32, 0.6, 0.6, 0.2])
        res = get_roc_for_fpr(attack_proba=y_proba, attack_true=y_true, targeted_fpr=0.5)[0]
        assert res[0] == fpr
        assert res[1] == tpr
        assert res[2] == thr
        print(res)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_worst_case_targeted_fpr_2(art_warning):
    try:
        tpr = 0.75
        thr = 0.35
        fpr = 0.0
        y_true = np.array([1, 0, 1, 1, 1, 0])
        y_proba = np.array([0.35, 0.33, 0.32, 0.6, 0.6, 0.2])
        res = get_roc_for_fpr(attack_proba=y_proba, attack_true=y_true, targeted_fpr=0.0)[0]
        assert res[0] == fpr
        assert res[1] == tpr
        assert res[2] == thr
        print(res)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_worst_case_multiple_targeted_fpr(art_warning):
    try:

        y_true = np.array([1, 0, 1, 1, 1, 0])
        y_proba = np.array([0.35, 0.33, 0.32, 0.6, 0.6, 0.2])
        res = get_roc_for_multi_fprs(attack_proba=y_proba, attack_true=y_true, targeted_fprs=[0.0, 0.5])
        assert res[0][0] == 0.0
        assert res[1][0] == 0.75
        assert res[2][0] == 0.35

        assert res[0][1] == 0.5
        assert res[1][1] == 1.0
        assert res[2][1] == 0.32

        print(res)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_worst_case_score_per_class(art_warning):
    try:

        y_true = np.array([1, 0, 1, 1, 1, 0, 1, 0])
        y_proba = np.array([0.35, 0.33, 0.32, 0.6, 0.6, 0.2, 0.9, 0.1])
        target_model_labels = np.array([1, 1, 1, 1, 2, 1, 2, 2])
        res = get_roc_for_fpr(attack_proba=y_proba, attack_true=y_true, target_model_labels=target_model_labels)
        print(res)
    except ARTTestException as e:
        art_warning(e)
