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
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import pytest
import numpy as np

from art.metrics import PDTP
from tests.utils import ARTTestException

logger = logging.getLogger(__name__)

@pytest.mark.skipMlFramework("dl_frameworks")
def test_membership_leakage_decision_tree(art_warning, decision_tree_estimator, get_iris_dataset):
    try:
        classifier = decision_tree_estimator()
        (x_train, y_train), _ = get_iris_dataset
        prev = classifier.model.tree_
        leakage = PDTP(classifier, classifier.clone_for_refitting(), x_train, y_train)
        print(leakage)
        print(np.average(leakage))
        print(np.max(leakage))
        assert(classifier.model.tree_ == prev)
    except ARTTestException as e:
        art_warning(e)

def test_membership_leakage_tabular(art_warning, tabular_dl_estimator, get_iris_dataset):
    try:
        classifier = tabular_dl_estimator()
        (x_train, y_train), _ = get_iris_dataset
        leakage = PDTP(classifier, classifier.clone_for_refitting(), x_train, y_train)
        print(leakage)
        print(np.average(leakage))
        print(np.max(leakage))
    except ARTTestException as e:
        art_warning(e)


# def test_membership_leakage_image(art_warning, image_dl_estimator, get_default_mnist_subset):
#     try:
#         classifier, _ = image_dl_estimator()
#         (x_train, y_train), (x_test, y_test) = get_default_mnist_subset
#         leakage = PDTP(classifier, classifier.clone_for_refitting(), x_train, y_train)
#         print(leakage)
#         print(np.average(leakage))
#         print(np.max(leakage))
#     except ARTTestException as e:
#             art_warning(e)
