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
import random

from art.metrics import PDTP
from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.mark.skip_framework("dl_frameworks")
def test_membership_leakage_decision_tree(art_warning, decision_tree_estimator, get_iris_dataset):
    try:
        classifier = decision_tree_estimator()
        extra_classifier = decision_tree_estimator()
        (x_train, y_train), _ = get_iris_dataset
        prev = classifier.model.tree_
        leakage = PDTP(classifier, extra_classifier, x_train, y_train)
        logger.info("Average PDTP leakage: %.2f", (np.average(leakage)))
        logger.info("Max PDTP leakage: %.2f", (np.max(leakage)))
        assert classifier.model.tree_ == prev
        assert np.all(leakage >= 1.0)
        assert leakage.shape[0] == x_train.shape[0]
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("keras", "kerastf", "tensorflow1", "mxnet")
def test_membership_leakage_tabular(art_warning, tabular_dl_estimator, get_iris_dataset):
    try:
        classifier = tabular_dl_estimator()
        extra_classifier = tabular_dl_estimator()
        (x_train, y_train), _ = get_iris_dataset
        leakage = PDTP(classifier, extra_classifier, x_train, y_train)
        logger.info("Average PDTP leakage: %.2f", (np.average(leakage)))
        logger.info("Max PDTP leakage: %.2f", (np.max(leakage)))
        assert np.all(leakage >= 1.0)
        assert leakage.shape[0] == x_train.shape[0]
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("keras", "kerastf", "tensorflow1", "mxnet")
def test_membership_leakage_image(art_warning, image_dl_estimator, get_default_mnist_subset):
    try:
        classifier, _ = image_dl_estimator()
        extra_classifier, _ = image_dl_estimator()
        (x_train, y_train), _ = get_default_mnist_subset
        indexes = random.sample(range(x_train.shape[0]), 100)
        leakage = PDTP(classifier, extra_classifier, x_train, y_train, indexes=indexes, num_iter=1)
        logger.info("Average PDTP leakage: %.2f", (np.average(leakage)))
        logger.info("Max PDTP leakage: %.2f", (np.max(leakage)))
        assert np.all(leakage >= 1.0)
        assert leakage.shape[0] == len(indexes)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("keras", "kerastf", "tensorflow1", "mxnet")
def test_errors(art_warning, tabular_dl_estimator, get_iris_dataset, image_data_generator):
    try:
        classifier = tabular_dl_estimator()
        not_classifier = image_data_generator()
        (x_train, y_train), (x_test, y_test) = get_iris_dataset
        with pytest.raises(ValueError):
            PDTP(not_classifier, classifier, x_train, y_train)
        with pytest.raises(ValueError):
            PDTP(classifier, not_classifier, x_train, y_train)
        with pytest.raises(ValueError):
            PDTP(classifier, classifier, np.delete(x_train, 1, 1), y_train)
        with pytest.raises(ValueError):
            PDTP(classifier, classifier, x_train, y_test)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("pytorch", "tensorflow", "scikitlearn")
def test_not_implemented(art_warning, tabular_dl_estimator, get_iris_dataset, image_data_generator):
    try:
        classifier = tabular_dl_estimator()
        (x_train, y_train), _ = get_iris_dataset
        with pytest.raises(ValueError):
            PDTP(classifier, classifier, x_train, y_train)
    except ARTTestException as e:
        art_warning(e)
