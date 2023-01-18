# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2023
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
A unittest class for testing the subset scanning detector.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import pytest

import numpy as np

from art.attacks.evasion.fast_gradient import FastGradientMethod
from art.defences.detector.evasion import SubsetScanningDetector

from tests.utils import ARTTestException, get_image_classifier_kr, get_image_classifier_tf, get_image_classifier_pt

logger = logging.getLogger(__name__)


@pytest.fixture()
def get_classifier(framework):
    def _get_classifier():
        if framework in ("keras", "kerastf"):
            classifier = get_image_classifier_kr()
        elif framework == "tensorflow2":
            classifier, _ = get_image_classifier_tf()
        elif framework == "pytorch":
            classifier = get_image_classifier_pt()
        else:
            classifier = None

        return classifier

    return _get_classifier


@pytest.mark.only_with_platform("keras", "kerastf", "tensorflow2", "pytorch")
def test_subsetscannning_detector_scan_clean(art_warning, get_default_mnist_subset, get_classifier):
    (x_train, _), (x_test, _) = get_default_mnist_subset
    classifier = get_classifier()

    # Data for detector
    bgd_data = x_train
    clean_data = x_test

    try:
        detector = SubsetScanningDetector(classifier, bgd_data=bgd_data, layer=1)
        _, _, dpwr = detector.scan(clean_x=clean_data, adv_x=clean_data)
        assert dpwr > 0
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("keras", "kerastf", "tensorflow2", "pytorch")
def test_subsetscannning_detector_scan_adv(art_warning, get_default_mnist_subset, get_classifier):
    (x_train, _), (x_test, _) = get_default_mnist_subset
    classifier = get_classifier()

    # Generate adversarial samples
    attacker = FastGradientMethod(classifier, eps=0.5)
    x_test_adv = attacker.generate(x_test)

    # Data for detector
    bgd_data = x_train
    clean_data = x_test
    adv_data = x_test_adv

    try:
        detector = SubsetScanningDetector(classifier, bgd_data=bgd_data, layer=1)
        _, _, dpwr = detector.scan(clean_x=clean_data, adv_x=adv_data)
        assert dpwr > 0
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("keras", "kerastf", "tensorflow2", "pytorch")
def test_subsetscannning_detector_scan_size(art_warning, get_default_mnist_subset, get_classifier):
    (x_train, _), (x_test, _) = get_default_mnist_subset
    classifier = get_classifier()

    # Generate adversarial samples
    attacker = FastGradientMethod(classifier, eps=0.5)
    x_test_adv = attacker.generate(x_test)

    # Compile training data for detector:
    bgd_data = x_train
    clean_data = x_test
    adv_data = np.concatenate((x_test, x_test_adv), axis=0)

    try:
        detector = SubsetScanningDetector(classifier, bgd_data=bgd_data, layer=1)
        _, _, dpwr = detector.scan(clean_x=clean_data, adv_x=adv_data, clean_size=85, adv_size=15)
        assert dpwr > 0
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("keras", "kerastf", "tensorflow2", "pytorch")
def test_subsetscannning_detector_detect(art_warning, get_default_mnist_subset, get_classifier):
    (x_train, _), (x_test, _) = get_default_mnist_subset
    classifier = get_classifier()

    # Generate adversarial samples
    attacker = FastGradientMethod(classifier, eps=0.5)
    x_test_adv = attacker.generate(x_test)

    # Data for detector
    bgd_data = x_train
    adv_data = x_test_adv

    try:
        detector = SubsetScanningDetector(classifier, bgd_data=bgd_data, layer=1)
        _, is_adversarial = detector.detect(adv_data)
        assert len(is_adversarial) == len(adv_data)
    except ARTTestException as e:
        art_warning(e)
