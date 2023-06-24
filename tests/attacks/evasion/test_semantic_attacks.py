# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2021
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
import logging

import numpy as np
import pytest
import torch

from art.attacks.evasion.semantic_attacks import (
    HueGradientPyTorch,
    SaturationGradientPyTorch
)
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.estimators.classification.classifier import ClassifierMixin

from tests.attacks.utils import backend_test_classifier_type_check_fail
from tests.utils import ARTTestException, get_cifar10_image_classifier_pt

logger = logging.getLogger(__name__)


@pytest.fixture()
def fix_get_cifar10_subset(get_cifar10_dataset):
    (x_train_cifar10, y_train_cifar10), (x_test_cifar10, y_test_cifar10) = get_cifar10_dataset
    n_train = 100
    n_test = 11
    yield x_train_cifar10[:n_train], y_train_cifar10[:n_train], x_test_cifar10[:n_test], y_test_cifar10[:n_test]


@pytest.mark.skip_framework(
    "tensorflow1", "tensorflow2", "tensorflow2v1", "keras", "non_dl_frameworks", "mxnet", "kerastf"
)
def test_hue_compute_factor_perturbation(art_warning, fix_get_cifar10_subset):
    try:
        (x_train, y_train, x_test, y_test) = fix_get_cifar10_subset
        factor = np.zeros((x_train.shape[0],)).astype(np.float32)

        assert np.min(x_test) >= 0.0
        assert np.max(x_test) <= 1.0

        classifier = get_cifar10_image_classifier_pt(from_logits=False, load_init=True)
        attack = HueGradientPyTorch(classifier, verbose=False)

        gradients = attack._compute_factor_perturbation(
            factor=torch.from_numpy(factor),
            x=torch.from_numpy(x_train),
            y=torch.from_numpy(y_train),
            mask=None
        )

        assert gradients.shape == factor.shape

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework(
    "tensorflow1", "tensorflow2", "tensorflow2v1", "keras", "non_dl_frameworks", "mxnet", "kerastf"
)
def test_hue_generate(art_warning, fix_get_cifar10_subset):
    try:
        (x_train, y_train, x_test, y_test) = fix_get_cifar10_subset

        classifier = get_cifar10_image_classifier_pt(from_logits=False, load_init=True)
        attack = HueGradientPyTorch(classifier, verbose=False)

        x_train_adv = attack.generate(x=x_train, y=y_train)

        assert x_train.shape == x_train_adv.shape
        assert np.min(x_train_adv) >= 0.0
        assert np.max(x_train_adv) <= 1.0

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework(
    "tensorflow1", "tensorflow2", "tensorflow2v1", "keras", "non_dl_frameworks", "mxnet", "kerastf"
)
def test_hue_check_params(art_warning):
    try:
        classifier = get_cifar10_image_classifier_pt(from_logits=False, load_init=True)

        with pytest.raises(ValueError):
            _ = HueGradientPyTorch(classifier, norm=0)
        with pytest.raises(ValueError):
            _ = HueGradientPyTorch(classifier, norm=22)

        with pytest.raises(ValueError):
            _ = HueGradientPyTorch(classifier, factor_min=0, factor_max=0)
        with pytest.raises(ValueError):
            _ = HueGradientPyTorch(classifier, factor_min=2, factor_max=-2)

        with pytest.raises(ValueError):
            _ = HueGradientPyTorch(classifier, step_size="test")
        with pytest.raises(ValueError):
            _ = HueGradientPyTorch(classifier, step_size=-0.5)

        with pytest.raises(ValueError):
            _ = HueGradientPyTorch(classifier, max_iter="test")
        with pytest.raises(ValueError):
            _ = HueGradientPyTorch(classifier, max_iter=-5)

        with pytest.raises(ValueError):
            _ = HueGradientPyTorch(classifier, targeted="true")

        with pytest.raises(TypeError):
            _ = HueGradientPyTorch(classifier, num_random_init=1.0)
        with pytest.raises(ValueError):
            _ = HueGradientPyTorch(classifier, num_random_init=-1)

        with pytest.raises(ValueError):
            _ = HueGradientPyTorch(classifier, batch_size=-1)

        with pytest.raises(ValueError):
            _ = HueGradientPyTorch(classifier, verbose="true")

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_hue_classifier_type_check_fail(art_warning):
    try:
        backend_test_classifier_type_check_fail(
            HueGradientPyTorch, [BaseEstimator, LossGradientsMixin, ClassifierMixin]
        )
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework(
    "tensorflow1", "tensorflow2", "tensorflow2v1", "keras", "non_dl_frameworks", "mxnet", "kerastf"
)
def test_saturation_compute_factor_perturbation(art_warning, fix_get_cifar10_subset):
    try:
        (x_train, y_train, x_test, y_test) = fix_get_cifar10_subset
        factor = np.ones((x_train.shape[0],)).astype(np.float32)

        assert np.min(x_test) >= 0.0
        assert np.max(x_test) <= 1.0

        classifier = get_cifar10_image_classifier_pt(from_logits=False, load_init=True)
        attack = SaturationGradientPyTorch(classifier, verbose=False)

        gradients = attack._compute_factor_perturbation(
            factor=torch.from_numpy(factor),
            x=torch.from_numpy(x_train),
            y=torch.from_numpy(y_train),
            mask=None
        )

        assert gradients.shape == factor.shape

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework(
    "tensorflow1", "tensorflow2", "tensorflow2v1", "keras", "non_dl_frameworks", "mxnet", "kerastf"
)
def test_saturation_generate(art_warning, fix_get_cifar10_subset):
    try:
        (x_train, y_train, x_test, y_test) = fix_get_cifar10_subset

        classifier = get_cifar10_image_classifier_pt(from_logits=False, load_init=True)
        attack = SaturationGradientPyTorch(classifier, verbose=False)

        x_train_adv = attack.generate(x=x_train, y=y_train)

        assert x_train.shape == x_train_adv.shape
        assert np.min(x_train_adv) >= 0.0
        assert np.max(x_train_adv) <= 1.0

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework(
    "tensorflow1", "tensorflow2", "tensorflow2v1", "keras", "non_dl_frameworks", "mxnet", "kerastf"
)
def test_saturation_check_params(art_warning):
    try:
        classifier = get_cifar10_image_classifier_pt(from_logits=False, load_init=True)

        with pytest.raises(ValueError):
            _ = SaturationGradientPyTorch(classifier, norm=0)
        with pytest.raises(ValueError):
            _ = SaturationGradientPyTorch(classifier, norm=22)

        with pytest.raises(ValueError):
            _ = SaturationGradientPyTorch(classifier, factor_min=0, factor_max=0)
        with pytest.raises(ValueError):
            _ = SaturationGradientPyTorch(classifier, factor_min=-2, factor_max=2)

        with pytest.raises(ValueError):
            _ = SaturationGradientPyTorch(classifier, step_size="test")
        with pytest.raises(ValueError):
            _ = SaturationGradientPyTorch(classifier, step_size=-0.5)

        with pytest.raises(ValueError):
            _ = SaturationGradientPyTorch(classifier, max_iter="test")
        with pytest.raises(ValueError):
            _ = SaturationGradientPyTorch(classifier, max_iter=-5)

        with pytest.raises(ValueError):
            _ = SaturationGradientPyTorch(classifier, targeted="true")

        with pytest.raises(TypeError):
            _ = SaturationGradientPyTorch(classifier, num_random_init=1.0)
        with pytest.raises(ValueError):
            _ = SaturationGradientPyTorch(classifier, num_random_init=-1)

        with pytest.raises(ValueError):
            _ = SaturationGradientPyTorch(classifier, batch_size=-1)

        with pytest.raises(ValueError):
            _ = SaturationGradientPyTorch(classifier, verbose="true")

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_saturation_classifier_type_check_fail(art_warning):
    try:
        backend_test_classifier_type_check_fail(
            SaturationGradientPyTorch, [BaseEstimator, LossGradientsMixin, ClassifierMixin]
        )
    except ARTTestException as e:
        art_warning(e)
