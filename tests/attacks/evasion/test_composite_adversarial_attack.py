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
import logging

import numpy as np
import pytest

from art.attacks.evasion import CompositeAdversarialAttackPyTorch
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
    "tensorflow1", "tensorflow2", "tensorflow2v1", "keras", "non_dl_frameworks", "mxnet", "kerastf", "huggingface"
)
def test_generate(art_warning, fix_get_cifar10_subset):
    try:
        (x_train, y_train, x_test, y_test) = fix_get_cifar10_subset

        classifier = get_cifar10_image_classifier_pt(from_logits=False, load_init=True)
        attack = CompositeAdversarialAttackPyTorch(classifier)

        x_train_adv = attack.generate(x=x_train, y=y_train)
        x_test_adv = attack.generate(x=x_test, y=y_test)

        assert x_train.shape == x_train_adv.shape
        assert np.min(x_train_adv) >= 0.0
        assert np.max(x_train_adv) <= 1.0
        assert x_test.shape == x_test_adv.shape
        assert np.min(x_test_adv) >= 0.0
        assert np.max(x_test_adv) <= 1.0

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework(
    "tensorflow1", "tensorflow2", "tensorflow2v1", "keras", "non_dl_frameworks", "mxnet", "kerastf"
)
def test_check_params(art_warning):
    try:
        classifier = get_cifar10_image_classifier_pt(from_logits=False, load_init=True)

        with pytest.raises(ValueError):
            _ = CompositeAdversarialAttackPyTorch(classifier, enabled_attack=(0, 1, 2, 3, 4, 5, 6, 7))

        with pytest.raises(ValueError):
            _ = CompositeAdversarialAttackPyTorch(classifier, hue_epsilon=(-10.0, 0.0))
        with pytest.raises(ValueError):
            _ = CompositeAdversarialAttackPyTorch(classifier, hue_epsilon=(0.0, 10.0))
        with pytest.raises(TypeError):
            _ = CompositeAdversarialAttackPyTorch(classifier, hue_epsilon=(-1, 2.0))
        with pytest.raises(TypeError):
            _ = CompositeAdversarialAttackPyTorch(classifier, hue_epsilon=3.14)
        with pytest.raises(TypeError):
            _ = CompositeAdversarialAttackPyTorch(classifier, hue_epsilon=(0.0, 10.0, 20.0))
        with pytest.raises(TypeError):
            _ = CompositeAdversarialAttackPyTorch(classifier, hue_epsilon=("1.0", 2.0))

        with pytest.raises(ValueError):
            _ = CompositeAdversarialAttackPyTorch(classifier, sat_epsilon=(-10.0, 0.0))
        with pytest.raises(ValueError):
            _ = CompositeAdversarialAttackPyTorch(classifier, sat_epsilon=(0.0, -10.0))
        with pytest.raises(TypeError):
            _ = CompositeAdversarialAttackPyTorch(classifier, sat_epsilon=(1, 2.0))
        with pytest.raises(TypeError):
            _ = CompositeAdversarialAttackPyTorch(classifier, sat_epsilon=2.0)
        with pytest.raises(TypeError):
            _ = CompositeAdversarialAttackPyTorch(classifier, sat_epsilon=(0.0, 10.0, 20.0))
        with pytest.raises(TypeError):
            _ = CompositeAdversarialAttackPyTorch(classifier, sat_epsilon=("1.0", 2.0))

        with pytest.raises(ValueError):
            _ = CompositeAdversarialAttackPyTorch(classifier, rot_epsilon=(-450.0, 359.0))
        with pytest.raises(ValueError):
            _ = CompositeAdversarialAttackPyTorch(classifier, rot_epsilon=(10.0, -10.0))
        with pytest.raises(TypeError):
            _ = CompositeAdversarialAttackPyTorch(classifier, rot_epsilon=(1.0, 2))
        with pytest.raises(TypeError):
            _ = CompositeAdversarialAttackPyTorch(classifier, rot_epsilon=10)
        with pytest.raises(TypeError):
            _ = CompositeAdversarialAttackPyTorch(classifier, rot_epsilon=(0.0, 10.0, 20.0))
        with pytest.raises(TypeError):
            _ = CompositeAdversarialAttackPyTorch(classifier, rot_epsilon=("10", 20.0))

        with pytest.raises(ValueError):
            _ = CompositeAdversarialAttackPyTorch(classifier, bri_epsilon=(-10.0, 0.0))
        with pytest.raises(ValueError):
            _ = CompositeAdversarialAttackPyTorch(classifier, bri_epsilon=(0.0, 10.0))
        with pytest.raises(TypeError):
            _ = CompositeAdversarialAttackPyTorch(classifier, bri_epsilon=(-1, 1.0))
        with pytest.raises(TypeError):
            _ = CompositeAdversarialAttackPyTorch(classifier, bri_epsilon=1.0)
        with pytest.raises(TypeError):
            _ = CompositeAdversarialAttackPyTorch(classifier, bri_epsilon=(0.0, 10.0, 20.0))
        with pytest.raises(TypeError):
            _ = CompositeAdversarialAttackPyTorch(classifier, bri_epsilon=("1.0", 2.0))

        with pytest.raises(ValueError):
            _ = CompositeAdversarialAttackPyTorch(classifier, con_epsilon=(-10.0, 10.0))
        with pytest.raises(ValueError):
            _ = CompositeAdversarialAttackPyTorch(classifier, con_epsilon=(0.0, -10.0))
        with pytest.raises(TypeError):
            _ = CompositeAdversarialAttackPyTorch(classifier, con_epsilon=(1, 2.0))
        with pytest.raises(TypeError):
            _ = CompositeAdversarialAttackPyTorch(classifier, con_epsilon=2.0)
        with pytest.raises(TypeError):
            _ = CompositeAdversarialAttackPyTorch(classifier, con_epsilon=(0.0, 10.0, 20.0))
        with pytest.raises(TypeError):
            _ = CompositeAdversarialAttackPyTorch(classifier, con_epsilon=("1.0", 2.0))

        with pytest.raises(ValueError):
            _ = CompositeAdversarialAttackPyTorch(classifier, pgd_epsilon=(-0.5, 2.0))
        with pytest.raises(ValueError):
            _ = CompositeAdversarialAttackPyTorch(classifier, pgd_epsilon=(8 / 255, -8 / 255))
        with pytest.raises(TypeError):
            _ = CompositeAdversarialAttackPyTorch(classifier, pgd_epsilon=(-2, 1))
        with pytest.raises(TypeError):
            _ = CompositeAdversarialAttackPyTorch(classifier, pgd_epsilon=8 / 255)
        with pytest.raises(TypeError):
            _ = CompositeAdversarialAttackPyTorch(classifier, pgd_epsilon=(0.0, 10.0, 20.0))
        with pytest.raises(TypeError):
            _ = CompositeAdversarialAttackPyTorch(classifier, pgd_epsilon=("2/255", 3 / 255))

        with pytest.raises(TypeError):
            _ = CompositeAdversarialAttackPyTorch(classifier, early_stop="true")
        with pytest.raises(TypeError):
            _ = CompositeAdversarialAttackPyTorch(classifier, early_stop=1)

        with pytest.raises(TypeError):
            _ = CompositeAdversarialAttackPyTorch(classifier, max_iter="max")
        with pytest.raises(ValueError):
            _ = CompositeAdversarialAttackPyTorch(classifier, max_iter=-5)
        with pytest.raises(TypeError):
            _ = CompositeAdversarialAttackPyTorch(classifier, max_iter=2.5)

        with pytest.raises(TypeError):
            _ = CompositeAdversarialAttackPyTorch(classifier, max_inner_iter="max")
        with pytest.raises(ValueError):
            _ = CompositeAdversarialAttackPyTorch(classifier, max_inner_iter=-5)
        with pytest.raises(TypeError):
            _ = CompositeAdversarialAttackPyTorch(classifier, max_inner_iter=2.5)

        with pytest.raises(ValueError):
            _ = CompositeAdversarialAttackPyTorch(classifier, attack_order="schedule")

        with pytest.raises(ValueError):
            _ = CompositeAdversarialAttackPyTorch(classifier, batch_size=-1)

        with pytest.raises(TypeError):
            _ = CompositeAdversarialAttackPyTorch(classifier, verbose="true")

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_classifier_type_check_fail(art_warning):
    try:
        backend_test_classifier_type_check_fail(
            CompositeAdversarialAttackPyTorch, [BaseEstimator, LossGradientsMixin, ClassifierMixin]
        )
    except ARTTestException as e:
        art_warning(e)
