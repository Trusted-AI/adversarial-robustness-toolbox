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

import logging

import numpy as np
import pytest

from art.attacks.evasion import MomentumIterativeMethod
from art.estimators.estimator import BaseEstimator, LossGradientsMixin

from tests.attacks.utils import backend_test_classifier_type_check_fail
from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.fixture()
def fix_get_mnist_subset(get_mnist_dataset):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_mnist_dataset
    n_train = 100
    n_test = 11
    yield x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test]


@pytest.mark.skip_framework("tensorflow")  # See issue #2439
def test_images(art_warning, fix_get_mnist_subset, image_dl_estimator_for_attack, framework):
    try:
        (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset
        classifier = image_dl_estimator_for_attack(MomentumIterativeMethod)

        attack = MomentumIterativeMethod(classifier, eps=0.3, eps_step=0.1, decay=1.0, max_iter=10)
        x_train_mnist_adv = attack.generate(x=x_train_mnist)

        assert np.mean(x_train_mnist) == pytest.approx(0.12659499049186707, 0.01)
        assert np.max(np.abs(x_train_mnist - x_train_mnist_adv)) == pytest.approx(0.3)
        expected_mean_diff = pytest.approx(0.1288, 0.003)
        assert np.mean(np.abs(x_train_mnist - x_train_mnist_adv)) == expected_mean_diff

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("tensorflow")  # See issue #2439
def test_images_targeted(art_warning, fix_get_mnist_subset, image_dl_estimator_for_attack, framework):
    try:
        (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset
        classifier = image_dl_estimator_for_attack(MomentumIterativeMethod)

        attack = MomentumIterativeMethod(classifier, eps=0.3, eps_step=0.1, decay=1.0, max_iter=10)
        x_train_mnist_adv = attack.generate(x=x_train_mnist, y=y_train_mnist)

        assert np.max(np.abs(x_train_mnist - x_train_mnist_adv)) == pytest.approx(0.3)
        expected_mean_diff = pytest.approx(0.1077, 0.01)
        assert np.mean(np.abs(x_train_mnist - x_train_mnist_adv)) == expected_mean_diff

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_check_params(art_warning, image_dl_estimator_for_attack):
    try:
        classifier = image_dl_estimator_for_attack(MomentumIterativeMethod)

        with pytest.raises(ValueError):
            _ = MomentumIterativeMethod(classifier, norm=0)

        with pytest.raises(ValueError):
            _ = MomentumIterativeMethod(classifier, eps=-1)

        with pytest.raises(ValueError):
            _ = MomentumIterativeMethod(classifier, eps=1, eps_step=-1)

        with pytest.raises(ValueError):
            _ = MomentumIterativeMethod(classifier, decay=-1)

        with pytest.raises(ValueError):
            _ = MomentumIterativeMethod(classifier, max_iter=-1)

        with pytest.raises(ValueError):
            _ = MomentumIterativeMethod(classifier, targeted="true")

        with pytest.raises(ValueError):
            _ = MomentumIterativeMethod(classifier, batch_size=0)

        with pytest.raises(ValueError):
            _ = MomentumIterativeMethod(classifier, verbose="true")

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_classifier_type_check_fail(art_warning):
    try:
        backend_test_classifier_type_check_fail(MomentumIterativeMethod, [BaseEstimator, LossGradientsMixin])
    except ARTTestException as e:
        art_warning(e)
