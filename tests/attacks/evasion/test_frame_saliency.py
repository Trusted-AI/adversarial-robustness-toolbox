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
import numpy as np
import pytest

from art.attacks.evasion import FastGradientMethod, FrameSaliencyAttack
from art.estimators.estimator import BaseEstimator, LossGradientsMixin

from tests.utils import ExpectedValue
from tests.attacks.utils import backend_check_adverse_values, backend_check_adverse_frames
from tests.attacks.utils import backend_test_classifier_type_check_fail
from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.fixture()
def fix_get_mnist_subset(get_mnist_dataset):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_mnist_dataset
    n_train = 100
    n_test = 11
    yield x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test]


@pytest.mark.skip_framework("pytorch")
@pytest.mark.framework_agnostic
def test_one_shot(art_warning, fix_get_mnist_subset, image_dl_estimator_for_attack):
    try:
        classifier = image_dl_estimator_for_attack(FastGradientMethod)

        # for the one-shot method, frame saliency attack should resort to plain FastGradientMethod
        expected_values = {
            "x_test_mean": ExpectedValue(0.2346725, 0.002),
            "x_test_min": ExpectedValue(-1.0, 0.00001),
            "x_test_max": ExpectedValue(1.0, 0.00001),
            "y_test_pred_adv_expected": ExpectedValue(np.asarray([4, 4, 4, 7, 7, 4, 7, 2, 2, 3, 0]), 2),
        }

        attacker = FastGradientMethod(classifier, eps=1.0, batch_size=128)
        attack = FrameSaliencyAttack(classifier, attacker, "one_shot")
        backend_check_adverse_values(attack, fix_get_mnist_subset, expected_values)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("pytorch")
@pytest.mark.framework_agnostic
def test_iterative_saliency(art_warning, fix_get_mnist_subset, image_dl_estimator_for_attack):
    try:
        classifier = image_dl_estimator_for_attack(FastGradientMethod)

        expected_values_axis_1 = {
            "nb_perturbed_frames": ExpectedValue(np.asarray([10, 1, 2, 12, 16, 1, 2, 7, 4, 11, 5]), 2)
        }

        expected_values_axis_2 = {
            "nb_perturbed_frames": ExpectedValue(np.asarray([11, 1, 2, 6, 14, 2, 2, 13, 4, 8, 4]), 2)
        }

        attacker = FastGradientMethod(classifier, eps=0.3, batch_size=128)
        attack = FrameSaliencyAttack(classifier, attacker, "iterative_saliency")
        backend_check_adverse_frames(attack, fix_get_mnist_subset, expected_values_axis_1)

        # test with non-default frame index:
        attack = FrameSaliencyAttack(classifier, attacker, "iterative_saliency", frame_index=2)
        backend_check_adverse_frames(attack, fix_get_mnist_subset, expected_values_axis_2)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("pytorch")
@pytest.mark.framework_agnostic
def test_iterative_saliency_refresh(art_warning, fix_get_mnist_subset, image_dl_estimator_for_attack):
    try:
        classifier = image_dl_estimator_for_attack(FastGradientMethod)

        expected_values_axis_1 = {
            "nb_perturbed_frames": ExpectedValue(np.asarray([5, 1, 3, 10, 8, 1, 3, 8, 4, 7, 7]), 2)
        }

        expected_values_axis_2 = {
            "nb_perturbed_frames": ExpectedValue(np.asarray([11, 1, 2, 6, 14, 2, 2, 13, 4, 8, 4]), 2)
        }

        attacker = FastGradientMethod(classifier, eps=0.3, batch_size=128)
        attack = FrameSaliencyAttack(classifier, attacker, "iterative_saliency_refresh")
        backend_check_adverse_frames(attack, fix_get_mnist_subset, expected_values_axis_1)

        # test with non-default frame index:
        attack = FrameSaliencyAttack(classifier, attacker, "iterative_saliency", frame_index=2)
        backend_check_adverse_frames(attack, fix_get_mnist_subset, expected_values_axis_2)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_check_params(art_warning, image_dl_estimator_for_attack):
    try:
        from art.attacks.evasion import FastGradientMethod

        classifier = image_dl_estimator_for_attack(FrameSaliencyAttack, from_logits=True)

        attacker = FastGradientMethod(estimator=classifier)

        with pytest.raises(ValueError):
            _ = FrameSaliencyAttack(classifier, attacker="attack")

        with pytest.raises(ValueError):
            _ = FrameSaliencyAttack(classifier, attacker=attacker, method="test")

        with pytest.raises(ValueError):
            _ = FrameSaliencyAttack(classifier, attacker=attacker, frame_index=0)

        with pytest.raises(ValueError):
            _ = FrameSaliencyAttack(classifier, attacker=attacker, batch_size=-1)

        with pytest.raises(ValueError):
            _ = FrameSaliencyAttack(classifier, attacker=attacker, verbose="true")

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_classifier_type_check_fail(art_warning):
    try:
        backend_test_classifier_type_check_fail(FastGradientMethod, [LossGradientsMixin, BaseEstimator])
    except ARTTestException as e:
        art_warning(e)
