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

from art.attacks.evasion import FastGradientMethod
from art.estimators.estimator import BaseEstimator, LossGradientsMixin

from tests.attacks.utils import backend_check_adverse_values, backend_test_defended_images
from tests.attacks.utils import backend_test_random_initialisation_images, backend_targeted_images
from tests.attacks.utils import backend_targeted_tabular, backend_untargeted_tabular, backend_masked_images
from tests.attacks.utils import backend_test_classifier_type_check_fail
from tests.utils import ExpectedValue, ARTTestException

logger = logging.getLogger(__name__)


@pytest.fixture()
def fix_get_mnist_subset(get_mnist_dataset):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_mnist_dataset
    n_train = 100
    n_test = 11
    yield x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test]


# currently NOT setting this test as framework_agnostic since no TensorFlow implementation
# of the defended classifier exists
def test_classifier_defended_images(art_warning, fix_get_mnist_subset, image_dl_estimator_for_attack):
    try:
        classifier = image_dl_estimator_for_attack(FastGradientMethod, defended=True)
        attack = FastGradientMethod(classifier, eps=1.0, batch_size=128)
        backend_test_defended_images(attack, fix_get_mnist_subset)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_random_initialisation_images(art_warning, fix_get_mnist_subset, image_dl_estimator_for_attack):
    try:
        classifier = image_dl_estimator_for_attack(FastGradientMethod)
        attack = FastGradientMethod(classifier, num_random_init=3)
        backend_test_random_initialisation_images(attack, fix_get_mnist_subset)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_targeted_images(art_warning, fix_get_mnist_subset, image_dl_estimator_for_attack):
    try:
        classifier = image_dl_estimator_for_attack(FastGradientMethod)
        attack = FastGradientMethod(classifier, eps=1.0, targeted=True)
        attack_params = {"minimal": True, "eps_step": 0.01, "eps": 1.0}
        attack.set_params(**attack_params)

        backend_targeted_images(attack, fix_get_mnist_subset)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_masked_images(art_warning, fix_get_mnist_subset, image_dl_estimator_for_attack):
    try:
        classifier = image_dl_estimator_for_attack(FastGradientMethod)
        attack = FastGradientMethod(classifier, eps=1.0, num_random_init=1)
        backend_masked_images(attack, fix_get_mnist_subset)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_minimal_perturbations_images(art_warning, fix_get_mnist_subset, image_dl_estimator_for_attack):
    try:
        classifier = image_dl_estimator_for_attack(FastGradientMethod)

        expected_values = {
            "x_test_mean": ExpectedValue(0.03896513, 0.01),
            "x_test_min": ExpectedValue(-0.30000000, 0.00001),
            "x_test_max": ExpectedValue(0.30000000, 0.00001),
            "y_test_pred_adv_expected": ExpectedValue(np.asarray([4, 2, 4, 7, 0, 4, 7, 2, 0, 7, 0]), 2),
        }

        attack = FastGradientMethod(classifier, eps=1.0, batch_size=11)

        # Test eps of float type
        attack_params = {"minimal": True, "eps_step": 0.1, "eps": 5.0}
        attack.set_params(**attack_params)

        backend_check_adverse_values(attack, fix_get_mnist_subset, expected_values)

        # Test eps of array type 1
        (_, _, x_test_mnist, _) = fix_get_mnist_subset
        eps = np.ones(shape=x_test_mnist.shape) * 5.0
        eps_step = np.ones_like(eps) * 0.1

        attack_params = {"minimal": True, "eps_step": eps_step, "eps": eps}
        attack.set_params(**attack_params)

        backend_check_adverse_values(attack, fix_get_mnist_subset, expected_values)

        # Test eps of array type 2
        eps = np.ones(shape=x_test_mnist.shape[1:]) * 5.0
        eps_step = np.ones_like(eps) * 0.1

        attack_params = {"minimal": True, "eps_step": eps_step, "eps": eps}
        attack.set_params(**attack_params)

        backend_check_adverse_values(attack, fix_get_mnist_subset, expected_values)

        # Test eps of array type 3
        eps = np.ones(shape=x_test_mnist.shape[2:]) * 5.0
        eps_step = np.ones_like(eps) * 0.1

        attack_params = {"minimal": True, "eps_step": eps_step, "eps": eps}
        attack.set_params(**attack_params)

        backend_check_adverse_values(attack, fix_get_mnist_subset, expected_values)

        # Test eps of array type 4
        eps = np.ones(shape=x_test_mnist.shape[3:]) * 5.0
        eps_step = np.ones_like(eps) * 0.1

        attack_params = {"minimal": True, "eps_step": eps_step, "eps": eps}
        attack.set_params(**attack_params)

        backend_check_adverse_values(attack, fix_get_mnist_subset, expected_values)

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.parametrize("norm", [np.inf, 1, 2])
@pytest.mark.skip_framework("pytorch")  # temporarily skipping for pytorch until find bug fix in bounded test
@pytest.mark.framework_agnostic
def test_norm_images(art_warning, norm, fix_get_mnist_subset, image_dl_estimator_for_attack):
    try:
        classifier = image_dl_estimator_for_attack(FastGradientMethod)

        if norm == np.inf:
            expected_values = {
                "x_test_mean": ExpectedValue(0.2346725, 0.002),
                "x_test_min": ExpectedValue(-1.0, 0.00001),
                "x_test_max": ExpectedValue(1.0, 0.00001),
                "y_test_pred_adv_expected": ExpectedValue(np.asarray([4, 4, 4, 7, 7, 4, 7, 2, 2, 3, 0]), 2),
            }

        elif norm == 1:
            expected_values = {
                "x_test_mean": ExpectedValue(0.00051374, 0.002),
                "x_test_min": ExpectedValue(-0.01486498, 0.001),
                "x_test_max": ExpectedValue(0.014761963, 0.001),
                "y_test_pred_adv_expected": ExpectedValue(np.asarray([7, 1, 1, 4, 4, 1, 4, 4, 4, 4, 4]), 4),
            }
        elif norm == 2:
            expected_values = {
                "x_test_mean": ExpectedValue(0.007636416, 0.001),
                "x_test_min": ExpectedValue(-0.211054801, 0.001),
                "x_test_max": ExpectedValue(0.209592223, 0.001),
                "y_test_pred_adv_expected": ExpectedValue(np.asarray([7, 2, 4, 4, 4, 7, 7, 4, 0, 4, 4]), 2),
            }

        attack = FastGradientMethod(classifier, eps=1.0, norm=norm, batch_size=128)

        backend_check_adverse_values(attack, fix_get_mnist_subset, expected_values)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("scikitlearn")  # temporarily skipping for scikitlearn until find bug fix in bounded test
@pytest.mark.parametrize("targeted, clipped", [(True, True), (True, False), (False, True), (False, False)])
@pytest.mark.framework_agnostic
def test_tabular(art_warning, tabular_dl_estimator, framework, get_iris_dataset, targeted, clipped):
    try:
        classifier = tabular_dl_estimator(clipped=clipped)

        if targeted:
            attack = FastGradientMethod(classifier, targeted=True, eps=0.1, batch_size=128)
            backend_targeted_tabular(attack, get_iris_dataset)
        else:
            attack = FastGradientMethod(classifier, eps=0.1)
            backend_untargeted_tabular(attack, get_iris_dataset, clipped=clipped)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_non_classification(art_warning, fix_get_mnist_subset, image_dl_estimator_for_attack, fix_get_rcnn):
    try:
        classifier = fix_get_rcnn
        attack = FastGradientMethod(classifier, num_random_init=3)
        backend_test_random_initialisation_images(attack, fix_get_mnist_subset)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_check_params(art_warning, image_dl_estimator_for_attack):
    try:
        classifier = image_dl_estimator_for_attack(FastGradientMethod)

        with pytest.raises(ValueError):
            _ = FastGradientMethod(classifier, norm=0)

        with pytest.raises(ValueError):
            _ = FastGradientMethod(classifier, eps=-1, eps_step=1)
        with pytest.raises(ValueError):
            _ = FastGradientMethod(classifier, eps=np.array([-1, -1, -1]), eps_step=np.array([1, 1, 1]))

        with pytest.raises(ValueError):
            _ = FastGradientMethod(classifier, eps=1, eps_step=-1)
        with pytest.raises(ValueError):
            _ = FastGradientMethod(classifier, eps=np.array([1, 1, 1]), eps_step=np.array([-1, -1, -1]))

        with pytest.raises(TypeError):
            _ = FastGradientMethod(classifier, eps=1, eps_step=np.array([1, 1, 1]))

        with pytest.raises(ValueError):
            _ = FastGradientMethod(classifier, targeted="true")

        with pytest.raises(TypeError):
            _ = FastGradientMethod(classifier, num_random_init=1.0)
        with pytest.raises(ValueError):
            _ = FastGradientMethod(classifier, num_random_init=-1)

        with pytest.raises(ValueError):
            _ = FastGradientMethod(classifier, batch_size=-1)

        with pytest.raises(ValueError):
            _ = FastGradientMethod(classifier, minimal="true")

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_classifier_type_check_fail(art_warning):
    try:
        backend_test_classifier_type_check_fail(FastGradientMethod, [BaseEstimator, LossGradientsMixin])
    except ARTTestException as e:
        art_warning(e)
