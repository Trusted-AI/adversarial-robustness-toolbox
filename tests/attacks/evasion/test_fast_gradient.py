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

from tests.utils import ExpectedValue
from tests.attacks.utils import backend_check_adverse_values, backend_test_defended_images
from tests.attacks.utils import backend_test_random_initialisation_images, backend_targeted_images
from tests.attacks.utils import backend_targeted_tabular, backend_untargeted_tabular, backend_masked_images
from tests.attacks.utils import backend_test_classifier_type_check_fail

logger = logging.getLogger(__name__)


@pytest.fixture()
def fix_get_mnist_subset(get_mnist_dataset):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_mnist_dataset
    n_train = 100
    n_test = 11
    yield x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test]


@pytest.mark.framework_agnostic
def test_classifier_defended_images(fix_get_mnist_subset, image_dl_estimator_for_attack):

    classifier_list = image_dl_estimator_for_attack(FastGradientMethod, defended=True)
    # TODO this if statement must be removed once we have a classifier for both image and tabular data
    if classifier_list is None:
        logging.warning("Couldn't perform  this test because no classifier is defined")
        return

    for classifier in classifier_list:
        attack = FastGradientMethod(classifier, eps=1, batch_size=128)
        backend_test_defended_images(attack, fix_get_mnist_subset)


@pytest.mark.framework_agnostic
def test_random_initialisation_images(fix_get_mnist_subset, image_dl_estimator_for_attack):
    classifier_list = image_dl_estimator_for_attack(FastGradientMethod)
    # TODO this if statement must be removed once we have a classifier for both image and tabular data
    if classifier_list is None:
        logging.warning("Couldn't perform  this test because no classifier is defined")
        return

    for classifier in classifier_list:
        attack = FastGradientMethod(classifier, num_random_init=3)
        backend_test_random_initialisation_images(attack, fix_get_mnist_subset)


@pytest.mark.framework_agnostic
def test_targeted_images(fix_get_mnist_subset, image_dl_estimator_for_attack):
    classifier_list = image_dl_estimator_for_attack(FastGradientMethod)
    # TODO this if statement must be removed once we have a classifier for both image and tabular data
    if classifier_list is None:
        logging.warning("Couldn't perform  this test because no classifier is defined")
        return

    for classifier in classifier_list:
        attack = FastGradientMethod(classifier, eps=1.0, targeted=True)
        attack_params = {"minimal": True, "eps_step": 0.01, "eps": 1.0}
        attack.set_params(**attack_params)

        backend_targeted_images(attack, fix_get_mnist_subset)


@pytest.mark.framework_agnostic
def test_masked_images(fix_get_mnist_subset, image_dl_estimator_for_attack):
    classifier_list = image_dl_estimator_for_attack(FastGradientMethod)
    # TODO this if statement must be removed once we have a classifier for both image and tabular data
    if classifier_list is None:
        logging.warning("Couldn't perform  this test because no classifier is defined")
        return

    for classifier in classifier_list:
        attack = FastGradientMethod(classifier, eps=1.0, num_random_init=1)
        backend_masked_images(attack, fix_get_mnist_subset)


@pytest.mark.framework_agnostic
def test_minimal_perturbations_images(fix_get_mnist_subset, image_dl_estimator_for_attack):
    classifier_list = image_dl_estimator_for_attack(FastGradientMethod)
    # TODO this if statement must be removed once we have a classifier for both image and tabular data
    if classifier_list is None:
        logging.warning("Couldn't perform  this test because no classifier is defined")
        return

    for classifier in classifier_list:
        attack = FastGradientMethod(classifier, eps=1.0, batch_size=11)
        attack_params = {"minimal": True, "eps_step": 0.1, "eps": 5.0}
        attack.set_params(**attack_params)

        expected_values = {
            "x_test_mean": ExpectedValue(0.03896513, 0.01),
            "x_test_min": ExpectedValue(-0.30000000, 0.00001),
            "x_test_max": ExpectedValue(0.30000000, 0.00001),
            "y_test_pred_adv_expected": ExpectedValue(np.asarray([4, 2, 4, 7, 0, 4, 7, 2, 0, 7, 0]), 2),
        }
        backend_check_adverse_values(attack, fix_get_mnist_subset, expected_values)


@pytest.mark.parametrize("norm", [np.inf, 1, 2])
@pytest.mark.skipMlFramework("pytorch")  # temporarily skipping for pytorch until find bug fix in bounded test
@pytest.mark.framework_agnostic
def test_norm_images(norm, fix_get_mnist_subset, image_dl_estimator_for_attack):
    classifier_list = image_dl_estimator_for_attack(FastGradientMethod)
    # TODO this if statement must be removed once we have a classifier for both image and tabular data
    if classifier_list is None:
        logging.warning("Couldn't perform  this test because no classifier is defined")
        return

    elif norm == np.inf:
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

    for classifier in classifier_list:
        attack = FastGradientMethod(classifier, eps=1, norm=norm, batch_size=128)

        backend_check_adverse_values(attack, fix_get_mnist_subset, expected_values)


@pytest.mark.skipMlFramework("scikitlearn")  # temporarily skipping for scikitlearn until find bug fix in bounded test
@pytest.mark.parametrize("targeted, clipped", [(True, True), (True, False), (False, True), (False, False)])
@pytest.mark.framework_agnostic
def test_tabular(get_tabular_classifier_list, framework, get_iris_dataset, targeted, clipped):
    classifier_list = get_tabular_classifier_list(FastGradientMethod, clipped=clipped)

    if classifier_list is None:
        logging.warning("Couldn't perform  this test because no classifier is defined")
        return
    for classifier in classifier_list:
        if targeted:
            attack = FastGradientMethod(classifier, targeted=True, eps=0.1, batch_size=128)
            backend_targeted_tabular(attack, get_iris_dataset)
        else:
            attack = FastGradientMethod(classifier, eps=0.1)
            backend_untargeted_tabular(attack, get_iris_dataset, clipped=clipped)


@pytest.mark.framework_agnostic
def test_classifier_type_check_fail():
    backend_test_classifier_type_check_fail(FastGradientMethod, [BaseEstimator, LossGradientsMixin])


if __name__ == "__main__":
    pytest.cmdline.main("-q {} --mlFramework=tensorflow --durations=0".format(__file__).split(" "))
