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
import logging
import pytest

import numpy as np

from art.attacks.evasion import SquareAttack
from art.estimators.estimator import BaseEstimator, NeuralNetworkMixin

from tests.attacks.utils import backend_test_classifier_type_check_fail
from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.fixture()
def fix_get_mnist_subset(get_mnist_dataset):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_mnist_dataset
    n_train = 100
    n_test = 10
    yield x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test]


@pytest.mark.framework_agnostic
@pytest.mark.parametrize("norm", [2, "inf"])
def test_generate(art_warning, fix_get_mnist_subset, image_dl_estimator_for_attack, norm):
    try:
        classifier = image_dl_estimator_for_attack(SquareAttack)

        attack = SquareAttack(
            estimator=classifier, norm=norm, max_iter=5, eps=0.3, p_init=0.8, nb_restarts=1, verbose=False
        )

        (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset

        x_train_mnist_adv = attack.generate(x=x_train_mnist, y=y_train_mnist)

        if norm == "inf":
            expected_mean = 0.053533513
            expected_max = 0.3
        elif norm == 2:
            expected_mean = 0.00073682
            expected_max = 0.25

        assert np.mean(np.abs(x_train_mnist_adv - x_train_mnist)) == pytest.approx(expected_mean, abs=0.025)
        assert np.max(np.abs(x_train_mnist_adv - x_train_mnist)) == pytest.approx(expected_max, abs=0.05)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_classifier_type_check_fail(art_warning):
    try:
        backend_test_classifier_type_check_fail(SquareAttack, [BaseEstimator, NeuralNetworkMixin])
    except ARTTestException as e:
        art_warning(e)
