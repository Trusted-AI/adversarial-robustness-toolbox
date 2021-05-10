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

from art.attacks.evasion import AutoAttack
from art.attacks.evasion.auto_projected_gradient_descent import AutoProjectedGradientDescent
from art.attacks.evasion.deepfool import DeepFool
from art.attacks.evasion.square_attack import SquareAttack
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin

from tests.attacks.utils import backend_test_classifier_type_check_fail
from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.fixture()
def fix_get_mnist_subset(get_mnist_dataset):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_mnist_dataset
    n_train = 100
    n_test = 10
    yield x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test]


@pytest.mark.skip_framework("tensorflow1", "keras", "pytorch", "non_dl_frameworks", "mxnet", "kerastf")
def test_generate_default(art_warning, fix_get_mnist_subset, image_dl_estimator):
    try:
        classifier, _ = image_dl_estimator(from_logits=True)

        attack = AutoAttack(
            estimator=classifier,
            norm=np.inf,
            eps=0.3,
            eps_step=0.1,
            attacks=None,
            batch_size=32,
            estimator_orig=None,
        )

        (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset

        x_train_mnist_adv = attack.generate(x=x_train_mnist, y=y_train_mnist)

        assert np.mean(np.abs(x_train_mnist_adv - x_train_mnist)) == pytest.approx(0.0292, abs=0.105)
        assert np.max(np.abs(x_train_mnist_adv - x_train_mnist)) == pytest.approx(0.3, abs=0.05)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("tensorflow1", "keras", "pytorch", "non_dl_frameworks", "mxnet", "kerastf")
def test_generate_attacks_and_targeted(art_warning, fix_get_mnist_subset, image_dl_estimator):
    try:
        classifier, _ = image_dl_estimator(from_logits=True)

        norm = np.inf
        eps = 0.3
        eps_step = 0.1
        batch_size = 32

        attacks = list()
        attacks.append(
            AutoProjectedGradientDescent(
                estimator=classifier,
                norm=norm,
                eps=eps,
                eps_step=eps_step,
                max_iter=100,
                targeted=True,
                nb_random_init=5,
                batch_size=batch_size,
                loss_type="cross_entropy",
            )
        )
        attacks.append(
            AutoProjectedGradientDescent(
                estimator=classifier,
                norm=norm,
                eps=eps,
                eps_step=eps_step,
                max_iter=100,
                targeted=False,
                nb_random_init=5,
                batch_size=batch_size,
                loss_type="difference_logits_ratio",
            )
        )
        attacks.append(DeepFool(classifier=classifier, max_iter=100, epsilon=1e-6, nb_grads=3, batch_size=batch_size))
        attacks.append(SquareAttack(estimator=classifier, norm=norm, max_iter=5000, eps=eps, p_init=0.8, nb_restarts=5))

        (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset

        # First test with defined_attack_only=False
        attack = AutoAttack(
            estimator=classifier,
            norm=norm,
            eps=eps,
            eps_step=eps_step,
            attacks=attacks,
            batch_size=batch_size,
            estimator_orig=None,
            targeted=False,
        )

        x_train_mnist_adv = attack.generate(x=x_train_mnist, y=y_train_mnist)

        assert np.mean(np.abs(x_train_mnist_adv - x_train_mnist)) == pytest.approx(0.0182, abs=0.105)
        assert np.max(np.abs(x_train_mnist_adv - x_train_mnist)) == pytest.approx(0.3, abs=0.05)

        # Then test with defined_attack_only=True
        attack = AutoAttack(
            estimator=classifier,
            norm=norm,
            eps=eps,
            eps_step=eps_step,
            attacks=attacks,
            batch_size=batch_size,
            estimator_orig=None,
            targeted=True,
        )

        x_train_mnist_adv = attack.generate(x=x_train_mnist, y=y_train_mnist)

        assert np.mean(x_train_mnist_adv - x_train_mnist) == pytest.approx(0.0179, abs=0.0075)
        assert np.max(np.abs(x_train_mnist_adv - x_train_mnist)) == pytest.approx(eps, abs=0.005)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_classifier_type_check_fail(art_warning):
    try:
        backend_test_classifier_type_check_fail(AutoAttack, [BaseEstimator, ClassifierMixin])
    except ARTTestException as e:
        art_warning(e)
