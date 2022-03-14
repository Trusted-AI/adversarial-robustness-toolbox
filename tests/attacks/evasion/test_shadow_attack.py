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

from art.attacks.evasion import ShadowAttack
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.estimators.classification.classifier import ClassifierMixin

from tests.attacks.utils import backend_test_classifier_type_check_fail
from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.fixture()
def fix_get_mnist_subset(get_mnist_dataset):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_mnist_dataset
    n_train = 100
    n_test = 11
    yield x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test]


@pytest.mark.only_with_platform("pytorch", "tensorflow2")
def test_generate(art_warning, fix_get_mnist_subset, image_dl_estimator_for_attack):
    try:
        classifier = image_dl_estimator_for_attack(ShadowAttack)
        attack = ShadowAttack(
            estimator=classifier,
            sigma=0.5,
            nb_steps=3,
            learning_rate=0.1,
            lambda_tv=0.3,
            lambda_c=1.0,
            lambda_s=0.5,
            batch_size=32,
            targeted=True,
            verbose=False,
        )

        (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset

        x_train_mnist_adv = attack.generate(x=x_train_mnist[0:1], y=y_train_mnist[0:1])

        assert np.max(np.abs(x_train_mnist_adv - x_train_mnist[0:1])) == pytest.approx(0.38116083, abs=0.06)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch", "tensorflow2")
def test_get_regularisation_loss_gradients(art_warning, fix_get_mnist_subset, image_dl_estimator_for_attack):
    try:
        classifier = image_dl_estimator_for_attack(ShadowAttack)

        attack = ShadowAttack(
            estimator=classifier,
            sigma=0.5,
            nb_steps=3,
            learning_rate=0.1,
            lambda_tv=0.3,
            lambda_c=1.0,
            lambda_s=0.5,
            batch_size=32,
            targeted=True,
            verbose=False,
        )

        (x_train_mnist, _, _, _) = fix_get_mnist_subset

        gradients = attack._get_regularisation_loss_gradients(x_train_mnist[0:1])

        gradients_expected = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -0.27294118,
                -0.36906054,
                0.83799828,
                0.40741005,
                0.65682181,
                -0.13141348,
                -0.39729583,
                -0.12235294,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )

        if attack.framework == "pytorch":
            np.testing.assert_array_almost_equal(gradients[0, 0, 14, :], gradients_expected, decimal=3)
        else:
            np.testing.assert_array_almost_equal(gradients[0, 14, :, 0], gradients_expected, decimal=3)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_check_params(art_warning, image_dl_estimator_for_attack):
    try:
        classifier = image_dl_estimator_for_attack(ShadowAttack)

        with pytest.raises(ValueError):
            _ = ShadowAttack(classifier, sigma="test")
        with pytest.raises(ValueError):
            _ = ShadowAttack(classifier, sigma=-0.5)

        with pytest.raises(ValueError):
            _ = ShadowAttack(classifier, nb_steps=0.5)
        with pytest.raises(ValueError):
            _ = ShadowAttack(classifier, nb_steps=-5)

        with pytest.raises(ValueError):
            _ = ShadowAttack(classifier, learning_rate=5)
        with pytest.raises(ValueError):
            _ = ShadowAttack(classifier, learning_rate=-5.0)

        with pytest.raises(ValueError):
            _ = ShadowAttack(classifier, lambda_tv=5)
        with pytest.raises(ValueError):
            _ = ShadowAttack(classifier, lambda_tv=-5.0)

        with pytest.raises(ValueError):
            _ = ShadowAttack(classifier, lambda_c=5)
        with pytest.raises(ValueError):
            _ = ShadowAttack(classifier, lambda_c=-5.0)

        with pytest.raises(ValueError):
            _ = ShadowAttack(classifier, lambda_s=5)
        with pytest.raises(ValueError):
            _ = ShadowAttack(classifier, lambda_s=-5.0)

        with pytest.raises(ValueError):
            _ = ShadowAttack(classifier, batch_size=5.0)
        with pytest.raises(ValueError):
            _ = ShadowAttack(classifier, batch_size=-5)

        with pytest.raises(ValueError):
            _ = ShadowAttack(classifier, targeted=5.0)

        with pytest.raises(ValueError):
            _ = ShadowAttack(classifier, verbose=5.0)

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_classifier_type_check_fail(art_warning):
    try:
        backend_test_classifier_type_check_fail(ShadowAttack, [BaseEstimator, LossGradientsMixin, ClassifierMixin])
    except ARTTestException as e:
        art_warning(e)
