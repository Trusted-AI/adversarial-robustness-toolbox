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

from art.attacks.evasion import FeatureAdversariesTensorFlowV2
from art.estimators.estimator import BaseEstimator, NeuralNetworkMixin
from tests.attacks.utils import backend_test_classifier_type_check_fail
from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.fixture()
def fix_get_mnist_subset(get_mnist_dataset):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_mnist_dataset
    n_train = 100
    n_test = 11
    yield x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test]


@pytest.mark.skip_framework("tensorflow1", "keras", "kerastf", "mxnet", "non_dl_frameworks", "pytorch")
def test_images_pgd(art_warning, fix_get_mnist_subset, image_dl_estimator_for_attack):
    try:
        (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset

        classifier = image_dl_estimator_for_attack(FeatureAdversariesTensorFlowV2)

        attack = FeatureAdversariesTensorFlowV2(
            classifier, delta=1.0, layer=1, batch_size=32, step_size=0.05, max_iter=2, random_start=False
        )
        x_train_mnist_adv = attack.generate(x=x_train_mnist[0:3], y=x_test_mnist[0:3])
        assert np.mean(x_train_mnist[0:3]) == pytest.approx(0.13015705, 0.01)
        assert np.mean(np.abs(x_train_mnist_adv - x_train_mnist[0:3])) != 0.0
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("tensorflow1", "keras", "kerastf", "mxnet", "non_dl_frameworks", "pytorch")
def test_images_unconstrained_adam(art_warning, fix_get_mnist_subset, image_dl_estimator_for_attack):
    try:
        import tensorflow as tf

        (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset

        classifier = image_dl_estimator_for_attack(FeatureAdversariesTensorFlowV2)

        attack = FeatureAdversariesTensorFlowV2(
            classifier, delta=1.0, layer=1, batch_size=32, optimizer=tf.optimizers.Adam, max_iter=1, random_start=False
        )
        x_train_mnist_adv = attack.generate(x=x_train_mnist[0:3], y=x_test_mnist[0:3])
        assert np.mean(x_train_mnist[0:3]) == pytest.approx(0.13015705, 0.01)
        assert np.mean(np.abs(x_train_mnist_adv - x_train_mnist[0:3])) != 0.0
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_classifier_type_check_fail(art_warning):
    try:
        backend_test_classifier_type_check_fail(
            FeatureAdversariesTensorFlowV2, [BaseEstimator, NeuralNetworkMixin], delta=1.0
        )
    except ARTTestException as e:
        art_warning(e)
