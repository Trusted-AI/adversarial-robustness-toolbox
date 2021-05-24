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

import pytest
import logging

from art.attacks.evasion import BoundaryAttack
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin

from tests.attacks.utils import backend_targeted_tabular, backend_untargeted_tabular, backend_targeted_images
from tests.attacks.utils import back_end_untargeted_images, backend_test_classifier_type_check_fail
from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.fixture()
def fix_get_mnist_subset(get_mnist_dataset):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_mnist_dataset
    n_train = 10
    n_test = 10
    yield x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test]


@pytest.mark.framework_agnostic
@pytest.mark.parametrize("clipped_classifier, targeted", [(True, True), (True, False), (False, True), (False, False)])
def test_tabular(art_warning, tabular_dl_estimator, framework, get_iris_dataset, clipped_classifier, targeted):
    try:
        classifier = tabular_dl_estimator(clipped=clipped_classifier)
        attack = BoundaryAttack(classifier, targeted=targeted, max_iter=10, verbose=False)
        if targeted:
            backend_targeted_tabular(attack, get_iris_dataset)
        else:
            backend_untargeted_tabular(attack, get_iris_dataset, clipped=clipped_classifier)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
@pytest.mark.parametrize("targeted", [True, False])
def test_images(art_warning, fix_get_mnist_subset, image_dl_estimator_for_attack, framework, targeted):
    try:
        classifier = image_dl_estimator_for_attack(BoundaryAttack)
        attack = BoundaryAttack(estimator=classifier, targeted=targeted, max_iter=20, verbose=False)
        if targeted:
            backend_targeted_images(attack, fix_get_mnist_subset)
        else:
            back_end_untargeted_images(attack, fix_get_mnist_subset, framework)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_classifier_type_check_fail(art_warning):
    try:
        backend_test_classifier_type_check_fail(BoundaryAttack, [BaseEstimator, ClassifierMixin])
    except ARTTestException as e:
        art_warning(e)
