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

from art.attacks.poisoning import GradientMatchingAttack
from art.utils import to_categorical

from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.mark.skip_framework("non_dl_frameworks", "pytorch", "mxnet")
def test_poison(art_warning, get_default_mnist_subset, image_dl_estimator):
    try:
        (x_train, y_train), (x_test, y_test) = get_default_mnist_subset
        classifier, _ = image_dl_estimator()

        class_source = 0
        class_target = 1
        index_target = np.where(y_test.argmax(axis=1)==class_source)[0][5]
        x_target = x_test[index_target:index_target+1]
        y_target = to_categorical([class_target], num_classes=10)

        attack = GradientMatchingAttack(classifier, epsilon=0.3, verbose=False)
        x_poison, y_poison = attack.poison(x_target, y_target, x_train, y_train, percent_poison=0.1)

        np.testing.assert_equal(x_poison.shape, x_train.shape)
        np.testing.assert_equal(y_poison.shape, y_train.shape)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.parametrize("params", [dict(percent_poison=-0.2), dict(percent_poison=1.2)])
@pytest.mark.skip_framework("non_dl_frameworks", "pytorch", "mxnet")
def test_failure_modes(art_warning, image_dl_estimator, params):
    try:
        (x_train, y_train), (x_test, y_test) = get_default_mnist_subset
        classifier, _ = image_dl_estimator()

        class_source = 0
        class_target = 1
        index_target = np.where(y_test.argmax(axis=1)==class_source)[0][5]
        x_target = x_test[index_target:index_target+1]
        y_target = to_categorical([class_target], num_classes=10)

        attack = GradientMatchingAttack(classifier, epsilon=0.3, verbose=False, **params)
        x_poison, y_poison = attack.poison(x_target, y_target, x_train, y_train)

        np.testing.assert_equal(x_poison.shape, x_train.shape)
        np.testing.assert_equal(y_poison.shape, y_train.shape)
    except ARTTestException as e:
        art_warning(e)
