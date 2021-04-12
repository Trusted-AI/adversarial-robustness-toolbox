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

from art.attacks.poisoning import PoisoningAttackCleanLabelBackdoor, PoisoningAttackBackdoor
from art.attacks.poisoning.perturbations import add_pattern_bd
from art.utils import to_categorical

from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.mark.skip_framework("non_dl_frameworks", "pytorch", "mxnet")
def test_poison(art_warning, get_default_mnist_subset, image_dl_estimator):
    try:
        (x_train, y_train), (_, _) = get_default_mnist_subset
        classifier, _ = image_dl_estimator()
        target = to_categorical([9], 10)[0]
        backdoor = PoisoningAttackBackdoor(add_pattern_bd)
        attack = PoisoningAttackCleanLabelBackdoor(backdoor, classifier, target)
        poison_data, poison_labels = attack.poison(x_train, y_train)

        np.testing.assert_equal(poison_data.shape, x_train.shape)
        np.testing.assert_equal(poison_labels.shape, y_train.shape)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.parametrize("params", [dict(pp_poison=-0.2), dict(pp_poison=1.2)])
@pytest.mark.skip_framework("non_dl_frameworks", "pytorch", "mxnet")
def test_failure_modes(art_warning, image_dl_estimator, params):
    try:
        classifier, _ = image_dl_estimator()
        target = to_categorical([9], 10)[0]
        backdoor = PoisoningAttackBackdoor(add_pattern_bd)
        with pytest.raises(ValueError):
            _ = PoisoningAttackCleanLabelBackdoor(backdoor, classifier, target, **params)
    except ARTTestException as e:
        art_warning(e)
