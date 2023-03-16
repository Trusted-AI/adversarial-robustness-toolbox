# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2023
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
import os

import numpy as np
import pytest

from art.attacks.poisoning import BadDetGlobalMisclassificationAttack, PoisoningAttackBackdoor
from art.attacks.poisoning.perturbations import add_single_bd, add_pattern_bd, insert_image

from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.mark.framework_agnostic
@pytest.mark.parametrize("percent_poison", [0.3, 1.0])
@pytest.mark.parametrize("channels_first", [True, False])
def test_poison_single_bd(art_warning, image_batch, percent_poison, channels_first):
    x, y = image_batch
    backdoor = PoisoningAttackBackdoor(add_single_bd)

    try:
        attack = BadDetGlobalMisclassificationAttack(
            backdoor=backdoor,
            class_target=1,
            percent_poison=percent_poison,
            channels_first=channels_first,
        )
        poison_data, poison_labels = attack.poison(x, y)

        np.testing.assert_equal(poison_data.shape, x.shape)
        np.testing.assert_equal(poison_labels[0]["boxes"].shape, y[0]["boxes"].shape)
        np.testing.assert_equal(poison_labels[0]["labels"].shape, y[0]["labels"].shape)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
@pytest.mark.parametrize("percent_poison", [0.3, 1.0])
@pytest.mark.parametrize("channels_first", [True, False])
def test_poison_pattern_bd(art_warning, image_batch, percent_poison, channels_first):
    x, y = image_batch
    backdoor = PoisoningAttackBackdoor(add_pattern_bd)

    try:
        attack = BadDetGlobalMisclassificationAttack(
            backdoor=backdoor,
            class_target=1,
            percent_poison=percent_poison,
            channels_first=channels_first,
        )
        poison_data, poison_labels = attack.poison(x, y)

        np.testing.assert_equal(poison_data.shape, x.shape)
        np.testing.assert_equal(poison_labels[0]["boxes"].shape, y[0]["boxes"].shape)
        np.testing.assert_equal(poison_labels[0]["labels"].shape, y[0]["labels"].shape)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
@pytest.mark.parametrize("percent_poison", [0.3, 1.0])
@pytest.mark.parametrize("channels_first", [True, False])
def test_poison_image(art_warning, image_batch, percent_poison, channels_first):
    x, y = image_batch

    file_path = os.path.join(os.getcwd(), "utils/data/backdoors/alert.png")

    def perturbation(x):
        return insert_image(x, backdoor_path=file_path, channels_first=False, size=(2, 2), mode="RGB")

    backdoor = PoisoningAttackBackdoor(perturbation)

    try:
        attack = BadDetGlobalMisclassificationAttack(
            backdoor=backdoor,
            class_target=1,
            percent_poison=percent_poison,
            channels_first=channels_first,
        )
        poison_data, poison_labels = attack.poison(x, y)

        np.testing.assert_equal(poison_data.shape, x.shape)
        np.testing.assert_equal(poison_labels[0]["boxes"].shape, y[0]["boxes"].shape)
        np.testing.assert_equal(poison_labels[0]["labels"].shape, y[0]["labels"].shape)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_check_params(art_warning):
    backdoor = PoisoningAttackBackdoor(add_single_bd)

    try:
        with pytest.raises(ValueError):
            _ = BadDetGlobalMisclassificationAttack(None)

        with pytest.raises(ValueError):
            _ = BadDetGlobalMisclassificationAttack(backdoor=backdoor, percent_poison=-0.1)

        with pytest.raises(ValueError):
            _ = BadDetGlobalMisclassificationAttack(backdoor=backdoor, percent_poison=0)

        with pytest.raises(ValueError):
            _ = BadDetGlobalMisclassificationAttack(backdoor=backdoor, percent_poison=1.1)

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_non_image_data_error(art_warning, tabular_batch):
    x, y = tabular_batch
    backdoor = PoisoningAttackBackdoor(add_single_bd)

    try:
        attack = BadDetGlobalMisclassificationAttack(backdoor=backdoor)

        exc_msg = "Unrecognized input dimension. BadDet GMA can only be applied to image data."
        with pytest.raises(ValueError, match=exc_msg):
            _, _ = attack.poison(x, y)
    except ARTTestException as e:
        art_warning(e)
