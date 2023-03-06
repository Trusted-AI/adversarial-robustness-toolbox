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

from art.attacks.poisoning import BadDetRegionalMisclassificationAttack, PoisoningAttackBackdoor
from art.attacks.poisoning.perturbations import add_single_bd, add_pattern_bd, insert_image
from art.config import ART_NUMPY_DTYPE

from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.fixture()
def image_batch(channels_first):
    """
    Image fixtures of shape NHWC and NCHW and a sample object detection label.
    """
    channels = 3
    height = 20
    width = 16

    if channels_first:
        data_shape = (2, channels, height, width)
    else:
        data_shape = (2, height, width, channels)
    x = (0.5 * np.ones((data_shape))).astype(ART_NUMPY_DTYPE)

    y = []
    for _ in range(len(x)):
        y_1, x_1 = np.random.uniform(0, (height / 2, width / 2))
        y_2, x_2 = np.random.uniform((y_1 + 3, x_1 + 3), (height, width))
        target_dict = {
            "boxes": np.array([[x_1, y_1, x_2, y_2]]),
            "labels": np.array([0]),
            "scores": np.array([0.5]),
        }
        y.append(target_dict)

    return x, y


@pytest.mark.framework_agnostic
@pytest.mark.parametrize("percent_poison", [0.3, 1.0])
@pytest.mark.parametrize("channels_first", [False])
def test_bad_det_rma_poison_single_bd(art_warning, image_batch, percent_poison, channels_first):
    x, y = image_batch
    backdoor = PoisoningAttackBackdoor(add_single_bd)

    try:
        attack = BadDetRegionalMisclassificationAttack(
            backdoor=backdoor,
            class_source=0,
            class_target=1,
            percent_poison=percent_poison,
            channels_first=channels_first,
        )
        poison_data, poison_labels = attack.poison(x, y)

        np.testing.assert_equal(poison_data.shape, x.shape)
        np.testing.assert_equal(poison_labels[0]["boxes"].shape, y[0]["boxes"].shape)
        np.testing.assert_equal(poison_labels[0]["labels"].shape, y[0]["labels"].shape)
        np.testing.assert_equal(poison_labels[0]["scores"].shape, y[0]["scores"].shape)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
@pytest.mark.parametrize("percent_poison", [0.3, 1.0])
@pytest.mark.parametrize("channels_first", [False])
def test_bad_det_rma_poison_pattern_bd(art_warning, image_batch, percent_poison, channels_first):
    x, y = image_batch
    backdoor = PoisoningAttackBackdoor(add_pattern_bd)

    try:
        attack = BadDetRegionalMisclassificationAttack(
            backdoor=backdoor,
            class_source=0,
            class_target=1,
            percent_poison=percent_poison,
            channels_first=channels_first,
        )
        poison_data, poison_labels = attack.poison(x, y)

        np.testing.assert_equal(poison_data.shape, x.shape)
        np.testing.assert_equal(poison_labels[0]["boxes"].shape, y[0]["boxes"].shape)
        np.testing.assert_equal(poison_labels[0]["labels"].shape, y[0]["labels"].shape)
        np.testing.assert_equal(poison_labels[0]["scores"].shape, y[0]["scores"].shape)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
@pytest.mark.parametrize("percent_poison", [0.3, 1.0])
@pytest.mark.parametrize("channels_first", [True, False])
def test_bad_det_rma_poison_image(art_warning, image_batch, percent_poison, channels_first):
    x, y = image_batch

    file_path = os.path.join(os.getcwd(), "utils/data/backdoors/alert.png")

    def perturbation(x):
        return insert_image(x, backdoor_path=file_path, channels_first=channels_first, size=(2, 2), mode="RGB")

    backdoor = PoisoningAttackBackdoor(perturbation)

    try:
        attack = BadDetRegionalMisclassificationAttack(
            backdoor=backdoor,
            class_source=0,
            class_target=1,
            percent_poison=percent_poison,
            channels_first=channels_first,
        )
        poison_data, poison_labels = attack.poison(x, y)

        np.testing.assert_equal(poison_data.shape, x.shape)
        np.testing.assert_equal(poison_labels[0]["boxes"].shape, y[0]["boxes"].shape)
        np.testing.assert_equal(poison_labels[0]["labels"].shape, y[0]["labels"].shape)
        np.testing.assert_equal(poison_labels[0]["scores"].shape, y[0]["scores"].shape)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_check_params(art_warning):
    backdoor = PoisoningAttackBackdoor(add_single_bd)

    try:
        with pytest.raises(ValueError):
            _ = BadDetRegionalMisclassificationAttack(None)

        with pytest.raises(ValueError):
            _ = BadDetRegionalMisclassificationAttack(backdoor=backdoor, percent_poison=-0.1)

        with pytest.raises(ValueError):
            _ = BadDetRegionalMisclassificationAttack(backdoor=backdoor, percent_poison=0)

        with pytest.raises(ValueError):
            _ = BadDetRegionalMisclassificationAttack(backdoor=backdoor, percent_poison=1.1)

    except ARTTestException as e:
        art_warning(e)
