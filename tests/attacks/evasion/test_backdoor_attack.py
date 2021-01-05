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

import os
import logging

import numpy as np
import pytest

from art.attacks.poisoning.backdoor_attack import PoisoningAttackBackdoor
from art.attacks.poisoning.perturbations import add_pattern_bd, add_single_bd, insert_image
from art.utils import to_categorical

logger = logging.getLogger(__name__)


@pytest.fixture()
def backdoor_path():
    yield os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                       "..",
                       "utils",
                       "data",
                       "backdoors",
                       "alert.png",
                       )


@pytest.fixture()
def fix_get_mnist_subset(get_mnist_dataset):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_mnist_dataset
    n_train = 1000
    n_test = 100
    yield x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test]


@pytest.fixture()
def perturbation_dic(fix_get_mnist_subset, backdoor_path):
    (x_train_mnist, _, _, _) = fix_get_mnist_subset

    def pattern_based_perturbation(x):
        max_val = np.max(x_train_mnist)
        return np.expand_dims(add_pattern_bd(x.squeeze(3), pixel_value=max_val), axis=3)

    def pixel_based_perturbation(x):
        max_val = np.max(x_train_mnist)
        return np.expand_dims(add_single_bd(x.squeeze(3), pixel_value=max_val), axis=3)

    def image_based_perturbation(x):
        return np.expand_dims(
            insert_image(
                x.squeeze(3), backdoor_path=backdoor_path, size=(5, 5), random=False, x_shift=3, y_shift=3
            ),
            axis=3,
        )

    def poison_func_4(x):
        return np.expand_dims(
            insert_image(x.squeeze(3), backdoor_path=backdoor_path, size=(5, 5), random=True), axis=3
        )

    perturbation_dic = {"pattern_based_perturbation": pattern_based_perturbation,
                        "pixel_based_perturbation": pixel_based_perturbation,
                        "image_based_perturbation": image_based_perturbation,
                        "multiple_perturbations": [poison_func_4, pattern_based_perturbation]}

    yield perturbation_dic


@pytest.mark.skipMlFramework("pytorch", "scikitlearn", "mxnet", "tensorflow2v1")
@pytest.mark.parametrize("perturbation", ["pattern_based_perturbation", "pixel_based_perturbation",
                                          "image_based_perturbation", "multiple_perturbations"])
def test_backdoor_pattern(fix_get_mnist_subset, image_dl_estimator, perturbation_dic, perturbation):
    estimator, _ = image_dl_estimator()

    (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset

    poison_func = perturbation_dic[perturbation]

    x_poisoned_raw = np.copy(x_train_mnist)
    y_poisoned_raw = np.copy(y_train_mnist)
    is_poison_train = np.zeros(np.shape(y_poisoned_raw)[0])

    PP_POISON = 0.33

    for i in range(10):
        src = i
        tgt = (i + 1) % 10
        n_points_in_tgt = np.round(np.sum(np.argmax(y_train_mnist, axis=1) == tgt))
        num_poison = int((PP_POISON * n_points_in_tgt) / (1 - PP_POISON))
        src_imgs = np.copy(x_train_mnist[np.argmax(y_train_mnist, axis=1) == src])

        n_points_in_src = np.shape(src_imgs)[0]
        if num_poison:
            indices_to_be_poisoned = np.random.choice(n_points_in_src, num_poison)

            imgs_to_be_poisoned = src_imgs[indices_to_be_poisoned]
            backdoor_attack = PoisoningAttackBackdoor(poison_func)
            poison_images, poison_labels = backdoor_attack.poison(
                imgs_to_be_poisoned, y=to_categorical(np.ones(num_poison) * tgt, 10)
            )
            x_poisoned_raw = np.append(x_poisoned_raw, poison_images, axis=0)
            y_poisoned_raw = np.append(y_poisoned_raw, poison_labels, axis=0)
            is_poison_train = np.append(is_poison_train, np.ones(num_poison))

    is_poison_train = is_poison_train != 0

    # Shuffle training data
    n_train = np.shape(y_poisoned_raw)[0]
    shuffled_indices = np.arange(n_train)
    np.random.shuffle(shuffled_indices)
    x_train = x_poisoned_raw[shuffled_indices]
    y_train = y_poisoned_raw[shuffled_indices]

    estimator.fit(x_train, y_train, nb_epochs=3, batch_size=32)


def test_image_failure_modes(fix_get_mnist_subset, image_dl_estimator, backdoor_path):
    """
    Tests failure modes for image perturbation functions
    """

    (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset

    def image_perturbation_1(x):
        return np.expand_dims(
            insert_image(x.squeeze(3), backdoor_path=backdoor_path, random=True, size=(100, 100)), axis=3
        )

    backdoor_attack = PoisoningAttackBackdoor(image_perturbation_1)
    adv_target = np.argmax(y_train_mnist) + 1 % 10

    with pytest.raises(ValueError) as context:
        backdoor_attack.poison(x_train_mnist, y=adv_target)
        assert "Backdoor does not fit inside original image" in str(context.exception)

    def image_perturbation_2(x):
        return np.expand_dims(insert_image(x, backdoor_path=backdoor_path, random=True, size=(100, 100)), axis=3)

    backdoor_attack = PoisoningAttackBackdoor(image_perturbation_2)

    with pytest.raises(ValueError) as context:
        backdoor_attack.poison(np.zeros(5), y=np.ones(5))
        assert "Invalid array shape" in str(context.exception)
