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

from tests.utils import TestBase, master_seed, get_image_classifier_kr

logger = logging.getLogger(__name__)

PP_POISON = 0.33
NB_EPOCHS = 3


@pytest.fixture()
def fix_get_mnist_subset(get_mnist_dataset):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_mnist_dataset
    n_train = 1000
    n_test = 100
    yield x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test]


@pytest.fixture()
def poison(fix_get_mnist_subset):
    yield Poison(fix_get_mnist_subset)


@pytest.mark.skipMlFramework("pytorch", "scikitlearn", "mxnet", "tensorflow2v1")
@pytest.mark.parametrize("perturbation", ["pattern_based_perturbation", "pixel_based_perturbation",
                                          "image_based_perturbation", "pattern_based_perturbation"])
def test_backdoor_pattern(fix_get_mnist_subset, image_dl_estimator, poison, perturbation):

    estimator, _ = image_dl_estimator()

    _back_end(fix_get_mnist_subset, estimator, poison.perturbation_dic[perturbation])


def test_image_failure_modes(fix_get_mnist_subset, image_dl_estimator, poison):
    """
    Tests failure modes for image perturbation functions
    """
    # poison = Poison(fix_get_mnist_subset)

    (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset

    backdoor_attack = PoisoningAttackBackdoor(poison.image_perturbation_1)
    adv_target = np.argmax(y_train_mnist) + 1 % 10

    with pytest.raises(ValueError) as context:
        backdoor_attack.poison(x_train_mnist, y=adv_target)
        assert "Backdoor does not fit inside original image" in str(context.exception)

    backdoor_attack = PoisoningAttackBackdoor(poison.image_perturbation_2)

    with pytest.raises(ValueError) as context:
        backdoor_attack.poison(np.zeros(5), y=np.ones(5))
        assert "Invalid array shape" in str(context.exception)


def _back_end(fix_get_mnist_subset, estimator, poison_function):
    (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset

    (is_poison_train, x_poisoned_raw, y_poisoned_raw) = poison_dataset(
        x_train_mnist, y_train_mnist, poison_function)
    # Shuffle training data
    n_train = np.shape(y_poisoned_raw)[0]
    shuffled_indices = np.arange(n_train)
    np.random.shuffle(shuffled_indices)
    x_train = x_poisoned_raw[shuffled_indices]
    y_train = y_poisoned_raw[shuffled_indices]

    estimator.fit(x_train, y_train, nb_epochs=NB_EPOCHS, batch_size=32)


def poison_dataset(x_clean, y_clean, poison_func):
    x_poison = np.copy(x_clean)
    y_poison = np.copy(y_clean)
    is_poison = np.zeros(np.shape(y_poison)[0])

    for i in range(10):
        src = i
        tgt = (i + 1) % 10
        n_points_in_tgt = np.round(np.sum(np.argmax(y_clean, axis=1) == tgt))
        num_poison = int((PP_POISON * n_points_in_tgt) / (1 - PP_POISON))
        src_imgs = np.copy(x_clean[np.argmax(y_clean, axis=1) == src])

        n_points_in_src = np.shape(src_imgs)[0]
        if num_poison:
            indices_to_be_poisoned = np.random.choice(n_points_in_src, num_poison)

            imgs_to_be_poisoned = src_imgs[indices_to_be_poisoned]
            backdoor_attack = PoisoningAttackBackdoor(poison_func)
            poison_images, poison_labels = backdoor_attack.poison(
                imgs_to_be_poisoned, y=to_categorical(np.ones(num_poison) * tgt, 10)
            )
            x_poison = np.append(x_poison, poison_images, axis=0)
            y_poison = np.append(y_poison, poison_labels, axis=0)
            is_poison = np.append(is_poison, np.ones(num_poison))

    is_poison = is_poison != 0

    return is_poison, x_poison, y_poison


class Poison():
    def __init__(self, mnist_dataset):
        (self.x_train_mnist, self.y_train_mnist, self.x_test_mnist, self.y_test_mnist) = mnist_dataset
        self.backdoor_path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                                          "..",
                                          "utils",
                                          "data",
                                          "backdoors",
                                          "alert.png",
                                          )

        self.perturbation_dic = {"pattern_based_perturbation":self.pattern_based_perturbation,
                                 "pixel_based_perturbation": self.pixel_based_perturbation,
                                 "image_based_perturbation": self.image_based_perturbation}

    def pattern_based_perturbation(self, x):
        max_val = np.max(self.x_train_mnist)
        return np.expand_dims(add_pattern_bd(x.squeeze(3), pixel_value=max_val), axis=3)

    def pixel_based_perturbation(self, x):
        max_val = np.max(self.x_train_mnist)
        return np.expand_dims(add_single_bd(x.squeeze(3), pixel_value=max_val), axis=3)

    def image_based_perturbation(self, x):
        return np.expand_dims(
            insert_image(
                x.squeeze(3), backdoor_path=self.backdoor_path, size=(5, 5), random=False, x_shift=3, y_shift=3
            ),
            axis=3,
        )

    def poison_func_4(self, x):
        return np.expand_dims(
            insert_image(x.squeeze(3), backdoor_path=self.backdoor_path, size=(5, 5), random=True), axis=3
        )

    def image_perturbation_1(self, x):
        return np.expand_dims(
            insert_image(x.squeeze(3), backdoor_path=self.backdoor_path, random=True, size=(100, 100)), axis=3
        )

    def image_perturbation_2(self, x):
        return np.expand_dims(insert_image(x, backdoor_path=self.backdoor_path, random=True, size=(100, 100)), axis=3)


