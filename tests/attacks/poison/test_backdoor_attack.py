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
from tests.utils import ARTTestException

logger = logging.getLogger(__name__)

PP_POISON = 0.33
NB_EPOCHS = 3

backdoor_path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))),
                             "utils", "data", "backdoors", "alert.png")

max_val = 1


@pytest.fixture()
def poison_dataset(get_default_mnist_subset):
    (x_clean, y_clean), (_, _) = get_default_mnist_subset

    def _poison_dataset(poison_func):
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

    return _poison_dataset


@pytest.fixture()
def mxnet_or_pytorch(framework):
    return framework == 'mxnet' or framework == 'pytorch'


def poison_image_1(x, channels_first):
    channel = 1 if channels_first else 3
    return np.expand_dims(insert_image(x.squeeze(channel), backdoor_path=backdoor_path, size=(5, 5), random=False,
                                       x_shift=3, y_shift=3, mode='L'), axis=channel)


def poison_image_2(x, channels_first):
    channel = 1 if channels_first else 3
    return np.expand_dims(insert_image(x.squeeze(channel), backdoor_path=backdoor_path, size=(5, 5), random=True),
                          axis=channel)


def poison_image_3(x, channels_first):
    channel = 1 if channels_first else 3
    return np.expand_dims(insert_image(x.squeeze(channel), backdoor_path=backdoor_path, random=True, size=(100, 100)),
                          axis=channel)


@pytest.mark.skipMlFramework("non_dl_frameworks")
def test_backdoor_pattern(art_warning, image_dl_estimator, poison_dataset, mxnet_or_pytorch):
    """
    Test the backdoor attack with a pattern-based perturbation can be trained on classifier
    """
    try:
        krc, _ = image_dl_estimator()
        (is_poison_train, x_poisoned_raw, y_poisoned_raw) = poison_dataset(
            lambda x: add_pattern_bd(x, channels_first=mxnet_or_pytorch))
        n_train = np.shape(y_poisoned_raw)[0]
        shuffled_indices = np.arange(n_train)
        np.random.shuffle(shuffled_indices)
        x_train = x_poisoned_raw[shuffled_indices]
        y_train = y_poisoned_raw[shuffled_indices]

        krc.fit(x_train, y_train, nb_epochs=NB_EPOCHS, batch_size=32)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skipMlFramework("non_dl_frameworks")
def test_backdoor_pixel(art_warning, image_dl_estimator, poison_dataset, mxnet_or_pytorch):
    """
    Test the backdoor attack with a pattern-based perturbation can be trained on classifier
    """
    try:
        krc, _ = image_dl_estimator()
        (is_poison_train, x_poisoned_raw, y_poisoned_raw) = poison_dataset(
            lambda x: add_single_bd(x, channels_first=mxnet_or_pytorch))
        n_train = np.shape(y_poisoned_raw)[0]
        shuffled_indices = np.arange(n_train)
        np.random.shuffle(shuffled_indices)
        x_train = x_poisoned_raw[shuffled_indices]
        y_train = y_poisoned_raw[shuffled_indices]

        krc.fit(x_train, y_train, nb_epochs=NB_EPOCHS, batch_size=32)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skipMlFramework("non_dl_frameworks")
def test_backdoor_image(art_warning, image_dl_estimator, poison_dataset, mxnet_or_pytorch):
    """
    Test the backdoor attack with a pattern-based perturbation can be trained on classifier
    """
    try:
        krc, _ = image_dl_estimator()
        (is_poison_train, x_poisoned_raw, y_poisoned_raw) = poison_dataset(
            lambda x: poison_image_1(x, channels_first=mxnet_or_pytorch))
        n_train = np.shape(y_poisoned_raw)[0]
        shuffled_indices = np.arange(n_train)
        np.random.shuffle(shuffled_indices)
        x_train = x_poisoned_raw[shuffled_indices]
        y_train = y_poisoned_raw[shuffled_indices]

        krc.fit(x_train, y_train, nb_epochs=NB_EPOCHS, batch_size=32)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skipMlFramework("non_dl_frameworks")
def test_multiple_perturbations(art_warning, image_dl_estimator, poison_dataset, mxnet_or_pytorch):
    """
    Test the backdoor attack with a pattern-based perturbation can be trained on classifier
    """
    try:
        krc, _ = image_dl_estimator()

        def func1(x):
            return poison_image_1(x, channels_first=mxnet_or_pytorch)

        def func2(x):
            return add_pattern_bd(x, channels_first=mxnet_or_pytorch)

        def func3(x):
            return poison_image_2(x, channels_first=mxnet_or_pytorch)

        (is_poison_train, x_poisoned_raw, y_poisoned_raw) = poison_dataset([func1, func2, func3])
        n_train = np.shape(y_poisoned_raw)[0]
        shuffled_indices = np.arange(n_train)
        np.random.shuffle(shuffled_indices)
        x_train = x_poisoned_raw[shuffled_indices]
        y_train = y_poisoned_raw[shuffled_indices]

        krc.fit(x_train, y_train, nb_epochs=NB_EPOCHS, batch_size=32)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skipMlFramework("non_dl_frameworks")
def test_image_failure_mode(art_warning, image_dl_estimator, poison_dataset, mxnet_or_pytorch):
    """
    Test the backdoor attack with a pattern-based perturbation can be trained on classifier
    """
    try:
        krc, _ = image_dl_estimator()
        with pytest.raises(ValueError):
            poison_dataset(lambda x: poison_image_3(x, channels_first=mxnet_or_pytorch))
    except ARTTestException as e:
        art_warning(e)
