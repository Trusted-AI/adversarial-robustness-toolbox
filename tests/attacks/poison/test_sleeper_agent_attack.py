# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2022
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
from PIL import Image

from art.attacks.poisoning.sleeper_agent_attack import SleeperAgentAttack
from art.utils import to_categorical
from skimage.transform import resize

from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.mark.only_with_platform("pytorch", "tensorflow2")
def test_poison(art_warning, get_default_mnist_subset, image_dl_estimator):
    print("Getting into test fucntion")
    try:
        (x_train, y_train), (x_test, y_test) = get_default_mnist_subset
        classifier, _ = image_dl_estimator()
        x_train, y_train = x_train[:1000], y_train[:1000]
        max_ = 1
        min_ = 0
        mean = np.mean(x_train, axis=(0, 1, 2, 3))
        std = np.std(x_train, axis=(0, 1, 2, 3))
        x_train = (x_train - mean) / (std + 1e-7)
        x_test = (x_test - mean) / (std + 1e-7)
        min_ = (min_ - mean) / (std + 1e-7)
        max_ = (max_ - mean) / (std + 1e-7)
        patch_size = 8
        img = Image.open("trigger_10.png")
        numpydata = np.asarray(img)
        patch = resize(numpydata, (patch_size, patch_size, 3))
        patch = (patch - mean) / (std + 1e-7)
        class_source = 0
        class_target = 1
        K = 10
        x_train_ = np.copy(x_train)
        index_source = np.where(y_train.argmax(axis=1) == class_source)[0][0:K]
        index_target = np.where(y_train.argmax(axis=1) == class_target)[0]
        x_trigger = x_train_[index_source]
        y_trigger = to_categorical([class_target], nb_classes=10)
        y_trigger = np.tile(y_trigger, (len(index_source), 1))
        epsilon = 0.3
        percent_poison = 0.10

        attack = SleeperAgentAttack(
            classifier,
            percent_poison=0.10,
            max_trials=1,
            max_epochs=500,
            clip_values=(min_, max_),
            epsilon=16 / 255 * (max_ - min_),
            batch_size=500,
            verbose=1,
            indices_target=index_target,
            patching_strategy="random",
            selection_strategy="max-norm",
            patch=patch,
            retraining_factor=4,
            model_retrain=True,
            model_retraining_epoch=40,
            class_source=class_source,
            class_target=class_target,
        )
        x_poison, y_poison = attack.poison(x_trigger, y_trigger, x_train, y_train, x_test, y_test)
        np.testing.assert_(
            np.all(np.sum(np.reshape((x_poison - x_train) ** 2, [x_poison.shape[0], -1]), axis=1) < epsilon)
        )
        np.testing.assert_(
            np.sum(np.sum(np.reshape((x_poison - x_train) ** 2, [x_poison.shape[0], -1]), axis=1) > 0)
            <= percent_poison * x_train.shape[0]
        )
        np.testing.assert_equal(np.shape(x_poison), np.shape(x_train))
        np.testing.assert_equal(np.shape(y_poison), np.shape(y_train))
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch", "tensorflow2")
def test_check_params(art_warning, get_default_mnist_subset, image_dl_estimator):
    try:
        classifier, _ = image_dl_estimator(functional=True)

        with pytest.raises(ValueError):
            _ = SleeperAgentAttack(classifier, percent_poison=0.01, learning_rate_schedule=[0.1, 0.2, 0.3])
        with pytest.raises(ValueError):
            _ = SleeperAgentAttack(classifier, percent_poison=1.2)
        with pytest.raises(ValueError):
            _ = SleeperAgentAttack(classifier, percent_poison=0.01, max_epochs=0)
        with pytest.raises(ValueError):
            _ = SleeperAgentAttack(classifier, percent_poison=0.01, max_trials=0)
        with pytest.raises(ValueError):
            _ = SleeperAgentAttack(classifier, percent_poison=0.01, clip_values=1)
        with pytest.raises(ValueError):
            _ = SleeperAgentAttack(classifier, percent_poison=0.01, epsilon=-1)
        with pytest.raises(ValueError):
            _ = SleeperAgentAttack(classifier, percent_poison=0.01, batch_size=0)
        with pytest.raises(ValueError):
            _ = SleeperAgentAttack(classifier, percent_poison=0.01, verbose=1.1)

    except ARTTestException as e:
        art_warning(e)