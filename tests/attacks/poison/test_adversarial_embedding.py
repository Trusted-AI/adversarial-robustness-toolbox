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

from art.attacks.poisoning import PoisoningAttackAdversarialEmbedding
from art.attacks.poisoning.backdoor_attack import PoisoningAttackBackdoor
from art.attacks.poisoning.perturbations import add_pattern_bd
from tests.utils import ARTTestException

logger = logging.getLogger(__name__)

NB_EPOCHS = 3


@pytest.mark.skipMlFramework("non_dl_frameworks", "mxnet", "pytorch", "kerastf")
def test_poison(art_warning, image_dl_estimator, get_default_mnist_subset):
    """
    Test the backdoor attack with a pattern-based perturbation can be trained on classifier
    """
    try:
        krc, _ = image_dl_estimator(functional=True)
        (x_train, y_train), (_, _) = get_default_mnist_subset

        target_idx = 9
        target = np.zeros(10)
        target[target_idx] = 1
        target2 = np.zeros(10)
        target2[(target_idx + 1) % 10] = 1

        backdoor = PoisoningAttackBackdoor(add_pattern_bd)

        emb_attack = PoisoningAttackAdversarialEmbedding(krc, backdoor, 2, target)
        classifier = emb_attack.poison_estimator(x_train, y_train, nb_epochs=NB_EPOCHS)

        data, labels, bd = emb_attack.get_training_data()

        assert x_train.shape == data.shape
        assert y_train.shape == labels.shape
        assert bd.shape == (len(x_train), 2)

        # Assert successful cloning of classifier model
        assert classifier is not krc

        emb_attack2 = PoisoningAttackAdversarialEmbedding(krc, backdoor, 2, [(target, target2)])
        _ = emb_attack2.poison_estimator(x_train, y_train, nb_epochs=NB_EPOCHS)

        data, labels, bd = emb_attack2.get_training_data()
        assert x_train.shape == data.shape
        assert y_train.shape == labels.shape
        assert bd.shape == (len(x_train), 2)

        _ = PoisoningAttackAdversarialEmbedding(krc, backdoor, 2, [(target, target2)], pp_poison=[0.4])
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skipMlFramework("non_dl_frameworks", "mxnet", "pytorch")
@pytest.mark.parametrize("params", [dict(regularization=-1), dict(discriminator_layer_1=-1),
                                    dict(discriminator_layer_2=-1), dict(pp_poison=-1), dict(pp_poison=[]),
                                    dict(pp_poison=[-1])])
def test_errors(art_warning, image_dl_estimator, params):
    """
    Test the backdoor attack with a pattern-based perturbation can be trained on classifier
    """
    try:
        krc, _ = image_dl_estimator(functional=True)

        target_idx = 9
        target = np.zeros(10)
        target[target_idx] = 1
        target2 = np.zeros(10)
        target2[(target_idx + 1) % 10] = 1

        backdoor = PoisoningAttackBackdoor(add_pattern_bd)

        # invalid loss function
        with pytest.raises(ValueError):
            _ = PoisoningAttackAdversarialEmbedding(krc, backdoor, 2, target, **params)
    except ARTTestException as e:
        art_warning(e)
