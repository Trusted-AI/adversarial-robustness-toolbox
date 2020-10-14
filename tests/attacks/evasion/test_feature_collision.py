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

from art.attacks.poisoning.feature_collision_attack import FeatureCollisionAttack

from tests.utils import ARTTestException


logger = logging.getLogger(__name__)


def test_feature_collision(art_warning, get_default_mnist_subset, image_dl_estimator_for_attack):
    try:
        (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_default_mnist_subset
        classifier, sess = image_dl_estimator_for_attack(FeatureCollisionAttack)

        # poison dataset
        x_poison = np.copy(x_train_mnist)
        y_poison = np.copy(y_train_mnist)
        base = np.expand_dims(x_train_mnist[0], axis=0)
        target = np.expand_dims(x_train_mnist[1], axis=0)
        feature_layer = classifier.layer_names[-1]
        attack = FeatureCollisionAttack(classifier, target, feature_layer, max_iter=1)
        attack, attack_label = attack.poison(base)
        x_poison = np.append(x_poison, attack, axis=0)
        y_poison = np.append(y_poison, attack_label, axis=0)

        classifier.fit(x_poison, y_poison, nb_epochs=3, batch_size=32)
        # TODO nothing is really been tested in this test
    except ARTTestException as e:
        art_warning(e)
