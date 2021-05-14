# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2021
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

from art.attacks.poisoning import BullseyePolytopeAttackPyTorch

from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.mark.skip_framework("non_dl_frameworks", "tensorflow", "mxnet", "keras", "kerastf")
def test_poison(art_warning, get_default_mnist_subset, image_dl_estimator):
    try:
        (x_train, y_train), (_, _) = get_default_mnist_subset
        classifier, _ = image_dl_estimator(functional=True)
        target = np.expand_dims(x_train[3], 0)
        attack = BullseyePolytopeAttackPyTorch(classifier, target, len(classifier.layer_names) - 2)
        poison_data, poison_labels = attack.poison(x_train[5:10], y_train[5:10])

        np.testing.assert_equal(poison_data.shape, x_train[5:10].shape)
        np.testing.assert_equal(poison_labels.shape, y_train[5:10].shape)

        with pytest.raises(AssertionError):
            np.testing.assert_equal(poison_data, x_train[5:10])

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("non_dl_frameworks", "tensorflow", "mxnet", "keras", "kerastf")
def test_poison_multiple_layers(art_warning, get_default_mnist_subset, image_dl_estimator):
    try:
        (x_train, y_train), (_, _) = get_default_mnist_subset
        classifier, _ = image_dl_estimator(functional=True)
        target = np.expand_dims(x_train[3], 0)
        num_layers = len(classifier.layer_names)
        attack = BullseyePolytopeAttackPyTorch(classifier, target, [num_layers - 2, num_layers - 3])
        poison_data, poison_labels = attack.poison(x_train[5:10], y_train[5:10])

        np.testing.assert_equal(poison_data.shape, x_train[5:10].shape)
        np.testing.assert_equal(poison_labels.shape, y_train[5:10].shape)

        with pytest.raises(AssertionError):
            np.testing.assert_equal(poison_data, x_train[5:10])

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("non_dl_frameworks", "tensorflow", "mxnet", "keras", "kerastf")
def test_failure_modes(art_warning, get_default_mnist_subset, image_dl_estimator):
    try:
        (x_train, y_train), (_, _) = get_default_mnist_subset
        classifier, _ = image_dl_estimator(functional=True)
        target = np.expand_dims(x_train[3], 0)
        with pytest.raises(ValueError):
            _ = BullseyePolytopeAttackPyTorch(classifier, target, len(classifier.layer_names) - 2, learning_rate=-1)
        with pytest.raises(ValueError):
            _ = BullseyePolytopeAttackPyTorch(classifier, target, len(classifier.layer_names) - 2, max_iter=-1)
        with pytest.raises(TypeError):
            _ = BullseyePolytopeAttackPyTorch(classifier, target, 2.5)
        with pytest.raises(ValueError):
            _ = BullseyePolytopeAttackPyTorch(classifier, target, len(classifier.layer_names) - 2, opt="new optimizer")
        with pytest.raises(ValueError):
            _ = BullseyePolytopeAttackPyTorch(classifier, target, len(classifier.layer_names) - 2, momentum=1.2)
        with pytest.raises(ValueError):
            _ = BullseyePolytopeAttackPyTorch(classifier, target, len(classifier.layer_names) - 2, decay_iter=-1)
        with pytest.raises(ValueError):
            _ = BullseyePolytopeAttackPyTorch(classifier, target, len(classifier.layer_names) - 2, epsilon=-1)
        with pytest.raises(ValueError):
            _ = BullseyePolytopeAttackPyTorch(classifier, target, len(classifier.layer_names) - 2, dropout=2)
        with pytest.raises(ValueError):
            _ = BullseyePolytopeAttackPyTorch(classifier, target, len(classifier.layer_names) - 2, net_repeat=-1)
        with pytest.raises(ValueError):
            _ = BullseyePolytopeAttackPyTorch(classifier, target, -1)
        with pytest.raises(ValueError):
            _ = BullseyePolytopeAttackPyTorch(classifier, target, len(classifier.layer_names) - 2, decay_coeff=2)
    except ARTTestException as e:
        art_warning(e)
