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

from art.defences.trainer.adversarial_trainer_madry_pgd import AdversarialTrainerMadryPGD

logger = logging.getLogger(__name__)


@pytest.fixture()
def fix_get_mnist_subset(get_mnist_dataset):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_mnist_dataset
    n_train = 100
    n_test = 100
    yield x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test]


@pytest.mark.only_with_platform("pytorch", "tensorflow2", "huggingface", "tensorflow1", "tensorflow2v1")
def test_fit_predict(art_warning, image_dl_estimator, fix_get_mnist_subset):
    classifier, _ = image_dl_estimator()

    (x_train, y_train, x_test, y_test) = fix_get_mnist_subset
    x_test_original = x_test.copy()

    adv_trainer = AdversarialTrainerMadryPGD(classifier, nb_epochs=1, batch_size=128)
    adv_trainer.fit(x_train, y_train)

    predictions_new = np.argmax(adv_trainer.trainer.get_classifier().predict(x_test), axis=1)
    accuracy_new = np.mean(predictions_new == np.argmax(y_test, axis=1))

    assert accuracy_new == pytest.approx(0.375, abs=0.05)
    # Check that x_test has not been modified by attack and classifier
    assert np.allclose(x_test_original, x_test)


@pytest.mark.only_with_platform("pytorch", "tensorflow2", "tensorflow1", "huggingface", "tensorflow2v1")
def test_get_classifier(art_warning, image_dl_estimator):
    classifier, _ = image_dl_estimator()

    adv_trainer = AdversarialTrainerMadryPGD(classifier, nb_epochs=1, batch_size=128)
    _ = adv_trainer.get_classifier()
