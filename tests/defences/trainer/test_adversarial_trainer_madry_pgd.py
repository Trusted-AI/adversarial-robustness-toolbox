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
from art.utils import load_mnist

from tests.utils import master_seed, get_image_classifier_tf, get_image_classifier_pt, get_image_classifier_hf

logger = logging.getLogger(__name__)

BATCH_SIZE = 10
NB_TRAIN = 100
NB_TEST = 100


@pytest.fixture()
def get_mnist_classifier(framework, image_dl_estimator):
    def _get_classifier():
        if framework == "pytorch":
            classifier = get_image_classifier_pt()
            master_seed(seed=1234, set_torch=True)

        elif framework == "huggingface":
            classifier = get_image_classifier_hf()
            master_seed(seed=1234, set_torch=True)

        elif framework in ["tensorflow2", "tensorflow1"]:
            classifier, _ = get_image_classifier_tf()
            master_seed(seed=1234)
        else:
            return None

        return classifier

    return _get_classifier


@pytest.mark.only_with_platform("pytorch", "tensorflow2", "huggingface", "tensorflow1", "tensorflow2v1")
def test_fit_predict(art_warning, get_mnist_classifier, framework):
    classifier = get_mnist_classifier()

    (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
    x_train, y_train, x_test, y_test = (
        x_train[:NB_TRAIN],
        y_train[:NB_TRAIN],
        x_test[:NB_TEST],
        y_test[:NB_TEST],
    )
    if framework in ["pytorch", "huggingface"]:
        x_train = np.float32(np.rollaxis(x_train, 3, 1))
        x_test = np.float32(np.rollaxis(x_test, 3, 1))

    x_test_original = x_test.copy()

    if framework in ["pytorch", "huggingface"]:
        pt_classifier = get_image_classifier_pt()
        pt_predictions_new = np.argmax(pt_classifier.predict(x_test), axis=1)
        hf_preds = np.argmax(classifier.predict(x_test), axis=1)

        assert np.array_equal(pt_predictions_new, hf_preds)

    adv_trainer = AdversarialTrainerMadryPGD(classifier, nb_epochs=1, batch_size=128)
    adv_trainer.fit(x_train, y_train)

    predictions_new = np.argmax(adv_trainer.trainer.get_classifier().predict(x_test), axis=1)
    accuracy_new = np.sum(predictions_new == np.argmax(y_test, axis=1)) / NB_TEST

    # Pytorch was not checked in unittest, but has 1% lower performance than tf.
    if framework in ["pytorch", "huggingface"]:
        assert accuracy_new == 0.37
    else:
        assert accuracy_new == 0.38

    # Check that x_test has not been modified by attack and classifier
    assert np.allclose(x_test_original, x_test)


@pytest.mark.only_with_platform("pytorch", "tensorflow2", "tensorflow1", "huggingface", "tensorflow2v1")
def test_get_classifier(art_warning, get_mnist_classifier):
    classifier = get_mnist_classifier()

    adv_trainer = AdversarialTrainerMadryPGD(classifier, nb_epochs=1, batch_size=128)
    _ = adv_trainer.get_classifier()
