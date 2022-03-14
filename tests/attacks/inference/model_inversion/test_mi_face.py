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
import pytest

import numpy as np

from art.attacks.inference.model_inversion import MIFace
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin, ClassGradientsMixin

from tests.attacks.utils import backend_test_classifier_type_check_fail
from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.fixture()
def fix_get_mnist_subset(get_mnist_dataset):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_mnist_dataset
    n_train = 100
    n_test = 11
    yield x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test]


def backend_check_inferred_values(attack, mnist_dataset, classifier):
    # We assert that, when starting with zero information about the inputs x, we are able to infer - for each
    # class - a representative sample that is classified by the classifier as belonging to that class:

    x_train_infer_from_zero = attack.infer(None, y=np.arange(10))
    preds = np.argmax(classifier.predict(x_train_infer_from_zero), axis=1)
    np.testing.assert_array_equal(preds, np.arange(10))

    # Next we would like to assert that, when starting with blurry training instances, the inference attack will result
    # in instances that more closely resemble the original instances. However, this turns out not to be the case in
    # this scenario:

    (x_train_mnist, y_train_mnist, _, _) = mnist_dataset
    x_original = x_train_mnist[:10]
    x_noisy = np.clip(x_original + np.random.uniform(-0.01, 0.01, x_original.shape), 0, 1)
    x_train_infer_from_noisy = attack.infer(x_noisy, y=y_train_mnist[:10])

    diff_noisy = np.mean(np.reshape(np.abs(x_original - x_noisy), (len(x_original), -1)), axis=1)
    diff_inferred = np.mean(np.reshape(np.abs(x_original - x_train_infer_from_noisy), (len(x_original), -1)), axis=1)

    np.testing.assert_array_less(diff_noisy, diff_inferred)


@pytest.mark.framework_agnostic
def test_miface(art_warning, fix_get_mnist_subset, image_dl_estimator_for_attack):
    try:
        classifier = image_dl_estimator_for_attack(MIFace)

        # for the one-shot method, frame saliency attack should resort to plain FastGradientMethod
        # expected_values = {
        #     "x_test_mean": ExpectedValue(0.2346725, 0.002),
        #     "x_test_min": ExpectedValue(-1.0, 0.00001),
        #     "x_test_max": ExpectedValue(1.0, 0.00001),
        #     "y_test_pred_adv_expected": ExpectedValue(np.asarray([4, 4, 4, 7, 7, 4, 7, 2, 2, 3, 0]), 2),
        # }

        attack = MIFace(classifier, max_iter=150, batch_size=3)
        backend_check_inferred_values(attack, fix_get_mnist_subset, classifier)
    except ARTTestException as e:
        art_warning(e)


def test_check_params(art_warning, image_dl_estimator_for_attack):
    try:
        classifier = image_dl_estimator_for_attack(MIFace)

        with pytest.raises(ValueError):
            _ = MIFace(classifier, max_iter=-0.5)

        with pytest.raises(ValueError):
            _ = MIFace(classifier, window_length=-0.5)

        with pytest.raises(ValueError):
            _ = MIFace(classifier, threshold=-0.5)

        with pytest.raises(ValueError):
            _ = MIFace(classifier, learning_rate=-0.5)

        with pytest.raises(ValueError):
            _ = MIFace(classifier, batch_size=-0.5)

        with pytest.raises(ValueError):
            _ = MIFace(classifier, verbose=-0.5)

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_classifier_type_check_fail(art_warning):
    try:
        backend_test_classifier_type_check_fail(MIFace, [BaseEstimator, ClassifierMixin, ClassGradientsMixin])
    except ARTTestException as e:
        art_warning(e)
