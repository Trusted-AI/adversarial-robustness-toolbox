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
from sklearn.svm import SVC, NuSVC

from art.attacks.poisoning import PoisoningAttackSVM
from art.estimators.classification.scikitlearn import ScikitlearnSVC, SklearnClassifier
from art.utils import load_dataset
from tests.attacks.utils import backend_test_classifier_type_check_fail
from tests.utils import ARTTestException

logger = logging.getLogger(__name__)

NB_TRAIN = 10
NB_TEST = 100

_, _, min_, max_ = load_dataset("iris")


def find_duplicates(x_train):
    """
    Returns an array of booleans that is true if that element was previously in the array

    :param x_train: training data
    :type x_train: `np.ndarray`
    :return: duplicates array
    :rtype: `np.ndarray`
    """
    dup = np.zeros(x_train.shape[0])
    for idx, x in enumerate(x_train):
        dup[idx] = np.isin(x_train[:idx], x).all(axis=1).any()
    return dup


@pytest.fixture()
def get_iris(get_iris_dataset, image_dl_estimator):
    (x_train, y_train), (x_test, y_test) = get_iris_dataset
    no_zero = np.where(np.argmax(y_train, axis=1) != 0)
    x_train = x_train[no_zero, :2][0]
    y_train = y_train[no_zero]
    no_zero = np.where(np.argmax(y_test, axis=1) != 0)
    x_test = x_test[no_zero, :2][0]
    y_test = y_test[no_zero]
    labels = np.zeros((y_train.shape[0], 2))
    labels[np.argmax(y_train, axis=1) == 2] = np.array([1, 0])
    labels[np.argmax(y_train, axis=1) == 1] = np.array([0, 1])
    y_train = labels
    te_labels = np.zeros((y_test.shape[0], 2))
    te_labels[np.argmax(y_test, axis=1) == 2] = np.array([1, 0])
    te_labels[np.argmax(y_test, axis=1) == 1] = np.array([0, 1])
    y_test = te_labels
    n_sample = len(x_train)

    order = np.random.permutation(n_sample)
    x_train = x_train[order]
    y_train = y_train[order].astype(np.float)

    x_train = x_train[: int(0.9 * n_sample)]
    y_train = y_train[: int(0.9 * n_sample)]
    train_dups = find_duplicates(x_train)
    x_train = x_train[np.logical_not(train_dups)]
    y_train = y_train[np.logical_not(train_dups)]
    test_dups = find_duplicates(x_test)
    x_test = x_test[np.logical_not(test_dups)]
    y_test = y_test[np.logical_not(test_dups)]

    return (x_train, y_train), (x_test, y_test)


@pytest.mark.only_with_platform("scikitlearn")
@pytest.mark.parametrize("model", [SVC(kernel="sigmoid", gamma="auto"), NuSVC()])
def test_unsupported_classifier(art_warning, get_iris, model):
    try:
        (x_train, y_train), (x_test, y_test) = get_iris
        with pytest.raises(TypeError):
            _ = PoisoningAttackSVM(classifier=model, step=0.01, eps=1.0, x_train=x_train, y_train=y_train, x_val=x_test,
                                   y_val=y_test)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("scikitlearn")
def test_classifier_type_check_fail(art_warning):
    try:
        backend_test_classifier_type_check_fail(PoisoningAttackSVM, [ScikitlearnSVC])
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("scikitlearn")
@pytest.mark.parametrize("kernel", ["linear", "poly", "rbf"])
def test_svc_kernels(art_warning, get_iris, kernel):
    try:
        (x_train, y_train), (x_test, y_test) = get_iris
        x_train = x_train[:NB_TRAIN]
        y_train = y_train[:NB_TRAIN]
        x_test = x_test[:NB_TEST]
        y_test = y_test[:NB_TEST]
        x_test_original = x_test.copy()
        clip_values = (min_, max_)
        clean = SklearnClassifier(model=SVC(kernel=kernel, gamma="auto"), clip_values=clip_values)
        clean.fit(x_train, y_train)
        poison = SklearnClassifier(model=SVC(kernel=kernel, gamma="auto"), clip_values=clip_values)
        poison.fit(x_train, y_train)
        attack = PoisoningAttackSVM(poison, 0.01, 1.0, x_train, y_train, x_test, y_test, 100)
        attack_y = np.array([1, 1]) - y_train[0]
        attack_point, _ = attack.poison(np.array([x_train[0]]), y=np.array([attack_y]))
        poison.fit(
            x=np.vstack([x_train, attack_point]),
            y=np.vstack([y_train, np.array([1, 1]) - np.copy(y_train[0].reshape((1, 2)))]),
        )

        acc = np.average(np.all(clean.predict(x_test) == y_test, axis=1)) * 100
        poison_acc = np.average(np.all(poison.predict(x_test) == y_test, axis=1)) * 100
        logger.info("Clean Accuracy {}%".format(acc))
        logger.info("Poison Accuracy {}%".format(poison_acc))
        assert acc >= poison_acc

        # Check that x_test has not been modified by attack and classifier
        np.testing.assert_almost_equal(float(np.max(np.abs(x_test_original - x_test))), 0.0)
    except ARTTestException as e:
        art_warning(e)
