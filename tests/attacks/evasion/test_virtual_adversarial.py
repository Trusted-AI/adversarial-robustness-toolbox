# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
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

from art.attacks.evasion.virtual_adversarial import VirtualAdversarialMethod
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import get_labels_np_array

from tests.utils import ARTTestException
from tests.attacks.utils import backend_test_classifier_type_check_fail

logger = logging.getLogger(__name__)


@pytest.fixture()
def fix_get_mnist_subset(get_mnist_dataset):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_mnist_dataset
    n_train = 100
    n_test = 10
    yield x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test]


# Working with Pytorch, Tensorflow, Keras
@pytest.mark.skipMlFramework("mxnet")
def test_image(art_warning, fix_get_mnist_subset, image_dl_estimator, framework):
    try:
        (x_train, y_train, x_test, y_test) = fix_get_mnist_subset

        estimator, sess = image_dl_estimator()

        scores = get_labels_np_array(estimator.predict(x_train))
        acc = np.sum(np.argmax(scores, axis=1) == np.argmax(y_train, axis=1)) / y_train.shape[0]
        logger.info("[PyTorch, MNIST] Accuracy on training set: %.2f%%", (acc * 100))

        scores = get_labels_np_array(estimator.predict(x_test))
        acc = np.sum(np.argmax(scores, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info("[PyTorch, MNIST] Accuracy on test set: %.2f%%", (acc * 100))

        back_end_test_mnist(estimator, x_test, y_test)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.parametrize("clipped", [True, False])
def test_tabular(art_warning, get_iris_dataset, tabular_dl_estimator, framework, clipped):
    try:
        (_, _), (x_test, y_test) = get_iris_dataset

        estimator = tabular_dl_estimator(clipped=clipped)

        attack = VirtualAdversarialMethod(estimator, eps=1)

        if framework == "tensorflow2" or framework == "tensorflow1" or framework == "pytorch":
            # TODO this if statement should check for an estimator's feature instead of relying on the framework param
            with pytest.raises(TypeError) as context:
                _ = attack.generate(x_test.astype(np.float32))

                assert (
                    "This attack requires a classifier predicting probabilities in the range [0, 1] "
                    "as output." in str(context.exception)
                )
                assert "Values smaller than 0.0 or larger than 1.0 have been detected." in str(context.exception)
            return

        x_test_iris_adv = attack.generate(x_test)

        if clipped:
            # numpy np.testing.assert_array_less here doesn't seem to pass
            assert bool((x_test_iris_adv <= 1).all())
            assert bool((x_test_iris_adv >= 0).all())

        else:
            assert bool((x_test_iris_adv > 1).any())
            assert bool((x_test_iris_adv < 0).any())

        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, x_test, x_test_iris_adv)
        preds_adv = np.argmax(estimator.predict(x_test_iris_adv), axis=1)
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, np.argmax(y_test, axis=1), preds_adv)
        acc = np.sum(preds_adv == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info("Accuracy on Iris with VAT adversarial examples: %.2f%%", (acc * 100))
    except ARTTestException as e:
        art_warning(e)


def test_classifier_type_check_fail():
    backend_test_classifier_type_check_fail(VirtualAdversarialMethod, [BaseEstimator, ClassifierMixin])


def back_end_test_mnist(classifier, x_test, y_test):
    x_test_original = x_test.copy()

    df = VirtualAdversarialMethod(classifier, batch_size=100, max_iter=2)

    x_test_adv = df.generate(x_test)

    # self.assertFalse((x_test == x_test_adv).all())
    np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, x_test, x_test_adv)

    y_pred = get_labels_np_array(classifier.predict(x_test_adv))
    # self.assertFalse((y_test == y_pred).all())
    np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, y_test, y_pred)

    acc = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
    logger.info("Accuracy on adversarial examples: %.2f%%", (acc * 100))

    # Check that x_test has not been modified by attack and classifier
    # self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test))), 0.0, delta=0.00001)
    np.testing.assert_almost_equal(float(np.max(np.abs(x_test_original - x_test))), 0, decimal=5)
