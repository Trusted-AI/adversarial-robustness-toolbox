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

from art.attacks.evasion.wasserstein import Wasserstein
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import get_labels_np_array

from tests.attacks.utils import backend_test_classifier_type_check_fail
from tests.utils import ExpectedValue, ARTTestException

logger = logging.getLogger(__name__)


@pytest.fixture()
def fix_get_mnist_subset(get_mnist_dataset):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_mnist_dataset
    n_train = 10
    n_test = 10
    yield x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test]


def test_unsquared_images(art_warning):
    try:
        from art.estimators.estimator import (
            BaseEstimator,
            LossGradientsMixin,
            NeuralNetworkMixin,
        )

        from art.estimators.classification.classifier import (
            ClassGradientsMixin,
            ClassifierMixin,
        )

        class DummyClassifier(ClassGradientsMixin, ClassifierMixin, NeuralNetworkMixin, LossGradientsMixin,
                              BaseEstimator):
            def __init__(self):
                super(DummyClassifier, self).__init__(model=None, clip_values=None, channels_first=True)
                self._nb_classes = 10

            def class_gradient(self):
                return None

            def fit(self):
                pass

            def loss_gradient(self, x, y):
                return np.random.normal(size=(1, 3, 33, 32))

            def predict(self, x, batch_size=1):
                return np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])

            def get_activations(self):
                return None

            def save(self):
                pass

            def loss(self, x, y, **kwargs):
                pass

            def set_learning_phase(self):
                pass

            def input_shape(self):
                pass

        classifier = DummyClassifier()
        attack = Wasserstein(
            classifier,
            regularization=1,
            kernel_size=3,
            max_iter=1,
            conjugate_sinkhorn_max_iter=10,
            projected_sinkhorn_max_iter=10,
        )

        x = np.random.normal(size=(1, 3, 33, 32))
        x_adv = attack.generate(x)

        assert x_adv.shape == x.shape
    except ARTTestException as e:
        art_warning(e)


def test_classifier_type_check_fail(art_warning):
    try:
        backend_test_classifier_type_check_fail(Wasserstein, (BaseEstimator, LossGradientsMixin, ClassifierMixin))
    except ARTTestException as e:
        art_warning(e)


def test_image(fix_get_mnist_subset, image_dl_estimator, art_warning):
    try:
        (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset

        estimator, _ = image_dl_estimator()

        scores = get_labels_np_array(estimator.predict(x_train_mnist))
        acc = np.sum(np.argmax(scores, axis=1) == np.argmax(y_train_mnist, axis=1)) / y_train_mnist.shape[0]
        logger.info("[TF, MNIST] Accuracy on training set: %.2f%%", acc * 100)

        scores = get_labels_np_array(estimator.predict(x_test_mnist))
        acc = np.sum(np.argmax(scores, axis=1) == np.argmax(y_test_mnist, axis=1)) / y_test_mnist.shape[0]
        logger.info("[TF, MNIST] Accuracy on test set: %.2f%%", acc * 100)

        base_success_rate = 0.1
        num_iter = 5
        regularization = 100
        batch_size = 5
        eps = 0.3

        # Test Wasserstein with wasserstein ball and wasserstein norm
        attack = Wasserstein(
            estimator,
            regularization=regularization,
            max_iter=num_iter,
            conjugate_sinkhorn_max_iter=num_iter,
            projected_sinkhorn_max_iter=num_iter,
            norm="wasserstein",
            ball="wasserstein",
            targeted=False,
            p=2,
            eps_iter=2,
            eps_factor=1.05,
            eps=eps,
            eps_step=0.1,
            kernel_size=5,
            batch_size=batch_size,
        )

        x_train_adv = attack.generate(x_train_mnist)
        x_test_adv = attack.generate(x_test_mnist)

        # self.assertFalse((x_train_adv == x_train).all())
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, x_train_adv, x_train_mnist)
        # self.assertFalse((x_test_adv == x_test).all())
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, x_test_adv, x_test_mnist)

        train_y_pred = get_labels_np_array(estimator.predict(x_train_adv)).astype(float)
        test_y_pred = get_labels_np_array(estimator.predict(x_test_adv)).astype(float)

        train_success_rate = (
                np.sum(np.argmax(train_y_pred, axis=1) != np.argmax(estimator.predict(x_train_mnist), axis=1))
                / y_train_mnist.shape[0]
        )

        np.testing.assert_array_less(base_success_rate, train_success_rate)
        test_success_rate = (
                np.sum(np.argmax(test_y_pred, axis=1) != np.argmax(estimator.predict(x_test_mnist), axis=1))
                / y_test_mnist.shape[0]
        )

        np.testing.assert_array_less(base_success_rate, test_success_rate)
    except ARTTestException as e:
        art_warning(e)
