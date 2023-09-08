# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2023
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

from art.estimators.certification.randomized_smoothing import PyTorchSmoothMix
from tests.utils import ARTTestException, get_image_classifier_pt, get_cifar10_image_classifier_pt

logger = logging.getLogger(__name__)


@pytest.fixture()
def get_mnist_classifier(framework):
    def _get_classifier():
        if framework == "pytorch":
            import torch

            classifier = get_image_classifier_pt()
            optimizer = torch.optim.SGD(classifier.model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=50, gamma=0.1)
            rs = PyTorchSmoothMix(
                model=classifier.model,
                loss=classifier._loss,
                input_shape=classifier.input_shape,
                nb_classes=classifier.nb_classes,
                optimizer=optimizer,
                clip_values=classifier.clip_values,
                channels_first=classifier.channels_first,
                sample_size=100,
                scale=0.01,
                alpha=0.001,
                eta=5.0,
                num_noise_vec=2,
                num_steps=8,
                warmup=10,
                mix_step=0,
                maxnorm_s=None,
                maxnorm=None,
            )

        else:
            classifier, scheduler, rs = None, None, None

        return classifier, scheduler, rs

    return _get_classifier


@pytest.fixture()
def get_cifar10_classifier(framework):
    def _get_classifier():
        if framework == "pytorch":
            import torch

            classifier = get_cifar10_image_classifier_pt()
            optimizer = torch.optim.SGD(classifier.model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=50, gamma=0.1)
            rs = PyTorchSmoothMix(
                model=classifier.model,
                loss=classifier._loss,
                input_shape=classifier.input_shape,
                nb_classes=classifier.nb_classes,
                optimizer=optimizer,
                clip_values=classifier.clip_values,
                channels_first=classifier.channels_first,
                sample_size=100,
                scale=0.01,
                alpha=0.001,
                eta=5.0,
                num_noise_vec=2,
                num_steps=8,
                warmup=10,
                mix_step=0,
                maxnorm_s=None,
                maxnorm=None,
            )

        else:
            classifier, scheduler, rs = None, None, None

        return classifier, scheduler, rs

    return _get_classifier


@pytest.mark.only_with_platform("pytorch")
def test_smoothmix_mnist_predict(art_warning, get_default_mnist_subset, get_mnist_classifier):
    (_, _), (x_test, y_test) = get_default_mnist_subset
    x_test, y_test = x_test[:10], y_test[:10]

    try:
        classifier, _, rs = get_mnist_classifier()
        y_test_base = classifier.predict(x=x_test)
        y_test_smooth = rs.predict(x=x_test)

        np.testing.assert_array_equal(y_test_smooth.shape, y_test_base.shape)
        np.testing.assert_array_almost_equal(np.sum(y_test_smooth, axis=1), np.ones(len(y_test)))
        np.testing.assert_array_almost_equal(np.argmax(y_test_smooth, axis=1), np.argmax(y_test_base, axis=1))

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_smoothmix_mnist_fit(art_warning, get_default_mnist_subset, get_mnist_classifier):
    (_, _), (x_test, y_test) = get_default_mnist_subset
    x_test, y_test = x_test[:10], y_test[:10]

    try:
        _, scheduler, rs = get_mnist_classifier()
        rs.fit(x=x_test, y=y_test, batch_size=128, nb_epochs=1, scheduler=scheduler)

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_smoothmix_mnist_certification(art_warning, get_default_mnist_subset, get_mnist_classifier):
    (_, _), (x_test, y_test) = get_default_mnist_subset
    x_test, y_test = x_test[:10], y_test[:10]

    try:
        _, _, rs = get_mnist_classifier()
        pred, radius = rs.certify(x=x_test, n=250)

        np.testing.assert_array_equal(pred.shape, radius.shape)
        np.testing.assert_array_less(radius, 1)
        np.testing.assert_array_less(pred, y_test.shape[1])

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_smoothmix_cifar10_predict(art_warning, get_default_cifar10_subset, get_cifar10_classifier):
    (_, _), (x_test, y_test) = get_default_cifar10_subset
    x_test, y_test = x_test[:10], y_test[:10]

    try:
        classifier, _, rs = get_cifar10_classifier()
        y_test_base = classifier.predict(x=x_test)
        y_test_smooth = rs.predict(x=x_test)

        np.testing.assert_array_equal(y_test_smooth.shape, y_test_base.shape)
        np.testing.assert_array_almost_equal(np.sum(y_test_smooth, axis=1), np.ones((len(y_test))))
        np.testing.assert_array_almost_equal(np.argmax(y_test_smooth, axis=1), np.argmax(y_test_base, axis=1))

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_smoothmix_cifar10_fit(art_warning, get_default_cifar10_subset, get_cifar10_classifier):
    (_, _), (x_test, y_test) = get_default_cifar10_subset
    x_test, y_test = x_test[:10], y_test[:10]

    try:
        _, scheduler, rs = get_cifar10_classifier()
        rs.fit(x=x_test, y=y_test, batch_size=128, nb_epochs=1, scheduler=scheduler)

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_smoothmix_cifar10_certification(art_warning, get_default_cifar10_subset, get_cifar10_classifier):
    (_, _), (x_test, y_test) = get_default_cifar10_subset
    x_test, y_test = x_test[:10], y_test[:10]

    try:
        _, _, rs = get_cifar10_classifier()
        pred, radius = rs.certify(x=x_test, n=250)

        np.testing.assert_array_equal(pred.shape, radius.shape)
        np.testing.assert_array_less(radius, 1)
        np.testing.assert_array_less(pred, y_test.shape[1])

    except ARTTestException as e:
        art_warning(e)
