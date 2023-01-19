# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2022
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
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_array_less

from art.estimators.certification.smoothmix import PyTorchSmoothMix
from tests.utils import ARTTestException, get_image_classifier_pt, get_cifar10_image_classifier_pt

logger = logging.getLogger(__name__)


@pytest.mark.only_with_platform("pytorch")
def test_smoothmix_pytorch_mnist_certification(art_warning, get_default_mnist_subset):
    import torch

    (_, _), (x_test, y_test) = get_default_mnist_subset
    ptc = get_image_classifier_pt()
    optimizer = torch.optim.SGD(ptc.model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    try:
        classifier = PyTorchSmoothMix(
            model=ptc.model,
            loss=ptc._loss,
            input_shape=ptc.input_shape,
            nb_classes=ptc.nb_classes,
            optimizer=optimizer,
            clip_values=ptc.clip_values,
            channels_first=ptc.channels_first,
            sample_size=100,
            alpha=0.001,
            scale=0.01,
            num_noise_vec=2,
            num_steps=8,
            warmup=10,
            eta=5.0,
            mix_step=0,
            maxnorm_s=None,
            maxnorm=None,
        )
        pred, radius = classifier.certify(x=x_test, n=250)

        assert_array_equal(pred.shape, radius.shape)
        assert_array_less(radius, 1)
        assert_array_less(pred, y_test.shape[1])
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_smoothmix_pytorch_cifar10_certification(art_warning, get_default_cifar10_subset):
    import torch

    (_, _), (x_test, y_test) = get_default_cifar10_subset
    ptc = get_cifar10_image_classifier_pt()
    optimizer = torch.optim.SGD(ptc.model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    try:
        classifier = PyTorchSmoothMix(
            model=ptc.model,
            loss=ptc._loss,
            input_shape=ptc.input_shape,
            nb_classes=ptc.nb_classes,
            optimizer=optimizer,
            clip_values=ptc.clip_values,
            channels_first=ptc.channels_first,
            sample_size=100,
            alpha=0.001,
            scale=0.01,
            num_noise_vec=2,
            num_steps=8,
            warmup=10,
            eta=5.0,
            mix_step=0,
            maxnorm_s=None,
            maxnorm=None,
        )
        pred, radius = classifier.certify(x=x_test, n=250)

        assert_array_equal(pred.shape, radius.shape)
        assert_array_less(radius, 1)
        assert_array_less(pred, y_test.shape[1])
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_smoothmix_pytorch_mnist_predict(art_warning, get_default_mnist_subset):
    import torch

    (_, _), (x_test, y_test) = get_default_mnist_subset
    ptc = get_image_classifier_pt()
    optimizer = torch.optim.SGD(ptc.model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    try:
        classifier = PyTorchSmoothMix(
            model=ptc.model,
            loss=ptc._loss,
            input_shape=ptc.input_shape,
            nb_classes=ptc.nb_classes,
            optimizer=optimizer,
            clip_values=ptc.clip_values,
            channels_first=ptc.channels_first,
            sample_size=100,
            alpha=0.001,
            scale=0.01,
            num_noise_vec=2,
            num_steps=8,
            warmup=10,
            eta=5.0,
            mix_step=0,
            maxnorm_s=None,
            maxnorm=None,
        )
        y_test_smooth = classifier.predict(x=x_test)
        y_test_base = ptc.predict(x=x_test)

        assert_array_equal(y_test_smooth.shape, y_test_base.shape)
        assert_array_almost_equal(np.sum(y_test_smooth, axis=1), np.ones((len(y_test))))
        assert_array_almost_equal(np.sum(y_test_base, axis=1), np.ones((len(y_test))))
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_smoothmix_pytorch_cifar10_predict(art_warning, get_default_cifar10_subset):
    import torch

    (_, _), (x_test, y_test) = get_default_cifar10_subset
    ptc = get_cifar10_image_classifier_pt()
    optimizer = torch.optim.SGD(ptc.model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    try:
        classifier = PyTorchSmoothMix(
            model=ptc.model,
            loss=ptc._loss,
            input_shape=ptc.input_shape,
            nb_classes=ptc.nb_classes,
            optimizer=optimizer,
            clip_values=ptc.clip_values,
            channels_first=ptc.channels_first,
            sample_size=100,
            alpha=0.001,
            scale=0.01,
            num_noise_vec=2,
            num_steps=8,
            warmup=10,
            eta=5.0,
            mix_step=0,
            maxnorm_s=None,
            maxnorm=None,
        )
        y_test_smooth = classifier.predict(x=x_test)
        y_test_base = ptc.predict(x=x_test)

        assert_array_equal(y_test_smooth.shape, y_test_base.shape)
        assert_array_almost_equal(np.sum(y_test_smooth, axis=1), np.ones((len(y_test))))
        assert_array_almost_equal(np.sum(y_test_base, axis=1), np.ones((len(y_test))))
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_smoothmix_pytorch_mnist_fit(art_warning, get_default_mnist_subset):
    import torch

    (_, _), (x_test, y_test) = get_default_mnist_subset
    ptc = get_image_classifier_pt()
    optimizer = torch.optim.SGD(ptc.model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=50, gamma=0.1)

    try:
        classifier = PyTorchSmoothMix(
            model=ptc.model,
            loss=ptc._loss,
            input_shape=ptc.input_shape,
            nb_classes=ptc.nb_classes,
            optimizer=optimizer,
            clip_values=ptc.clip_values,
            channels_first=ptc.channels_first,
            sample_size=100,
            alpha=0.001,
            scale=0.01,
            num_noise_vec=2,
            num_steps=8,
            warmup=10,
            eta=5.0,
            mix_step=0,
            maxnorm_s=None,
            maxnorm=None,
        )
        classifier.fit(x=x_test, y=y_test, batch_size=128, nb_epochs=1, scheduler=scheduler)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_smoothmix_pytorch_cifar10_fit(art_warning, get_default_cifar10_subset):
    import torch

    (_, _), (x_test, y_test) = get_default_cifar10_subset
    ptc = get_cifar10_image_classifier_pt()
    optimizer = torch.optim.SGD(ptc.model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=50, gamma=0.1)

    try:
        classifier = PyTorchSmoothMix(
            model=ptc.model,
            loss=ptc._loss,
            input_shape=ptc.input_shape,
            nb_classes=ptc.nb_classes,
            optimizer=optimizer,
            clip_values=ptc.clip_values,
            channels_first=ptc.channels_first,
            sample_size=100,
            alpha=0.001,
            scale=0.01,
            num_noise_vec=2,
            num_steps=8,
            warmup=10,
            eta=5.0,
            mix_step=0,
            maxnorm_s=None,
            maxnorm=None,
        )
        classifier.fit(x=x_test, y=y_test, batch_size=128, nb_epochs=1, scheduler=scheduler)
    except ARTTestException as e:
        art_warning(e)
