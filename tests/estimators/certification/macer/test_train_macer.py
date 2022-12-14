# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2019
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
"""Unit Test Module for testing macer classifier training"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import os
import pytest
import numpy as np
from art.utils import load_dataset
from art.estimators.certification.randomized_smoothing import (
    TensorFlowV2RandomizedSmoothing,
    PyTorchRandomizedSmoothing,
)
from tests.utils import master_seed, get_image_classifier_pt, get_image_classifier_tf_v2

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
logger = logging.getLogger(__name__)


@pytest.fixture()
def get_mnist_data():
    # Get MNIST
    NB_TEST = 10

    (_, _), (x_test, y_test), _, _ = load_dataset("mnist")
    x_test, y_test = x_test[:NB_TEST], y_test[:NB_TEST]
    y_test = np.argmax(y_test, axis=1)
    return x_test, y_test


@pytest.fixture()
def set_seed():
    master_seed(seed=1234)


@pytest.mark.only_with_platform("pytorch")
def test_macer_randomized_smoothing_pytorch(get_mnist_data):
    """
    Test with a PyTorch Classifier.
    :return:
    """

    from torch import optim
    from torch.optim.lr_scheduler import MultiStepLR

    # Build PytorchClassifier
    ptc = get_image_classifier_pt(from_logits=True)

    # Get MNIST
    x_test, y_test = get_mnist_data

    x_test = x_test.transpose(0, 3, 1, 2).astype(np.float32)

    # Initialize RS object
    optimizer = optim.SGD(ptc.model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[200, 400], gamma=0.1)
    rs1 = PyTorchRandomizedSmoothing(
        model=ptc.model,
        optimizer=optimizer,
        scheduler=scheduler,
        input_shape=ptc.input_shape,
        nb_classes=ptc.nb_classes,
        channels_first=ptc.channels_first,
        clip_values=ptc.clip_values,
        scale=0.25,
        lbd=12.0,
        gamma=8.0,
        beta=16.0,
        gauss_num=16,
    )

    # fit
    rs1.fit(x_test, y_test, nb_epochs=1, batch_size=256, train_method="macer")


@pytest.mark.only_with_platform("pytorch")
def test_macer_randomized_smoothing_pytorch_no_optimizer(get_mnist_data):
    """
    Test with a PyTorch Classifier.
    :return:
    """

    from torch import optim
    from torch.optim.lr_scheduler import MultiStepLR

    # Build PytorchClassifier
    ptc = get_image_classifier_pt(from_logits=True)

    # Get MNIST
    x_test, y_test = get_mnist_data

    x_test = x_test.transpose(0, 3, 1, 2).astype(np.float32)

    # Initialize RS object
    optimizer = optim.SGD(ptc.model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[200, 400], gamma=0.1)
    rs2 = PyTorchRandomizedSmoothing(
        model=ptc.model,
        optimizer=None,
        scheduler=scheduler,
        input_shape=ptc.input_shape,
        nb_classes=ptc.nb_classes,
        channels_first=ptc.channels_first,
        clip_values=ptc.clip_values,
        scale=0.25,
        lbd=12.0,
        gamma=8.0,
        beta=16.0,
        gauss_num=16,
    )

    # fit fails when optimizer is None
    with pytest.raises(ValueError, match="An optimizer is needed to train the model, but none for provided."):
        rs2.fit(x_test, y_test, nb_epochs=1, batch_size=256, train_method="macer")


@pytest.mark.only_with_platform("pytorch")
def test_macer_randomized_smoothing_pytorch_no_scheduler(get_mnist_data):
    """
    Test with a PyTorch Classifier.
    :return:
    """

    from torch import optim

    # Build PytorchClassifier
    ptc = get_image_classifier_pt(from_logits=True)

    # Get MNIST
    x_test, y_test = get_mnist_data

    x_test = x_test.transpose(0, 3, 1, 2).astype(np.float32)

    # Initialize RS object
    optimizer = optim.SGD(ptc.model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    rs3 = PyTorchRandomizedSmoothing(
        model=ptc.model,
        optimizer=optimizer,
        scheduler=None,
        input_shape=ptc.input_shape,
        nb_classes=ptc.nb_classes,
        channels_first=ptc.channels_first,
        clip_values=ptc.clip_values,
        scale=0.25,
        lbd=12.0,
        gamma=8.0,
        beta=16.0,
        gauss_num=16,
    )

    # fit fails when scheduler is None
    with pytest.raises(ValueError, match="A scheduler is needed to train the model, but none for provided."):
        rs3.fit(x_test, y_test, nb_epochs=1, batch_size=256, train_method="macer")


# MACER implementation not compatible with Tensorflow v1
@pytest.mark.only_with_platform("tensorflow2")
def test_macer_randomized_smoothing_tensorflow(get_mnist_data):
    """
    Test with a Tensorflow Classifier.
    :return:
    """
    import tensorflow as tf

    ptc = get_image_classifier_tf_v2()

    # Get MNIST
    x_test, y_test = get_mnist_data
    x_test = x_test.astype("float32")
    y_test = y_test.astype("int32")

    # Initialize RS object
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, name="SGD", decay=5e-4)
    boundaries = [250, 400]
    values = [0.01, 0.001, 0.0001]
    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
    rs1 = TensorFlowV2RandomizedSmoothing(
        model=ptc.model,
        optimizer=optimizer,
        scheduler=learning_rate_fn,
        input_shape=ptc.input_shape,
        nb_classes=ptc.nb_classes,
        clip_values=ptc.clip_values,
        scale=0.25,
        lbd=12.0,
        gamma=8.0,
        beta=16.0,
        gauss_num=16,
    )
    if tf.executing_eagerly():
        # fit
        rs1.fit(x_test, y_test, nb_epochs=1, batch_size=256, train_method="macer")


#  MACER implementation not compatible with Tensorflow v1
@pytest.mark.only_with_platform("tensorflow2")
def test_macer_randomized_smoothing_tensorflow_no_optimizer(get_mnist_data):
    """
    Test with a Tensorflow Classifier.
    :return:
    """
    import tensorflow as tf

    ptc = get_image_classifier_tf_v2()

    # Get MNIST
    x_test, y_test = get_mnist_data
    x_test = x_test.astype("float32")
    y_test = y_test.astype("int32")

    # Initialize RS object
    boundaries = [250, 400]
    values = [0.01, 0.001, 0.0001]
    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
    rs2 = TensorFlowV2RandomizedSmoothing(
        model=ptc.model,
        optimizer=None,
        scheduler=learning_rate_fn,
        input_shape=ptc.input_shape,
        nb_classes=ptc.nb_classes,
        clip_values=ptc.clip_values,
        scale=0.25,
        lbd=12.0,
        gamma=8.0,
        beta=16.0,
        gauss_num=16,
    )

    # fit fails when optimizer is None
    with pytest.raises(ValueError, match="An optimizer is needed to train the model, but none for provided."):
        rs2.fit(x_test, y_test, nb_epochs=1, batch_size=256, train_method="macer")


# MACER implementation not compatible with Tensorflow v1
@pytest.mark.only_with_platform("tensorflow2")
def test_macer_randomized_smoothing_tensorflow_no_scheduler(get_mnist_data):
    """
    Test with a Tensorflow Classifier.
    :return:
    """
    import tensorflow as tf

    ptc = get_image_classifier_tf_v2()

    # Get MNIST
    x_test, y_test = get_mnist_data
    x_test = x_test.astype("float32")
    y_test = y_test.astype("int32")

    # Initialize RS object
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, name="SGD", decay=5e-4)

    rs3 = TensorFlowV2RandomizedSmoothing(
        model=ptc.model,
        optimizer=optimizer,
        scheduler=None,
        input_shape=ptc.input_shape,
        nb_classes=ptc.nb_classes,
        clip_values=ptc.clip_values,
        scale=0.25,
        lbd=12.0,
        gamma=8.0,
        beta=16.0,
        gauss_num=16,
    )

    # fit fails when scheduler is None
    with pytest.raises(ValueError, match="A scheduler is needed to train the model, but none for provided."):
        rs3.fit(x_test, y_test, nb_epochs=1, batch_size=256, train_method="macer")
