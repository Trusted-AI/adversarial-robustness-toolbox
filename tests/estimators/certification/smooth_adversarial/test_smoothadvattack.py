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
"""Unit Test Module for testing smooth adversarial classifier training"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import os
import pytest

import numpy as np

from art.utils import load_dataset

from tests.utils import master_seed, get_image_classifier_pt, get_image_classifier_tf_v2

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
logger = logging.getLogger(__name__)

BATCH_SIZE = 100
NB_TRAIN = 5000
NB_TEST = 10


@pytest.fixture()
def get_mnist_data():
    # Get MNIST
    (x_train, y_train), (x_test, y_test), _, _ = load_dataset("mnist")
    x_train, y_train = x_train[:NB_TRAIN], y_train[:NB_TRAIN]
    x_test, y_test = x_test[:NB_TEST], y_test[:NB_TEST]
    y_train = np.argmax(y_train, axis=1)
    y_test = np.argmax(y_test, axis=1)
    return (x_train, y_train), (x_test, y_test)


@pytest.fixture()
def set_seed():
    master_seed(seed=1234)


@pytest.mark.only_with_platform("pytorch")
def test_1_pt(get_mnist_data):
    """
    Test smooth adversarial attack.
    :return:
    """
    import torch
    from art.estimators.certification.randomized_smoothing.smooth_adversarial.smoothadvattack import PgdL2

    epoch_num = 1
    warmup = 10
    epsilon = 1.0
    num_steps = 10
    scale = 0.25
    num_noise_vec = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Build PytorchClassifier
    ptc = get_image_classifier_pt(from_logits=True)

    # Get MNIST
    (_, _), (x_test, y_test) = get_mnist_data

    x_test = x_test.transpose(0, 3, 1, 2).astype(np.float32)

    attacker_pgd = PgdL2(steps=num_steps, device=device, max_norm=epsilon)
    attacker_pgd.max_norm = np.min([epsilon, (epoch_num + 1) * epsilon / warmup])
    attacker_pgd.init_norm = np.min([epsilon, (epoch_num + 1) * epsilon / warmup])

    noise = torch.randn_like(torch.from_numpy(x_test), device=device) * scale
    inputs_attacked = attacker_pgd.attack(
        ptc.model,
        torch.from_numpy(x_test).to(device),
        torch.from_numpy(y_test).to(device),
        noise=noise,
        num_noise_vectors=num_noise_vec,
        no_grad=False,
    )
    # Checking if attacked inputs and inputs shapes are same
    assert inputs_attacked.shape == x_test.shape


@pytest.mark.only_with_platform("pytorch")
def test_2_pt(get_mnist_data):
    """
    Test smooth adversarial attack using DDN.
    :return:
    """
    import torch
    from art.estimators.certification.randomized_smoothing.smooth_adversarial.smoothadvattack import DDN

    epoch_num = 1
    warmup = 10
    epsilon = 1.0
    num_steps = 10
    scale = 0.25
    num_noise_vec = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Build PytorchClassifier
    ptc = get_image_classifier_pt(from_logits=True)

    # Get MNIST
    (_, _), (x_test, y_test) = get_mnist_data

    x_test = x_test.transpose(0, 3, 1, 2).astype(np.float32)

    attacker_pgd = DDN(steps=num_steps, device=device, max_norm=epsilon)
    attacker_pgd.max_norm = np.min([epsilon, (epoch_num + 1) * epsilon / warmup])
    attacker_pgd.init_norm = np.min([epsilon, (epoch_num + 1) * epsilon / warmup])

    noise = torch.randn_like(torch.from_numpy(x_test), device=device) * scale
    inputs_attacked = attacker_pgd.attack(
        ptc.model,
        torch.from_numpy(x_test).to(device),
        torch.from_numpy(y_test).to(device),
        noise=noise,
        num_noise_vectors=num_noise_vec,
        no_grad=False,
    )
    # Checking if attacked inputs and inputs shapes are same
    assert inputs_attacked.shape == x_test.shape


@pytest.mark.only_with_platform("tensorflow", "tensorflow2", "keras", "kerastf")
def test_3_tf(get_mnist_data):
    """
    Test smooth adversarial attack.
    :return:
    """
    import tensorflow as tf
    from art.estimators.certification.randomized_smoothing.smooth_adversarial.smoothadvattack_tensorflow import (
        PgdL2,
    )

    epoch_num = 1
    warmup = 10
    epsilon = 1.0
    num_steps = 10
    scale = 0.25
    num_noise_vec = 1

    tf_version = list(map(int, tf.__version__.lower().split("+")[0].split(".")))
    if tf_version[0] == 2:

        # Build TensorFlowV2Classifier
        tf.compat.v1.enable_eager_execution()
        classifier = get_image_classifier_tf_v2()

        if tf.executing_eagerly():
            # Get MNIST
            (_, _), (x_test, y_test) = get_mnist_data
            x_test = x_test.transpose(0, 3, 1, 2).astype(np.float32)

            attacker_pgd = PgdL2(steps=num_steps, max_norm=epsilon)
            attacker_pgd.max_norm = np.min([epsilon, (epoch_num + 1) * epsilon / warmup])
            attacker_pgd.init_norm = np.min([epsilon, (epoch_num + 1) * epsilon / warmup])

            noise = tf.random.normal(x_test.shape, 0, 1) * scale
            inputs_attacked = attacker_pgd.attack(
                classifier.model,
                tf.convert_to_tensor(x_test),
                tf.cast(tf.convert_to_tensor(y_test), tf.int32),
                noise=noise,
                num_noise_vectors=num_noise_vec,
                no_grad=False,
            )
            # Checking if attacked inputs and inputs shapes are same
            assert inputs_attacked.shape == x_test.shape


@pytest.mark.only_with_platform("tensorflow", "tensorflow2", "keras", "kerastf")
def test_4_tf(get_mnist_data):
    """
    Test smooth adversarial attack using DDN.
    :return:
    """
    import tensorflow as tf
    from art.estimators.certification.randomized_smoothing.smooth_adversarial.smoothadvattack_tensorflow import DDN

    epoch_num = 1
    warmup = 10
    epsilon = 1.0
    num_steps = 10
    scale = 0.25
    num_noise_vec = 1

    tf_version = list(map(int, tf.__version__.lower().split("+")[0].split(".")))
    if tf_version[0] == 2:

        # Build TensorFlowV2Classifier
        tf.compat.v1.enable_eager_execution()
        classifier = get_image_classifier_tf_v2()

        if tf.executing_eagerly():
            # Get MNIST
            (_, _), (x_test, y_test) = get_mnist_data
            x_test = x_test.transpose(0, 3, 1, 2).astype(np.float32)

            attacker_pgd = DDN(steps=num_steps, max_norm=epsilon)
            attacker_pgd.max_norm = np.min([epsilon, (epoch_num + 1) * epsilon / warmup])
            attacker_pgd.init_norm = np.min([epsilon, (epoch_num + 1) * epsilon / warmup])

            noise = tf.random.normal(x_test.shape, 0, 1) * scale
            inputs_attacked = attacker_pgd.attack(
                classifier.model,
                tf.convert_to_tensor(x_test),
                tf.cast(tf.convert_to_tensor(y_test), tf.int32),
                noise=noise,
                num_noise_vectors=num_noise_vec,
                no_grad=False,
            )
            # Checking if attacked inputs and inputs shapes are same
            assert inputs_attacked.shape == x_test.shape
