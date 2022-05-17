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
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import os
import unittest

import numpy as np
import tensorflow as tf
import torch

from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from art.attacks.evasion.fast_gradient import FastGradientMethod
from art.utils import load_dataset, random_targets, compute_accuracy
from art.estimators.certification.randomized_smoothing import (
    NumpyRandomizedSmoothing,
    TensorFlowV2RandomizedSmoothing,
    PyTorchRandomizedSmoothing,
)

from tests.utils import (
    master_seed,
    get_image_classifier_pt,
    get_image_classifier_kr,
    get_image_classifier_tf,
    get_tabular_classifier_pt,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
logger = logging.getLogger(__name__)

BATCH_SIZE = 100
NB_TRAIN = 5000
NB_TEST = 10


class TestTrainSmoothAdv(unittest.TestCase):
    """
    A unittest class for testing smooth adversarial classifier training.
    """

    @classmethod
    def setUpClass(cls):
        # Get MNIST
        (x_train, y_train), (x_test, y_test), _, _ = load_dataset("mnist")
        x_train, y_train = x_train[:NB_TRAIN], y_train[:NB_TRAIN]
        x_test, y_test = x_test[:NB_TEST], y_test[:NB_TEST]
        y_train = np.argmax(y_train, axis=1)
        y_test = np.argmax(y_test, axis=1)
        cls.mnist = (x_train, y_train), (x_test, y_test)

    def setUp(self):
        master_seed(seed=1234)

    def test_1_pt(self):
        """
        Test with a PyTorch Classifier.
        :return:
        """
        # Build PytorchClassifier
        ptc = get_image_classifier_pt(from_logits=True)

        # Get MNIST
        (_, _), (x_test, y_test) = self.mnist

        x_test = x_test.transpose(0, 3, 1, 2).astype(np.float32)

        # Initialize RS object
        optimizer=torch.optim.SGD(ptc.model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
        rs1 = PyTorchRandomizedSmoothing(
            model=ptc.model,
            loss=ptc._loss,
            optimizer=optimizer,
            scheduler=scheduler,
            input_shape=ptc.input_shape,
            nb_classes=ptc.nb_classes,
            channels_first=ptc.channels_first,
            clip_values=ptc.clip_values,
            sample_size=100,
            scale=0.25, 
            num_noise_vec= 10,
            train_multi_noise= True,
            attack_type="PGD",
            no_grad_attack = False,
            epsilon=1.0,
            num_steps=10,
            warmup=10
        )

        rs2 = PyTorchRandomizedSmoothing(
            model=ptc.model,
            loss=ptc._loss,
            optimizer=None,
            scheduler=scheduler,
            input_shape=ptc.input_shape,
            nb_classes=ptc.nb_classes,
            channels_first=ptc.channels_first,
            clip_values=ptc.clip_values,
            sample_size=100,
            scale=0.25, 
            num_noise_vec= 10,
            train_multi_noise= True,
            attack_type="PGD",
            no_grad_attack = False,
            epsilon=1.0,
            num_steps=10,
            warmup=10
        )
        
        rs3 = PyTorchRandomizedSmoothing(
            model=ptc.model,
            loss=ptc._loss,
            optimizer=optimizer,
            scheduler=None,
            input_shape=ptc.input_shape,
            nb_classes=ptc.nb_classes,
            channels_first=ptc.channels_first,
            clip_values=ptc.clip_values,
            sample_size=100,
            scale=0.25, 
            num_noise_vec= 10,
            train_multi_noise= True,
            attack_type="PGD",
            no_grad_attack = False,
            epsilon=1.0,
            num_steps=10,
            warmup=10
        )

        rs4 = PyTorchRandomizedSmoothing(
            model=ptc.model,
            loss=ptc._loss,
            optimizer=optimizer,
            scheduler=scheduler,
            input_shape=ptc.input_shape,
            nb_classes=ptc.nb_classes,
            channels_first=ptc.channels_first,
            clip_values=ptc.clip_values,
            sample_size=100,
            scale=0.25, 
            num_noise_vec= 10,
            train_multi_noise= True,
            attack_type="PGD",
            no_grad_attack = True,
            epsilon=1.0,
            num_steps=10,
            warmup=10
        )

        rs5 = PyTorchRandomizedSmoothing(
            model=ptc.model,
            loss=ptc._loss,
            optimizer=optimizer,
            scheduler=scheduler,
            input_shape=ptc.input_shape,
            nb_classes=ptc.nb_classes,
            channels_first=ptc.channels_first,
            clip_values=ptc.clip_values,
            sample_size=100,
            scale=0.25, 
            num_noise_vec= 10,
            train_multi_noise= True,
            attack_type="DDN",
            no_grad_attack = False,
            epsilon=1.0,
            num_steps=10,
            warmup=10
        )

        rs6 = PyTorchRandomizedSmoothing(
            model=ptc.model,
            loss=ptc._loss,
            optimizer=optimizer,
            scheduler=scheduler,
            input_shape=ptc.input_shape,
            nb_classes=ptc.nb_classes,
            channels_first=ptc.channels_first,
            clip_values=ptc.clip_values,
            sample_size=100,
            scale=0.25, 
            num_noise_vec= 1,
            train_multi_noise= True,
            attack_type="DDN",
            no_grad_attack = False,
            epsilon=1.0,
            num_steps=10,
            warmup=10
        )

        # fit
        rs1.fit(x_test, y_test, nb_epochs=1, batch_size=256, train_method = 'smoothadv')

        #fit fails when optimizer is None
        with self.assertRaisesRegexp(ValueError, 'An optimizer is needed to train the model, but none for provided.'):
            rs2.fit(x_test, y_test, nb_epochs=1, batch_size=256, train_method = 'smoothadv')

        #fit fails when scheduler is None
        with self.assertRaisesRegexp(ValueError, 'A scheduler is needed to train the model, but none for provided.'):
            rs3.fit(x_test, y_test, nb_epochs=1, batch_size=256, train_method = 'smoothadv')

        # fit with PGD attack and no grad
        rs4.fit(x_test, y_test, nb_epochs=1, batch_size=256, train_method = 'smoothadv')

        # fit with DDN attack and multi noise
        rs5.fit(x_test, y_test, nb_epochs=1, batch_size=256, train_method = 'smoothadv')

        # fit with DDN and single noise
        rs6.fit(x_test, y_test, nb_epochs=1, batch_size=256, train_method = 'smoothadv')

    def test_2_tf(self):
        """
        Test with a Smooth Adversarially trained TensorFlow Classifier.
        :return:
        """
        tf_version = list(map(int, tf.__version__.lower().split("+")[0].split(".")))
        if tf_version[0] == 2:

            # Build TensorFlowV2Classifier
            classifier, _ = get_image_classifier_tf()

            # Get MNIST
            (_, _), (x_test, y_test) = self.mnist

            # Initialize RS object
            initial_learning_rate = 0.1
            boundaries = [50, 100]
            values = [0.1, 0.01, 0.001]
            learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                boundaries, values)

            optimizer = tf.keras.optimizers.SGD(
              learning_rate=initial_learning_rate,
              momentum=0.9,
              name='SGD',
              decay=1e-4)

            loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

            rs1 = TensorFlowV2RandomizedSmoothing(
                    model=classifier.model,
                    nb_classes=classifier.nb_classes,
                    input_shape=classifier.input_shape,
                    loss_object=classifier.loss_object,
                    clip_values=classifier.clip_values,
                    scale=0.25,
                    num_noise_vec= 10,
                    train_multi_noise= True,
                    attack_type="PGD",
                    epsilon=1.0,
                    num_steps=10,
                    warmup=10,
                    optimizer = optimizer,
                    scheduler = learning_rate_fn
            )

            rs2 = TensorFlowV2RandomizedSmoothing(
                    model=classifier.model,
                    nb_classes=classifier.nb_classes,
                    input_shape=classifier.input_shape,
                    loss_object=classifier.loss_object,
                    clip_values=classifier.clip_values,
                    scale=0.25,
                    num_noise_vec= 10,
                    train_multi_noise= True,
                    attack_type="PGD",
                    epsilon=1.0,
                    num_steps=10,
                    warmup=10,
                    optimizer = None,
                    scheduler = learning_rate_fn
            )

            rs3 = TensorFlowV2RandomizedSmoothing(
                    model=classifier.model,
                    nb_classes=classifier.nb_classes,
                    input_shape=classifier.input_shape,
                    loss_object=classifier.loss_object,
                    clip_values=classifier.clip_values,
                    scale=0.25,
                    num_noise_vec= 10,
                    train_multi_noise= True,
                    attack_type="PGD",
                    epsilon=1.0,
                    num_steps=10,
                    warmup=10,
                    optimizer = optimizer,
                    scheduler = None
            )

            rs4 = TensorFlowV2RandomizedSmoothing(
                    model=classifier.model,
                    nb_classes=classifier.nb_classes,
                    input_shape=classifier.input_shape,
                    loss_object=classifier.loss_object,
                    clip_values=classifier.clip_values,
                    scale=0.25,
                    num_noise_vec= 10,
                    train_multi_noise= True,
                    attack_type="DDN",
                    epsilon=1.0,
                    num_steps=10,
                    warmup=10,
                    optimizer = optimizer,
                    scheduler = learning_rate_fn
            )

            rs5 = TensorFlowV2RandomizedSmoothing(
                    model=classifier.model,
                    nb_classes=classifier.nb_classes,
                    input_shape=classifier.input_shape,
                    loss_object=classifier.loss_object,
                    clip_values=classifier.clip_values,
                    scale=0.25,
                    num_noise_vec= 10,
                    train_multi_noise= True,
                    attack_type="PGD",
                    no_grad_attack=True,
                    epsilon=1.0,
                    num_steps=10,
                    warmup=10,
                    optimizer = optimizer,
                    scheduler = learning_rate_fn
            )

            # fit with PGD attack
            rs1.fit(x_test, y_test, nb_epochs=1, batch_size=256, train_method = 'smoothadv')

            # fit with PGD attack and no grad
            rs5.fit(x_test, y_test, nb_epochs=1, batch_size=256, train_method = 'smoothadv')

            # fit with DDN attack
            rs4.fit(x_test, y_test.astype(np.int32), nb_epochs=1, batch_size=256, train_method = 'smoothadv')

            #fit fails when optimizer is None
            with self.assertRaisesRegexp(ValueError, 'An optimizer is needed to train the model, but none for provided.'):
                rs2.fit(x_test, y_test, nb_epochs=1, batch_size=256, train_method = 'smoothadv')

            #fit fails when scheduler is None
            with self.assertRaisesRegexp(ValueError, 'A scheduler is needed to train the model, but none for provided.'):
                rs3.fit(x_test, y_test, nb_epochs=1, batch_size=256, train_method = 'smoothadv')

if __name__ == "__main__":
    unittest.main()
