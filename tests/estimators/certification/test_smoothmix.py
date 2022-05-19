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
import os
import unittest
from unittest.mock import MagicMock, ANY

import numpy as np
import tensorflow as tf
import torch

from art.attacks.evasion.fast_gradient import FastGradientMethod
from art.utils import load_dataset, random_targets, compute_accuracy
from art.estimators.certification.smoothmix import PyTorchSmoothMix

from tests.utils import (
    master_seed,
    get_image_classifier_pt,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
logger = logging.getLogger(__name__)

BATCH_SIZE = 100
NB_TRAIN = 5000
NB_TEST = 10


class TestSmoothMix(unittest.TestCase):
    """
    A unittest class for testing SmoothMix as a post-processing step for classifiers.
    """

    @classmethod
    def setUpClass(cls):
        # Get MNIST
        (x_train, y_train), (x_test, y_test), _, _ = load_dataset("mnist")
        x_train, y_train = x_train[:NB_TRAIN], y_train[:NB_TRAIN]
        x_test, y_test = x_test[:NB_TEST], y_test[:NB_TEST]
        cls.mnist = (x_train, y_train), (x_test, y_test)

    def setUp(self):
        master_seed(seed=1234)

    def test_1_smoothmix(self):
        """
        Test with a PyTorch classifier
        :return:
        """
        ptc = get_image_classifier_pt()

        # Get MNIST
        (_, _), (x_test, y_test) = self.mnist

        x_test = x_test.transpose(0, 3, 1, 2).astype(np.float32)

        # First FGSM attack:
        fgsm = FastGradientMethod(estimator=ptc, targeted=True)
        params = {"y": random_targets(y_test, ptc.nb_classes)}
        x_test_adv = fgsm.generate(x_test, **params)

        # Initialize RS object and attack with FGSM
        optimizer = torch.optim.SGD(ptc.model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=50, gamma=0.1)

        rs_constructor_kwargs = {
            "model": ptc.model,
            "loss": ptc._loss,
            "input_shape": ptc.input_shape,
            "nb_classes": ptc.nb_classes,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "clip_values": ptc.clip_values,
            "channels_first": ptc.channels_first,
            "sample_size": 100,
            "alpha": 0.001,
            "scale": 0.01,
            "num_noise_vec": 2,
            "attack_type": "PGD",
            "num_steps": 8,
            "warmup": 10,
            "eta": 5.0,
            "mix_step": 0,
            "maxnorm_s": None,
            "maxnorm": None,
        }

        # Check if PyTorchRandomizedSmoothing constructor is called with params needed to run SmoothMix training
        pt_rs_mock = MagicMock(side_effect=PyTorchSmoothMix, return_value=None)
        pt_rs_mock(
            model=ptc.model,
            loss=ptc._loss,
            input_shape=ptc.input_shape,
            nb_classes=ptc.nb_classes,
            optimizer=optimizer,
            scheduler=scheduler,
            clip_values=ptc.clip_values,
            channels_first=ptc.channels_first,
            sample_size=100,
            alpha=0.001,
            scale=0.01,
            num_noise_vec=2,
            attack_type="PGD",
            num_steps=8,
            warmup=10,
            eta=5.0,
            mix_step=0,
            maxnorm_s=None,
            maxnorm=None,
        )
        pt_rs_mock.assert_called_once_with(**rs_constructor_kwargs)

        rs = PyTorchSmoothMix(**rs_constructor_kwargs)

        fgsm_with_rs = FastGradientMethod(estimator=rs, targeted=True)
        x_test_adv_with_rs = fgsm_with_rs.generate(x_test, **params)

        # Compare results
        # check shapes are equal and values are within a certain range
        self.assertEqual(x_test_adv.shape, x_test_adv_with_rs.shape)
        self.assertTrue((np.abs(x_test_adv - x_test_adv_with_rs) < 0.75).all())

        # Check basic functionality of RS object
        # check predict
        y_test_smooth = rs.predict(x=x_test)
        y_test_base = ptc.predict(x=x_test)
        self.assertEqual(y_test_smooth.shape, y_test.shape)
        self.assertTrue((np.sum(y_test_smooth, axis=1) <= np.ones((NB_TEST,))).all())
        self.assertTrue((np.argmax(y_test_smooth, axis=1) == np.argmax(y_test_base, axis=1)).all())

        # check certification
        pred, radius = rs.certify(x=x_test, n=250)
        self.assertEqual(len(pred), NB_TEST)
        self.assertEqual(len(radius), NB_TEST)
        self.assertTrue((radius <= 1).all())
        self.assertTrue((pred < y_test.shape[1]).all())

        # loss gradient
        grad = rs.loss_gradient(x=x_test, y=y_test, sampling=True)
        assert grad.shape == (10, 1, 28, 28)

        # Check if fit method was called with  train_method = smoothmix
        rs_fit_kwargs = {
            "x": x_test,
            "y": y_test,
            "batch_size": ANY,
            "nb_epochs": ANY,
        }
        pt_rs_mock.fit(x=x_test, y=y_test, batch_size=128, nb_epochs=10)
        pt_rs_mock.fit.assert_called_once_with(**rs_fit_kwargs)

        # fit
        rs.fit(x=x_test, y=np.argmax(y_test, axis=1))


if __name__ == "__main__":
    unittest.main()
