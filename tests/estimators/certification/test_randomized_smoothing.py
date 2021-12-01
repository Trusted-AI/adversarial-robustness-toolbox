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


class TestRandomizedSmoothing(unittest.TestCase):
    """
    A unittest class for testing Randomized Smoothing as a post-processing step for classifiers.
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

    def test_3_kr(self):
        """
        Test with a Keras Classifier.
        :return:
        """
        # Build KerasClassifier
        classifier = get_image_classifier_kr()

        # Get MNIST
        (_, _), (x_test, y_test) = self.mnist

        # First FGSM attack:
        fgsm = FastGradientMethod(estimator=classifier, targeted=True)
        params = {"y": random_targets(y_test, classifier.nb_classes)}
        x_test_adv = fgsm.generate(x_test, **params)

        # Initialize RS object and attack with FGSM
        rs = NumpyRandomizedSmoothing(
            classifier=classifier,
            sample_size=100,
            scale=0.01,
            alpha=0.001,
        )
        fgsm_with_rs = FastGradientMethod(estimator=rs, targeted=True)
        x_test_adv_with_rs = fgsm_with_rs.generate(x_test, **params)

        # Compare results
        # check shapes are equal and values are within a certain range
        self.assertEqual(x_test_adv.shape, x_test_adv_with_rs.shape)
        self.assertTrue((np.abs(x_test_adv - x_test_adv_with_rs) < 0.75).all())

        # Check basic functionality of RS object
        # check predict
        y_test_smooth = rs.predict(x=x_test)
        y_test_base = classifier.predict(x=x_test)
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
        assert grad.shape == (10, 28, 28, 1)

        # fit
        rs.fit(x=x_test, y=y_test)

    def test_1_tf(self):
        """
        Test with a TensorFlow Classifier.
        :return:
        """
        tf_version = list(map(int, tf.__version__.lower().split("+")[0].split(".")))
        if tf_version[0] == 2:

            # Build TensorFlowV2Classifier
            classifier, _ = get_image_classifier_tf()

            # Get MNIST
            (_, _), (x_test, y_test) = self.mnist

            # First FGSM attack:
            fgsm = FastGradientMethod(estimator=classifier, targeted=True)
            params = {"y": random_targets(y_test, classifier.nb_classes)}
            x_test_adv = fgsm.generate(x_test, **params)

            # Initialize RS object and attack with FGSM
            rs = TensorFlowV2RandomizedSmoothing(
                model=classifier.model,
                nb_classes=classifier.nb_classes,
                input_shape=classifier.input_shape,
                loss_object=classifier.loss_object,
                train_step=classifier.train_step,
                channels_first=classifier.channels_first,
                clip_values=classifier.clip_values,
                preprocessing_defences=classifier.preprocessing_defences,
                postprocessing_defences=classifier.postprocessing_defences,
                preprocessing=classifier.preprocessing,
                sample_size=100,
                scale=0.01,
                alpha=0.001,
            )
            fgsm_with_rs = FastGradientMethod(estimator=rs, targeted=True)
            x_test_adv_with_rs = fgsm_with_rs.generate(x_test, **params)

            # Compare results
            # check shapes are equal and values are within a certain range
            self.assertEqual(x_test_adv.shape, x_test_adv_with_rs.shape)
            self.assertTrue((np.abs(x_test_adv - x_test_adv_with_rs) < 0.75).all())

            # Check basic functionality of RS object
            # check predict
            y_test_smooth = rs.predict(x=x_test)
            y_test_base = classifier.predict(x=x_test)
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
            assert grad.shape == (10, 28, 28, 1)

            # fit
            rs.fit(x=x_test, y=y_test)

    def test_2_pt(self):
        """
        Test with a PyTorch Classifier.
        :return:
        """
        # Build KerasClassifier
        ptc = get_image_classifier_pt()

        # Get MNIST
        (_, _), (x_test, y_test) = self.mnist

        x_test = x_test.transpose(0, 3, 1, 2).astype(np.float32)

        # First FGSM attack:
        fgsm = FastGradientMethod(estimator=ptc, targeted=True)
        params = {"y": random_targets(y_test, ptc.nb_classes)}
        x_test_adv = fgsm.generate(x_test, **params)

        # Initialize RS object and attack with FGSM
        rs = PyTorchRandomizedSmoothing(
            model=ptc.model,
            loss=ptc._loss,
            optimizer=torch.optim.Adam(ptc.model.parameters(), lr=0.01),
            input_shape=ptc.input_shape,
            nb_classes=ptc.nb_classes,
            channels_first=ptc.channels_first,
            clip_values=ptc.clip_values,
            sample_size=100,
            scale=0.01,
            alpha=0.001,
        )
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

        # fit
        rs.fit(x=x_test, y=y_test)


class TestRandomizedSmoothingVectors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Get Iris
        (x_train, y_train), (x_test, y_test), _, _ = load_dataset("iris")
        cls.iris = (x_train, y_train), (x_test, y_test)

    def setUp(self):
        master_seed(seed=1234)

    def test_iris_clipped(self):
        (_, _), (x_test, y_test) = self.iris

        ptc = get_tabular_classifier_pt()
        rs = PyTorchRandomizedSmoothing(
            model=ptc.model,
            loss=ptc._loss,
            input_shape=ptc.input_shape,
            nb_classes=ptc.nb_classes,
            channels_first=ptc.channels_first,
            clip_values=ptc.clip_values,
            sample_size=100,
            scale=0.01,
            alpha=0.001,
        )

        # Test untargeted attack
        attack = FastGradientMethod(ptc, eps=0.1)
        x_test_adv = attack.generate(x_test)
        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1).all())
        self.assertTrue((x_test_adv >= 0).all())

        preds_smooth = np.argmax(rs.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(y_test, axis=1) == preds_smooth).all())

        pred = rs.predict(x_test)
        pred2 = rs.predict(x_test_adv)
        acc, cov = compute_accuracy(pred, y_test)
        acc2, cov2 = compute_accuracy(pred2, y_test)
        logger.info("Accuracy on Iris with smoothing on adversarial examples: %.2f%%", (acc * 100))
        logger.info("Coverage on Iris with smoothing on adversarial examples: %.2f%%", (cov * 100))
        logger.info("Accuracy on Iris with smoothing: %.2f%%", (acc2 * 100))
        logger.info("Coverage on Iris with smoothing: %.2f%%", (cov2 * 100))

        # Check basic functionality of RS object
        # check predict
        y_test_smooth = rs.predict(x=x_test)
        self.assertEqual(y_test_smooth.shape, y_test.shape)
        self.assertTrue((np.sum(y_test_smooth, axis=1) <= 1).all())

        # check certification
        pred, radius = rs.certify(x=x_test, n=250)
        self.assertEqual(len(pred), len(x_test))
        self.assertEqual(len(radius), len(x_test))
        self.assertTrue((radius <= 1).all())
        self.assertTrue((pred < y_test.shape[1]).all())


if __name__ == "__main__":
    unittest.main()
