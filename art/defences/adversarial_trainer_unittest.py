# MIT License
#
# Copyright (C) IBM Corporation 2018
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

import unittest

import keras.backend as k
import numpy as np
import tensorflow as tf

from art.attacks.fast_gradient import FastGradientMethod
from art.attacks.deepfool import DeepFool
from art.classifiers.cnn import CNN
from art.defences.adversarial_trainer import AdversarialTrainer
from art.utils import load_dataset

BATCH_SIZE = 10
NB_TRAIN = 1000
NB_TEST = 100


class TestAdversarialTrainer(unittest.TestCase):
    """
    Test cases for the AdversarialTrainer class.
    """
    def test_one_attack_mnist(self):
        """
        Test the adversarial trainer using one FGSM attacker. The source and target models of the attack
        are two CNNs on MNIST trained for 5 epochs. The test cast check if accuracy on adversarial samples increases
        after adversarially training the model.

        :return: None
        """
        session = tf.Session()
        k.set_session(session)

        # Load MNIST
        (x_train, y_train), (x_test, y_test), _, _ = load_dataset('mnist')
        x_train, y_train, x_test, y_test = x_train[:NB_TRAIN], y_train[:NB_TRAIN], x_test[:NB_TEST], y_test[:NB_TEST]
        im_shape = x_train[0].shape

        # Create and fit target classifier
        comp_params = {'loss': 'categorical_crossentropy', 'optimizer': 'adam', 'metrics': ['accuracy']}
        params = {'epochs': 5, 'batch_size': BATCH_SIZE}
        classifier_tgt = CNN(im_shape, dataset='mnist')
        classifier_tgt.compile(comp_params)
        classifier_tgt.fit(x_train, y_train, **params)

        # Create source classifier
        classifier_src = CNN(im_shape, dataset='mnist')
        classifier_src.compile(comp_params)
        classifier_src.fit(x_train, y_train, **params)

        # Create FGSM attacker
        adv = FastGradientMethod(classifier_src, session)
        x_adv = adv.generate(x_test)
        acc = classifier_tgt.evaluate(x_adv, y_test)

        # Perform adversarial training
        adv_trainer = AdversarialTrainer(classifier_tgt, adv)
        adv_trainer.fit(x_train, y_train, **params)

        # Evaluate that accuracy on adversarial sample has improved
        acc_adv_trained = adv_trainer.classifier.evaluate(x_adv, y_test)
        self.assertTrue(acc_adv_trained >= acc)

    def test_multi_attack_mnist(self):
        """
        Test the adversarial trainer using two attackers: FGSM and DeepFool. The source and target models of the attack
        are two CNNs on MNIST trained for 5 epochs. FGSM and DeepFool both generate the attack images on the same
        source classifier. The test cast check if accuracy on adversarial samples increases
        after adversarially training the model.

        :return: None
        """
        session = tf.Session()
        k.set_session(session)

        # Load MNIST
        (x_train, y_train), (x_test, y_test), _, _ = load_dataset('mnist')
        x_train, y_train, x_test, y_test = x_train[:NB_TRAIN], y_train[:NB_TRAIN], x_test[:NB_TEST], y_test[:NB_TEST]
        im_shape = x_train[0].shape

        # Create and fit target classifier
        comp_params = {'loss': 'categorical_crossentropy', 'optimizer': 'adam', 'metrics': ['accuracy']}
        params = {'epochs': 5, 'batch_size': BATCH_SIZE}
        classifier_tgt = CNN(im_shape, dataset='mnist')
        classifier_tgt.compile(comp_params)
        classifier_tgt.fit(x_train, y_train, **params)

        # Create source classifier
        classifier_src = CNN(im_shape, dataset='mnist')
        classifier_src.compile(comp_params)
        classifier_tgt.fit(x_train, y_train, **params)

        # Create FGSM and DeepFool attackers
        adv1 = FastGradientMethod(classifier_src, session)
        adv2 = DeepFool(classifier_src, session)
        x_adv = np.vstack((adv1.generate(x_test), adv2.generate(x_test)))
        y_adv = np.vstack((y_test, y_test))
        print(y_adv.shape)
        acc = classifier_tgt.evaluate(x_adv, y_adv)

        # Perform adversarial training
        adv_trainer = AdversarialTrainer(classifier_tgt, [adv1, adv2])
        adv_trainer.fit(x_train, y_train, **params)

        # Evaluate that accuracy on adversarial sample has improved
        acc_adv_trained = adv_trainer.classifier.evaluate(x_adv, y_adv)
        self.assertTrue(acc_adv_trained >= acc)

    def test_shared_model_mnist(self):
        """
        Test the adversarial trainer using one FGSM attacker. The source and target models of the attack are the same
        CNN on MNIST trained for 5 epochs. The test cast check if accuracy on adversarial samples increases
        after adversarially training the model.

        :return: None
        """
        session = tf.Session()
        k.set_session(session)

        # Load MNIST
        (x_train, y_train), (x_test, y_test), _, _ = load_dataset('mnist')
        x_train, y_train, x_test, y_test = x_train[:NB_TRAIN], y_train[:NB_TRAIN], x_test[:NB_TEST], y_test[:NB_TEST]
        im_shape = x_train[0].shape

        # Create and fit classifier
        params = {'epochs': 5, 'batch_size': BATCH_SIZE}
        classifier = CNN(im_shape, dataset='mnist')
        classifier.compile({'loss': 'categorical_crossentropy', 'optimizer': 'adam', 'metrics': ['accuracy']})
        classifier.fit(x_train, y_train, **params)

        # Create FGSM attacker
        adv = FastGradientMethod(classifier, session)
        x_adv = adv.generate(x_test)
        acc = classifier.evaluate(x_adv, y_test)

        # Perform adversarial training
        adv_trainer = AdversarialTrainer(classifier, adv)
        adv_trainer.fit(x_train, y_train, **params)

        # Evaluate that accuracy on adversarial sample has improved
        acc_adv_trained = adv_trainer.classifier.evaluate(x_adv, y_test)
        self.assertTrue(acc_adv_trained >= acc)
