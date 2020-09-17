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
import os
import unittest

import numpy as np

from art.attacks.poisoning.backdoor_attack import PoisoningAttackBackdoor
from art.attacks.poisoning.adversarial_embedding_attack import PoisoningAttackAdversarialEmbedding
from art.attacks.poisoning.perturbations import add_pattern_bd
from art.utils import load_dataset

from tests.utils import master_seed, get_image_classifier_kr_tf

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
logger = logging.getLogger(__name__)

BATCH_SIZE = 100
NB_TRAIN = 5000
NB_TEST = 10
NB_EPOCHS = 1


class TestAdversarialEmbedding(unittest.TestCase):
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
        master_seed(seed=301)

    def test_keras(self):
        """
        Test with a KerasClassifier.
        :return:
        """
        # Build KerasClassifier
        krc = get_image_classifier_kr_tf(loss_type="label")

        # Get MNIST
        (x_train, y_train), (_, _) = self.mnist
        target_idx = 9
        target = np.zeros(10)
        target[target_idx] = 1
        target2 = np.zeros(10)
        target2[(target_idx + 1) % 10] = 1

        backdoor = PoisoningAttackBackdoor(add_pattern_bd)

        emb_attack = PoisoningAttackAdversarialEmbedding(krc, backdoor, 2, target)
        classifier = emb_attack.poison_estimator(x_train, y_train, nb_epochs=NB_EPOCHS)

        data, labels, bd = emb_attack.get_training_data()
        self.assertEqual(x_train.shape, data.shape)
        self.assertEqual(y_train.shape, labels.shape)
        self.assertEqual(bd.shape, (len(x_train), 2))

        # Assert successful cloning of classifier model
        self.assertTrue(classifier is not krc)

        emb_attack2 = PoisoningAttackAdversarialEmbedding(krc, backdoor, 2, [(target, target2)])
        _ = emb_attack2.poison_estimator(x_train, y_train, nb_epochs=NB_EPOCHS)

        data, labels, bd = emb_attack2.get_training_data()
        self.assertEqual(x_train.shape, data.shape)
        self.assertEqual(y_train.shape, labels.shape)
        self.assertEqual(bd.shape, (len(x_train), 2))

        _ = PoisoningAttackAdversarialEmbedding(krc, backdoor, 2, [(target, target2)], pp_poison=[0.4])

    def test_errors(self):
        krc = get_image_classifier_kr_tf(loss_type="function")
        krc_valid = get_image_classifier_kr_tf(loss_type="label")
        backdoor = PoisoningAttackBackdoor(add_pattern_bd)
        target_idx = 9
        target = np.zeros(10)
        target[target_idx] = 1
        target2 = np.zeros(10)
        target2[(target_idx + 1) % 10] = 1

        # invalid loss function
        with self.assertRaises(TypeError):
            _ = PoisoningAttackAdversarialEmbedding(krc, backdoor, 2, target)

        # feature layer not real name
        with self.assertRaises(ValueError):
            _ = PoisoningAttackAdversarialEmbedding(krc_valid, backdoor, "not a layer", target)

        # feature layer out of range
        with self.assertRaises(ValueError):
            _ = PoisoningAttackAdversarialEmbedding(krc_valid, backdoor, 20, target)

        # target misshaped
        with self.assertRaises(ValueError):
            _ = PoisoningAttackAdversarialEmbedding(krc_valid, backdoor, 20, np.expand_dims(target, axis=0))

        with self.assertRaises(ValueError):
            _ = PoisoningAttackAdversarialEmbedding(krc_valid, backdoor, 20, [target])

        with self.assertRaises(ValueError):
            _ = PoisoningAttackAdversarialEmbedding(krc_valid, backdoor, 20, target, regularization=-1)

        with self.assertRaises(ValueError):
            _ = PoisoningAttackAdversarialEmbedding(krc_valid, backdoor, 20, target, discriminator_layer_1=-1)

        with self.assertRaises(ValueError):
            _ = PoisoningAttackAdversarialEmbedding(krc_valid, backdoor, 20, target, discriminator_layer_2=-1)

        with self.assertRaises(ValueError):
            _ = PoisoningAttackAdversarialEmbedding(krc_valid, backdoor, 20, target, pp_poison=-1)

        with self.assertRaises(ValueError):
            _ = PoisoningAttackAdversarialEmbedding(krc_valid, backdoor, 20, [(target, target2)], pp_poison=[])

        with self.assertRaises(ValueError):
            _ = PoisoningAttackAdversarialEmbedding(krc_valid, backdoor, 20, [(target, target2)], pp_poison=[-1])


if __name__ == "__main__":
    unittest.main()
