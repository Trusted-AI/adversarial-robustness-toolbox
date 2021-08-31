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
import unittest

import numpy as np

from art.attacks.poisoning.feature_collision_attack import FeatureCollisionAttack

from tests.utils import TestBase, master_seed, get_image_classifier_kr  # , get_image_classifier_tf

logger = logging.getLogger(__name__)

NB_EPOCHS = 3


class TestFeatureCollision(TestBase):
    """
    A unittest class for testing Feature Collision attack.
    """

    @classmethod
    def setUpClass(cls):
        master_seed(seed=1234)
        super().setUpClass()

        cls.n_train = 10
        cls.n_test = 10
        cls.x_train_mnist = cls.x_train_mnist[0 : cls.n_train]
        cls.y_train_mnist = cls.y_train_mnist[0 : cls.n_train]
        cls.x_test_mnist = cls.x_test_mnist[0 : cls.n_test]
        cls.y_test_mnist = cls.y_test_mnist[0 : cls.n_test]

    def setUp(self):
        master_seed(seed=301)
        super().setUp()

    @staticmethod
    def poison_dataset(classifier, x_clean, y_clean):
        x_poison = np.copy(x_clean)
        y_poison = np.copy(y_clean)
        base = np.expand_dims(x_clean[0], axis=0)
        target = np.expand_dims(x_clean[1], axis=0)
        feature_layer = classifier.layer_names[-1]
        attack = FeatureCollisionAttack(classifier, target, feature_layer, max_iter=1)
        attack, attack_label = attack.poison(base)
        x_poison = np.append(x_poison, attack, axis=0)
        y_poison = np.append(y_poison, attack_label, axis=0)

        return x_poison, y_poison

    # def test_tensorflow(self):
    #     """
    #     First test with the TensorFlowClassifier.
    #     :return:
    #     """
    #     tfc, sess = get_image_classifier_tf()
    #     x_adv, y_adv = self.poison_dataset(tfc, self.x_train_mnist, self.y_train_mnist)
    #     tfc.fit(x_adv, y_adv, nb_epochs=NB_EPOCHS, batch_size=32)
    #
    #     if sess is not None:
    #         sess.close()

    def test_keras(self):
        """
        Test working keras implementation.
        :return:
        """
        krc = get_image_classifier_kr()
        x_adv, y_adv = self.poison_dataset(krc, self.x_train_mnist, self.y_train_mnist)
        krc.fit(x_adv, y_adv, nb_epochs=NB_EPOCHS, batch_size=32)

    def test_check_params(self):

        krc = get_image_classifier_kr(from_logits=True)

        with self.assertRaises(ValueError):
            _ = FeatureCollisionAttack(krc, target=self.x_train_mnist, feature_layer=1, learning_rate=-1)

        with self.assertRaises(TypeError):
            _ = FeatureCollisionAttack(krc, target=self.x_train_mnist, feature_layer=1.0)

        with self.assertRaises(ValueError):
            _ = FeatureCollisionAttack(krc, target=self.x_train_mnist, feature_layer=1, decay_coeff=-1)

        with self.assertRaises(ValueError):
            _ = FeatureCollisionAttack(krc, target=self.x_train_mnist, feature_layer=1, stopping_tol=-1)

        with self.assertRaises(ValueError):
            _ = FeatureCollisionAttack(krc, target=self.x_train_mnist, feature_layer=1, obj_threshold=-1)

        with self.assertRaises(ValueError):
            _ = FeatureCollisionAttack(krc, target=self.x_train_mnist, feature_layer=1, max_iter=-1)

        with self.assertRaises(ValueError):
            _ = FeatureCollisionAttack(krc, target=self.x_train_mnist, feature_layer=1, watermark=1)

        with self.assertRaises(ValueError):
            _ = FeatureCollisionAttack(krc, target=self.x_train_mnist, feature_layer=1, verbose="true")


if __name__ == "__main__":
    unittest.main()
