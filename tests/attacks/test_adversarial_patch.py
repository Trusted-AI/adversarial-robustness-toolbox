# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
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

from art.attacks.evasion.adversarial_patch.adversarial_patch import AdversarialPatch
from art.estimators.estimator import BaseEstimator, NeuralNetworkMixin
from art.estimators.classification.classifier import ClassifierMixin

from tests.utils import TestBase, master_seed
from tests.utils import get_image_classifier_tf, get_image_classifier_kr
from tests.utils import get_tabular_classifier_kr, get_image_classifier_pt
from tests.attacks.utils import backend_test_classifier_type_check_fail

logger = logging.getLogger(__name__)


class TestAdversarialPatch(TestBase):
    """
    A unittest class for testing Adversarial Patch attack.
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
        master_seed(seed=1234)
        super().setUp()

    def test_tensorflow(self):
        """
        First test with the TensorFlowClassifier.
        :return:
        """
        import tensorflow as tf

        tfc, sess = get_image_classifier_tf()

        attack_ap = AdversarialPatch(
            tfc,
            rotation_max=0.5,
            scale_min=0.4,
            scale_max=0.41,
            learning_rate=5.0,
            batch_size=10,
            max_iter=5,
            patch_shape=(28, 28, 1),
        )
        target = np.zeros(self.x_train_mnist.shape[0])
        patch_adv, _ = attack_ap.generate(self.x_train_mnist, target, shuffle=False)

        if tf.__version__[0] == "2":
            self.assertAlmostEqual(patch_adv[8, 8, 0], 0.55935985, delta=0.05)
            self.assertAlmostEqual(patch_adv[14, 14, 0], 0.5917497, delta=0.05)
            self.assertAlmostEqual(float(np.sum(patch_adv)), 400.0701904296875, delta=1.0)
        else:
            self.assertAlmostEqual(patch_adv[8, 8, 0], 0.7993435, delta=0.05)
            self.assertAlmostEqual(patch_adv[14, 14, 0], 1, delta=0.05)
            self.assertAlmostEqual(float(np.sum(patch_adv)), 392.339233398, delta=1.0)

        if sess is not None:
            sess.close()

    def test_keras(self):
        """
        Second test with the KerasClassifier.
        :return:
        """
        krc = get_image_classifier_kr()

        attack_ap = AdversarialPatch(
            krc, rotation_max=0.5, scale_min=0.4, scale_max=0.41, learning_rate=5.0, batch_size=10, max_iter=5
        )
        master_seed(seed=1234)
        target = np.zeros(self.x_train_mnist.shape[0])
        patch_adv, _ = attack_ap.generate(self.x_train_mnist, target)

        self.assertAlmostEqual(patch_adv[8, 8, 0], 0.7993435, delta=0.05)
        self.assertAlmostEqual(patch_adv[14, 14, 0], 1.0, delta=0.05)
        self.assertAlmostEqual(float(np.sum(patch_adv)), 392.3392333984375, delta=1.0)

    def test_pytorch(self):
        """
        Third test with the PyTorchClassifier.
        :return:
        """
        ptc = get_image_classifier_pt()

        x_train = np.reshape(self.x_train_mnist, (self.n_train, 1, 28, 28)).astype(np.float32)

        attack_ap = AdversarialPatch(
            ptc, rotation_max=0.5, scale_min=0.4, scale_max=0.41, learning_rate=5.0, batch_size=10, max_iter=5
        )
        master_seed(seed=1234)
        target = np.zeros(self.x_train_mnist.shape[0])
        patch_adv, _ = attack_ap.generate(x_train, target)

        self.assertAlmostEqual(patch_adv[0, 8, 8], 0.5581951, delta=0.05)
        self.assertAlmostEqual(patch_adv[0, 14, 14], 0.5420919, delta=0.05)
        self.assertAlmostEqual(float(np.sum(patch_adv)), 401.57159423828125, delta=1.0)

    def test_failure_feature_vectors(self):
        attack_params = {
            "rotation_max": 22.5,
            "scale_min": 0.1,
            "scale_max": 1.0,
            "learning_rate": 5.0,
            "number_of_steps": 5,
            "batch_size": 10,
        }
        classifier = get_tabular_classifier_kr()
        classifier._clip_values = (0, 1)
        attack = AdversarialPatch(classifier=classifier)
        attack.set_params(**attack_params)
        data = np.random.rand(10, 4)
        labels = np.random.randint(0, 3, 10)

        # Assert that value error is raised for feature vectors
        with self.assertRaises(ValueError) as context:
            attack.generate(data, labels)

        self.assertIn("Feature vectors detected.", str(context.exception))

    def test_classifier_type_check_fail(self):
        backend_test_classifier_type_check_fail(AdversarialPatch, [BaseEstimator, NeuralNetworkMixin, ClassifierMixin])


if __name__ == "__main__":
    unittest.main()
