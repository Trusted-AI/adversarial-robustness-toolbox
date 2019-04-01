from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

import keras.backend as k
import numpy as np
import tensorflow as tf

from art.attacks import AdversarialPatch
from art.utils import load_mnist, master_seed, get_classifier_tf, get_classifier_kr, get_classifier_pt

logger = logging.getLogger('testLogger')

BATCH_SIZE = 10
NB_TRAIN = 10
NB_TEST = 10


class TestAdversarialPatch(unittest.TestCase):
    """
    A unittest class for testing Adversarial Patch attack.
    """

    @classmethod
    def setUpClass(cls):
        # Get MNIST
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        x_train, y_train = x_train[:NB_TRAIN], y_train[:NB_TRAIN]
        x_test, y_test = x_test[:NB_TEST], y_test[:NB_TEST]
        cls.mnist = (x_train, y_train), (x_test, y_test)

    def setUp(self):
        # Set master seed
        master_seed(1234)

    def test_tfclassifier(self):
        """
        First test with the TFClassifier.
        :return:
        """
        # Build TFClassifier
        tfc, sess = get_classifier_tf()

        # Get MNIST
        (x_train, _), (_, _) = self.mnist

        # Attack
        attack_params = {"target_ys": np.zeros((10, 10)), "rotation_max": 22.5, "scale_min": 0.1, "scale_max": 1.0,
                         "learning_rate": 5.0, "number_of_steps": 5, "patch_shape": (28, 28, 1), "batch_size": 10}
        attack_ap = AdversarialPatch(tfc)
        patch_adv, patch_mask_adv = attack_ap.generate(x_train, **attack_params)

        self.assertTrue(patch_adv[8, 8, 0] - (-3.1541491702440285) < 0.01)
        self.assertTrue(patch_adv[14, 14, 0] - 19.77060710322136 < 0.01)
        self.assertTrue(np.sum(patch_adv) - 387.92340613537993 < 0.01)

        sess.close()
        tf.reset_default_graph()

    def test_krclassifier(self):
        """
        Second test with the KerasClassifier.
        :return:
        """
        # Build KerasClassifier
        krc, sess = get_classifier_kr()

        # Get MNIST
        (x_train, _), (_, _) = self.mnist

        # Attack
        attack_params = {"target_ys": np.zeros((10, 10)), "rotation_max": 22.5, "scale_min": 0.1, "scale_max": 1.0,
                         "learning_rate": 5.0, "number_of_steps": 5, "patch_shape": (28, 28, 1), "batch_size": 10}
        attack_ap = AdversarialPatch(krc)
        patch_adv, patch_mask_adv = attack_ap.generate(x_train, **attack_params)

        self.assertTrue(patch_adv[8, 8, 0] - (-3.2501425017774923) < 0.01)
        self.assertTrue(patch_adv[14, 14, 0] - 19.62935354176458 < 0.01)
        self.assertTrue(np.sum(patch_adv) - 427.5144147080584 < 0.01)

        k.clear_session()

    def test_ptclassifier(self):
        """
        Third test with the PyTorchClassifier.
        :return:
        """
        # Build PyTorchClassifier
        ptc = get_classifier_pt()

        # Get MNIST
        (x_train, _), (_, _) = self.mnist
        x_train = np.swapaxes(x_train, 1, 3)

        # Attack
        attack_params = {"target_ys": np.zeros((10, 10)), "rotation_max": 22.5, "scale_min": 0.1, "scale_max": 1.0,
                         "learning_rate": 5.0, "number_of_steps": 5, "patch_shape": (1, 28, 28), "batch_size": 10}
        attack_ap = AdversarialPatch(ptc)
        patch_adv, patch_mask_adv = attack_ap.generate(x_train, **attack_params)

        self.assertTrue(patch_adv[0, 8, 8] - (-3.1423605902784875) < 0.01)
        self.assertTrue(patch_adv[0, 14, 14] - 19.790434152473054 < 0.01)
        self.assertTrue(np.sum(patch_adv) - 383.5670772794207 < 0.01)

    if __name__ == '__main__':
        unittest.main()
