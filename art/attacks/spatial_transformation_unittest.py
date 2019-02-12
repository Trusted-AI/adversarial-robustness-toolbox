from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

import keras.backend as k
import numpy as np
import tensorflow as tf

from art.attacks.spatial_transformation import SpatialTransformation
from art.classifiers import KerasClassifier, PyTorchClassifier, TFClassifier
from art.utils import load_mnist, master_seed, get_model_tf, get_model_kr, get_model_pt

logger = logging.getLogger('testLogger')
logger.setLevel(10)

BATCH_SIZE = 100
NB_TRAIN = 1000
NB_TEST = 10


class TestSpatialTransformation(unittest.TestCase):
    """
    A unittest class for testing Spatial attack.
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
        # Build TF model
        loss, logits, input_ph, output_ph = get_model_tf()

        # Tensorflow session and initialization
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # Get MNIST
        (x_train, _), (x_test, _) = self.mnist

        # Train the classifier
        tfc = TFClassifier(clip_values=(0, 1), input_ph=input_ph, logits=logits, output_ph=output_ph, train=None,
                           loss=loss, learning=None, sess=sess)

        # Attack
        attack_params = {"max_translation": 10.0, "num_translations": 3, "max_rotation": 30.0, "num_rotations": 3}
        attack_st = SpatialTransformation(tfc)
        x_train_adv = attack_st.generate(x_train, **attack_params)

        self.assertTrue(abs(x_train_adv[0, 8, 13, 0] - 0.49004024) <= 0.01)

        self.assertTrue(abs(attack_st.fooling_rate - 0.707) <= 0.01)

        self.assertTrue(attack_st.attack_trans_x == 3)
        self.assertTrue(attack_st.attack_trans_y == 3)
        self.assertTrue(attack_st.attack_rot == 30.0)

        x_test_adv = attack_st.generate(x_test)

        self.assertTrue(abs(x_test_adv[0, 14, 14, 0] - 0.013572651) <= 0.01)

        sess.close()
        tf.reset_default_graph()

    def test_krclassifier(self):
        """
        Second test with the KerasClassifier.
        :return:
        """
        # Initialize a tf session
        session = tf.Session()
        k.set_session(session)

        # Get MNIST
        (x_train, _), (x_test, _) = self.mnist

        model = get_model_kr()

        # Get classifier
        krc = KerasClassifier((0, 1), model, use_logits=False)

        # Attack
        attack_params = {"max_translation": 10.0, "num_translations": 3, "max_rotation": 30.0, "num_rotations": 3}
        attack_st = SpatialTransformation(krc)
        x_train_adv = attack_st.generate(x_train, **attack_params)

        self.assertTrue(abs(x_train_adv[0, 8, 13, 0] - 0.49004024) <= 0.01)
        self.assertTrue(abs(attack_st.fooling_rate - 0.707) <= 0.01)

        self.assertTrue(attack_st.attack_trans_x == 3)
        self.assertTrue(attack_st.attack_trans_y == 3)
        self.assertTrue(attack_st.attack_rot == 30.0)

        x_test_adv = attack_st.generate(x_test)

        self.assertTrue(abs(x_test_adv[0, 14, 14, 0] - 0.013572651) <= 0.01)

        k.clear_session()

    def test_ptclassifier(self):
        """
        Third test with the PyTorchClassifier.
        :return:
        """
        # Get MNIST
        (x_train, _), (x_test, _) = self.mnist
        x_train = np.swapaxes(x_train, 1, 3)
        x_test = np.swapaxes(x_test, 1, 3)

        model, loss_fn, optimizer = get_model_pt()

        # Get classifier
        ptc = PyTorchClassifier((0, 1), model, loss_fn, optimizer, (1, 28, 28), 10)

        # Attack
        attack_params = {"max_translation": 10.0, "num_translations": 3, "max_rotation": 30.0, "num_rotations": 3}
        attack_st = SpatialTransformation(ptc)
        x_train_adv = attack_st.generate(x_train, **attack_params)

        self.assertTrue(abs(x_train_adv[0, 0, 13, 5] - 0.374206543) <= 0.01)
        self.assertTrue(abs(attack_st.fooling_rate - 0.361) <= 0.01)

        self.assertTrue(attack_st.attack_trans_x == 0)
        self.assertTrue(attack_st.attack_trans_y == -3)
        self.assertTrue(attack_st.attack_rot == 30.0)

        x_test_adv = attack_st.generate(x_test)

        self.assertTrue(abs(x_test_adv[0, 0, 14, 14] - 0.008591662) <= 0.01)


if __name__ == '__main__':
    unittest.main()
