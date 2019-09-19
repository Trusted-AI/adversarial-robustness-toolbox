from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

from art.utils import load_mnist, master_seed
from art.utils_test import get_classifier_tf_v2

import tensorflow as tf

logger = logging.getLogger('testLogger')

NB_TRAIN = 1000
NB_TEST = 20


@unittest.skipIf(tf.__version__[0] != '2', reason='Run unittests only for TensorFlow v2.')
class TensorFlowV2Classifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        x_train, y_train = x_train[:NB_TRAIN], y_train[:NB_TRAIN]
        x_test, y_test = x_test[:NB_TEST], y_test[:NB_TEST]
        cls.x_train = x_train
        cls.y_train = y_train
        cls.x_test = x_test
        cls.y_test = y_test

        cls.classifier = get_classifier_tf_v2()

    def setUp(self):
        master_seed(1234)

    def test_predict(self):
        y_predicted = self.classifier.predict(self.x_test[0:1])
        y_expected = [[0.12109935, 0.0498215, 0.0993958, 0.06410097, 0.11366927, 0.04645343, 0.06419806, 0.30685693,
                       0.07616713, 0.05823758]]

        for i in range(10):
            self.assertAlmostEqual(y_predicted[0, i], y_expected[0][i], places=4)

    def test_class_gradient_none_1(self):
        grad_predicted = self.classifier.class_gradient(self.x_test[0:1], label=None)
        grad_expected = [0.00159243, 0.00763082, -0.00487283, 0.0075138, 0.00529063, 0.00389043, -0.00108436,
                         -0.03286926, 0.00493717, 0.00797117]

        self.assertTrue(grad_predicted.shape == (1, 10, 28, 28, 1))

        for i_class in range(10):
            self.assertAlmostEqual(grad_predicted[0, i_class, 14, 14, 0], grad_expected[i_class], 4)

    def test_class_gradient_none_2(self):
        grad_predicted = self.classifier.class_gradient(self.x_test[0:2], label=None)
        grad_expected = [[0.00159243, 0.00763082, -0.00487283, 0.0075138, 0.00529063, 0.00389043,
                          -0.00108436, -0.03286926, 0.00493717, 0.00797117],
                         [-0.00439638, -0.0131422, -0.00130951, 0.00590583, 0.00821168, 0.00413466,
                          -0.00663735, -0.00358527, 0.00966956, 0.00114899]]

        self.assertTrue(grad_predicted.shape == (2, 10, 28, 28, 1))

        for i_sample in range(2):
            for i_class in range(10):
                self.assertAlmostEqual(grad_predicted[i_sample, i_class, 14, 14, 0],
                                       grad_expected[i_sample][i_class], 4)

    def test_class_gradient_int_1(self):
        grad_predicted = self.classifier.class_gradient(self.x_test[0:1], label=1)
        grad_expected = [[0.00230914, 0.00220461, 0.00333489, 0.00797444, 0.00470511, -0.0035654,
                          0.00829423, 0.00808538, 0.00259757, -0.00193153, -0.0028513, 0.00211392,
                          0.00561287, 0.00336864, 0.00763082, 0.00227913, 0.00453262, -0.0002169,
                          -0.00684221, -0.00227339, 0.00463727, 0., 0., 0.,
                          0., 0., 0., 0.]]

        self.assertTrue(grad_predicted.shape == (1, 1, 28, 28, 1))

        for i_shape in range(28):
            self.assertAlmostEqual(grad_predicted[0, 0, 14, i_shape, 0], grad_expected[0][i_shape], 4)

    def test_class_gradient_int_2(self):
        grad_predicted = self.classifier.class_gradient(self.x_test[0:2], label=1)
        grad_expected = [[0.00230914, 0.00220461, 0.00333489, 0.00797444, 0.00470511, -0.0035654,
                          0.00829423, 0.00808538, 0.00259757, -0.00193153, -0.0028513, 0.00211392,
                          0.00561287, 0.00336864, 0.00763082, 0.00227913, 0.00453262, -0.0002169,
                          -0.00684221, -0.00227339, 0.00463727, 0., 0., 0.,
                          0., 0., 0., 0.],
                         [0.00335857, 0.00320654, 0.00558169, 0.01684505, 0.00666651, -0.0109199,
                          0.01125671, 0.00819488, 0.01788009, 0.00936231, -0.01885438, -0.00487651,
                          -0.00045433, -0.0100058, -0.0131422, -0.0039416, 0.01837129, -0.00659209,
                          -0.00435764, -0.0005279, -0.01042055, 0.01376607, 0.00408139, 0.,
                          0., 0., 0., 0.]]

        self.assertTrue(grad_predicted.shape == (2, 1, 28, 28, 1))

        for i_sample in range(2):
            for i in range(28):
                self.assertAlmostEqual(grad_predicted[i_sample, 0, 14, i, 0], grad_expected[i_sample][i], 4)

    def test_class_gradient_list_1(self):
        grad_predicted = self.classifier.class_gradient(self.x_test[0:1], label=[1])
        grad_expected = [0.00230914, 0.00220461, 0.00333489, 0.00797444, 0.00470511, -0.0035654,
                         0.00829423, 0.00808538, 0.00259757, -0.00193153, -0.0028513, 0.00211392,
                         0.00561287, 0.00336864, 0.00763082, 0.00227913, 0.00453262, -0.0002169,
                         -0.00684221, -0.00227339, 0.00463727, 0., 0., 0.,
                         0., 0., 0., 0.]

        self.assertTrue(grad_predicted.shape == (1, 1, 28, 28, 1))

        for i in range(28):
            self.assertAlmostEqual(grad_predicted[0, 0, 14, i, 0], grad_expected[i], 4)

    def test_class_gradient_list_2(self):
        grad_predicted = self.classifier.class_gradient(self.x_test[0:2], label=[1, 2])
        grad_expected = [[2.30913871e-03, 2.20461124e-03, 3.33488820e-03, 7.97443957e-03,
                          4.70510989e-03, -3.56539942e-03, 8.29422800e-03, 8.08538301e-03,
                          2.59756986e-03, -1.93153096e-03, -2.85129989e-03, 2.11392409e-03,
                          5.61286953e-03, 3.36864446e-03, 7.63081519e-03, 2.27913485e-03,
                          4.53262183e-03, -2.16898321e-04, -6.84220525e-03, -2.27339185e-03,
                          4.63726978e-03, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                          0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                         [9.92421602e-03, 9.47497788e-03, 1.01928082e-02, 4.56807702e-03,
                          -6.60711716e-03, -9.42846849e-03, 3.31534747e-03, 1.29033007e-04,
                          -5.75811512e-03, -6.89728318e-03, 2.11321472e-03, -2.58737968e-03,
                          4.93051942e-03, -1.28652267e-03, -1.30951163e-03, -3.82406756e-04,
                          -3.86617555e-03, 1.26402790e-02, 9.20443097e-03, -6.64946500e-05,
                          -6.57990943e-04, -7.69793272e-03, -1.83563174e-03, 0.00000000e+00,
                          0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]]

        self.assertTrue(grad_predicted.shape == (2, 1, 28, 28, 1))

        for i_sample in range(2):
            for i in range(28):
                self.assertAlmostEqual(grad_predicted[i_sample, 0, 14, i, 0], grad_expected[i_sample][i], 4)

    def test_loss_gradient(self):
        grad_predicted = self.classifier.loss_gradient(self.x_test[0:1], self.y_test[0:1])
        grad_expected = [0.0559206, 0.05338925, 0.0648919, 0.07925164, -0.04029291, -0.11281465,
                         0.01850601, 0.00325054, 0.08163195, 0.03333949, 0.03176599, -0.02420464,
                         -0.07815557, -0.04698735, 0.10711591, 0.04086434, -0.03441073, 0.01071284,
                         -0.04229196, -0.01386157, 0.02827487, 0., 0., 0.,
                         0., 0., 0., 0.]

        self.assertTrue(grad_predicted.shape == (1, 28, 28, 1))

        for i in range(28):
            self.assertAlmostEqual(grad_predicted[0, 14, i, 0], grad_expected[i], 4)


if __name__ == '__main__':
    unittest.main()
