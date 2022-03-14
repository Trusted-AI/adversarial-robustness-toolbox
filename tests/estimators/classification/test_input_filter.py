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

import logging
import unittest

import numpy as np

from tests.utils import TestBase, master_seed, get_image_classifier_kr_tf

logger = logging.getLogger(__name__)

BATCH_SIZE = 10
NB_TRAIN = 500
NB_TEST = 100


class TestInputFilter(TestBase):
    @classmethod
    def setUpClass(cls):
        master_seed(seed=1234)
        super().setUpClass()

    def setUp(self):
        master_seed(1234)
        super().setUp()

    def test_fit(self):
        labels = np.argmax(self.y_test_mnist, axis=1)
        classifier = get_image_classifier_kr_tf()

        acc = np.sum(np.argmax(classifier.predict(self.x_test_mnist), axis=1) == labels) / NB_TEST
        logger.info("Accuracy: %.2f%%", (acc * 100))

        classifier.fit(self.x_train_mnist, self.y_train_mnist, batch_size=BATCH_SIZE, nb_epochs=2)
        acc2 = np.sum(np.argmax(classifier.predict(self.x_test_mnist), axis=1) == labels) / NB_TEST
        logger.info("Accuracy: %.2f%%", (acc2 * 100))

        self.assertEqual(acc, 0.32)
        self.assertEqual(acc2, 0.77)

        classifier.fit(self.x_train_mnist, y=self.y_train_mnist, batch_size=BATCH_SIZE, nb_epochs=2)
        classifier.fit(x=self.x_train_mnist, y=self.y_train_mnist, batch_size=BATCH_SIZE, nb_epochs=2)

    # def test_class_gradient(self):
    #     classifier = get_image_classifier_kr_tf()
    #
    #     # Test all gradients label
    #     gradients = classifier.class_gradient(self.x_test_mnist)
    #
    #     self.assertTrue(gradients.shape == (NB_TEST, 10, 28, 28, 1))
    #
    #     expected_gradients_1 = np.asarray(
    #         [
    #             -1.0557447e-03,
    #             -1.0079544e-03,
    #             -7.7426434e-04,
    #             1.7387432e-03,
    #             2.1773507e-03,
    #             5.0880699e-05,
    #             1.6497371e-03,
    #             2.6113100e-03,
    #             6.0904310e-03,
    #             4.1080985e-04,
    #             2.5268078e-03,
    #             -3.6661502e-04,
    #             -3.0568996e-03,
    #             -1.1665225e-03,
    #             3.8904310e-03,
    #             3.1726385e-04,
    #             1.3203260e-03,
    #             -1.1720930e-04,
    #             -1.4315104e-03,
    #             -4.7676818e-04,
    #             9.7251288e-04,
    #             0.0000000e00,
    #             0.0000000e00,
    #             0.0000000e00,
    #             0.0000000e00,
    #             0.0000000e00,
    #             0.0000000e00,
    #             0.0000000e00,
    #         ]
    #     )
    #     np.testing.assert_array_almost_equal(gradients[0, 5, 14, :, 0], expected_gradients_1, decimal=4)
    #
    #     expected_gradients_2 = np.asarray(
    #         [
    #             -0.00367321,
    #             -0.0002892,
    #             0.00037825,
    #             -0.00053344,
    #             0.00192121,
    #             0.00112047,
    #             0.0023135,
    #             0.0,
    #             0.0,
    #             -0.00391743,
    #             -0.0002264,
    #             0.00238103,
    #             -0.00073711,
    #             0.00270405,
    #             0.00389043,
    #             0.00440818,
    #             -0.00412769,
    #             -0.00441794,
    #             0.00081916,
    #             -0.00091284,
    #             0.00119645,
    #             -0.00849089,
    #             0.00547925,
    #             0.0,
    #             0.0,
    #             0.0,
    #             0.0,
    #             0.0,
    #         ]
    #     )
    #     np.testing.assert_array_almost_equal(gradients[0, 5, :, 14, 0], expected_gradients_2, decimal=4)
    #
    #     # Test 1 gradient label = 5
    #     gradients = classifier.class_gradient(self.x_test_mnist, label=5)
    #
    #     self.assertTrue(gradients.shape == (NB_TEST, 1, 28, 28, 1))
    #
    #     expected_gradients_1 = np.asarray(
    #         [
    #             -1.0557447e-03,
    #             -1.0079544e-03,
    #             -7.7426434e-04,
    #             1.7387432e-03,
    #             2.1773507e-03,
    #             5.0880699e-05,
    #             1.6497371e-03,
    #             2.6113100e-03,
    #             6.0904310e-03,
    #             4.1080985e-04,
    #             2.5268078e-03,
    #             -3.6661502e-04,
    #             -3.0568996e-03,
    #             -1.1665225e-03,
    #             3.8904310e-03,
    #             3.1726385e-04,
    #             1.3203260e-03,
    #             -1.1720930e-04,
    #             -1.4315104e-03,
    #             -4.7676818e-04,
    #             9.7251288e-04,
    #             0.0000000e00,
    #             0.0000000e00,
    #             0.0000000e00,
    #             0.0000000e00,
    #             0.0000000e00,
    #             0.0000000e00,
    #             0.0000000e00,
    #         ]
    #     )
    #     np.testing.assert_array_almost_equal(gradients[0, 0, 14, :, 0], expected_gradients_1, decimal=4)
    #
    #     expected_gradients_2 = np.asarray(
    #         [
    #             -0.00367321,
    #             -0.0002892,
    #             0.00037825,
    #             -0.00053344,
    #             0.00192121,
    #             0.00112047,
    #             0.0023135,
    #             0.0,
    #             0.0,
    #             -0.00391743,
    #             -0.0002264,
    #             0.00238103,
    #             -0.00073711,
    #             0.00270405,
    #             0.00389043,
    #             0.00440818,
    #             -0.00412769,
    #             -0.00441794,
    #             0.00081916,
    #             -0.00091284,
    #             0.00119645,
    #             -0.00849089,
    #             0.00547925,
    #             0.0,
    #             0.0,
    #             0.0,
    #             0.0,
    #             0.0,
    #         ]
    #     )
    #     np.testing.assert_array_almost_equal(gradients[0, 0, :, 14, 0], expected_gradients_2, decimal=4)
    #
    #     # Test a set of gradients label = array
    #     label = np.random.randint(5, size=NB_TEST)
    #     gradients = classifier.class_gradient(self.x_test_mnist, label=label)
    #
    #     self.assertTrue(gradients.shape == (NB_TEST, 1, 28, 28, 1))
    #
    #     expected_gradients_1 = np.asarray(
    #         [
    #             5.0867125e-03,
    #             4.8564528e-03,
    #             6.1040390e-03,
    #             8.6531248e-03,
    #             -6.0958797e-03,
    #             -1.4114540e-02,
    #             -7.1085989e-04,
    #             -5.0330814e-04,
    #             1.2943064e-02,
    #             8.2416134e-03,
    #             -1.9859476e-04,
    #             -9.8109958e-05,
    #             -3.8902222e-03,
    #             -1.2945873e-03,
    #             7.5137997e-03,
    #             1.7720886e-03,
    #             3.1399424e-04,
    #             2.3657181e-04,
    #             -3.0891625e-03,
    #             -1.0211229e-03,
    #             2.0828887e-03,
    #             0.0000000e00,
    #             0.0000000e00,
    #             0.0000000e00,
    #             0.0000000e00,
    #             0.0000000e00,
    #             0.0000000e00,
    #             0.0000000e00,
    #         ]
    #     )
    #     np.testing.assert_array_almost_equal(gradients[0, 0, 14, :, 0], expected_gradients_1, decimal=4)
    #
    #     expected_gradients_2 = np.asarray(
    #         [
    #             -0.00195835,
    #             -0.00134457,
    #             -0.00307221,
    #             -0.00340564,
    #             0.00175022,
    #             -0.00239714,
    #             -0.00122619,
    #             0.0,
    #             0.0,
    #             -0.00520899,
    #             -0.00046105,
    #             0.00414874,
    #             -0.00171095,
    #             0.00429184,
    #             0.0075138,
    #             0.00792442,
    #             0.0019566,
    #             0.00035517,
    #             0.00504575,
    #             -0.00037397,
    #             0.00022343,
    #             -0.00530034,
    #             0.0020528,
    #             0.0,
    #             0.0,
    #             0.0,
    #             0.0,
    #             0.0,
    #         ]
    #     )
    #     np.testing.assert_array_almost_equal(gradients[0, 0, :, 14, 0], expected_gradients_2, decimal=4)
    #
    # def test_loss_gradient(self):
    #     classifier = get_image_classifier_kr_tf()
    #
    #     # Test gradient
    #     gradients = classifier.loss_gradient(x=self.x_test_mnist, y=self.y_test_mnist)
    #
    #     self.assertTrue(gradients.shape == (NB_TEST, 28, 28, 1))
    #
    #     expected_gradients_1 = np.asarray(
    #         [
    #             0.0559206,
    #             0.05338925,
    #             0.0648919,
    #             0.07925165,
    #             -0.04029291,
    #             -0.11281465,
    #             0.01850601,
    #             0.00325054,
    #             0.08163195,
    #             0.03333949,
    #             0.031766,
    #             -0.02420463,
    #             -0.07815556,
    #             -0.04698735,
    #             0.10711591,
    #             0.04086434,
    #             -0.03441073,
    #             0.01071284,
    #             -0.04229195,
    #             -0.01386157,
    #             0.02827487,
    #             0.0,
    #             0.0,
    #             0.0,
    #             0.0,
    #             0.0,
    #             0.0,
    #             0.0,
    #         ]
    #     )
    #     np.testing.assert_array_almost_equal(gradients[0, 14, :, 0], expected_gradients_1, decimal=4)
    #
    #     expected_gradients_2 = np.asarray(
    #         [
    #             0.00210803,
    #             0.00213919,
    #             0.00520981,
    #             0.00548001,
    #             -0.0023059,
    #             0.00432077,
    #             0.00274945,
    #             0.0,
    #             0.0,
    #             -0.0583441,
    #             -0.00616604,
    #             0.0526219,
    #             -0.02373985,
    #             0.05273106,
    #             0.10711591,
    #             0.12773865,
    #             0.0689289,
    #             0.01337799,
    #             0.10032021,
    #             0.01681096,
    #             -0.00028647,
    #             -0.05588859,
    #             0.01474165,
    #             0.0,
    #             0.0,
    #             0.0,
    #             0.0,
    #             0.0,
    #         ]
    #     )
    #     np.testing.assert_array_almost_equal(gradients[0, :, 14, 0], expected_gradients_2, decimal=4)

    def test_layers(self):
        classifier = get_image_classifier_kr_tf()
        self.assertEqual(len(classifier.layer_names), 3)

        layer_names = classifier.layer_names
        for i, name in enumerate(layer_names):
            act_i = classifier.get_activations(self.x_test_mnist, i, batch_size=128)
            act_name = classifier.get_activations(self.x_test_mnist, name, batch_size=128)
            np.testing.assert_array_equal(act_name, act_i)


if __name__ == "__main__":
    unittest.main()
