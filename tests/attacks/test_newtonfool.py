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

import logging
import unittest

import numpy as np

from art.attacks.newtonfool import NewtonFool
from art.utils import load_mnist, master_seed, get_classifier_tf, get_classifier_kr, get_classifier_pt

logger = logging.getLogger('testLogger')

BATCH_SIZE = 100
NB_TRAIN = 1000
NB_TEST = 100


class TestNewtonFool(unittest.TestCase):
    """
    A unittest class for testing the NewtonFool attack.
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
        (_, _), (x_test, _) = self.mnist

        # Attack
        # import time
        nf = NewtonFool(tfc, max_iter=5)

        # print("Test Tensorflow....")
        # starttime = time.clock()
        # x_test_adv = nf.generate(x_test, batch_size=1)
        # self.assertFalse((x_test == x_test_adv).all())
        # endtime = time.clock()
        # print(1, endtime - starttime)

        # starttime = time.clock()
        # x_test_adv = nf.generate(x_test, batch_size=10)
        # endtime = time.clock()
        # print(10, endtime - starttime)

        # starttime = time.clock()
        x_test_adv = nf.generate(x_test, batch_size=100)
        # endtime = time.clock()
        # print(100, endtime - starttime)
        #
        # starttime = time.clock()
        # x_test_adv = nf.generate(x_test, batch_size=1000)
        # endtime = time.clock()
        # print(1000, endtime - starttime)

        self.assertFalse((x_test == x_test_adv).all())

        y_pred = tfc.predict(x_test)
        y_pred_adv = tfc.predict(x_test_adv)
        y_pred_bool = y_pred.max(axis=1, keepdims=1) == y_pred
        y_pred_max = y_pred.max(axis=1)
        y_pred_adv_max = y_pred_adv[y_pred_bool]
        self.assertTrue((y_pred_max >= y_pred_adv_max).all())

    def test_krclassifier(self):
        """
        Second test with the KerasClassifier.
        :return:
        """
        # Build KerasClassifier
        krc, sess = get_classifier_kr()

        # Get MNIST
        (_, _), (x_test, _) = self.mnist

        # Attack
        # import time
        nf = NewtonFool(krc, max_iter=5)

        # print("Test Keras....")
        # starttime = time.clock()
        # x_test_adv = nf.generate(x_test, batch_size=1)
        # endtime = time.clock()
        # print(1, endtime - starttime)

        # starttime = time.clock()
        # x_test_adv = nf.generate(x_test, batch_size=10)
        # endtime = time.clock()
        # print(10, endtime - starttime)

        # starttime = time.clock()
        x_test_adv = nf.generate(x_test, batch_size=100)
        # endtime = time.clock()
        # print(100, endtime - starttime)
        #
        # starttime = time.clock()
        # x_test_adv = nf.generate(x_test, batch_size=1000)
        # endtime = time.clock()
        # print(1000, endtime - starttime)

        self.assertFalse((x_test == x_test_adv).all())

        y_pred = krc.predict(x_test)
        y_pred_adv = krc.predict(x_test_adv)
        y_pred_bool = y_pred.max(axis=1, keepdims=1) == y_pred
        y_pred_max = y_pred.max(axis=1)
        y_pred_adv_max = y_pred_adv[y_pred_bool]
        self.assertTrue((y_pred_max >= y_pred_adv_max).all())

    def test_ptclassifier(self):
        """
        Third test with the PyTorchClassifier.
        :return:
        """
        # Build PyTorchClassifier
        ptc = get_classifier_pt()

        # Get MNIST
        (_, _), (x_test, _) = self.mnist
        x_test = np.swapaxes(x_test, 1, 3)

        # Attack
        # import time
        nf = NewtonFool(ptc, max_iter=5)

        # print("Test Pytorch....")
        # starttime = time.clock()
        # x_test_adv = nf.generate(x_test, batch_size=1)
        # endtime = time.clock()
        # print(1, endtime - starttime)

        # starttime = time.clock()
        # x_test_adv = nf.generate(x_test, batch_size=10)
        # endtime = time.clock()
        # print(10, endtime - starttime)

        # starttime = time.clock()
        x_test_adv = nf.generate(x_test, batch_size=100)
        # endtime = time.clock()
        # print(100, endtime - starttime)
        #
        # starttime = time.clock()
        # x_test_adv = nf.generate(x_test, batch_size=1000)
        # endtime = time.clock()
        # print(1000, endtime - starttime)

        self.assertFalse((x_test == x_test_adv).all())

        y_pred = ptc.predict(x_test)
        y_pred_adv = ptc.predict(x_test_adv)
        y_pred_bool = y_pred.max(axis=1, keepdims=1) == y_pred
        y_pred_max = y_pred.max(axis=1)
        y_pred_adv_max = y_pred_adv[y_pred_bool]
        self.assertTrue((y_pred_max >= y_pred_adv_max).all())


if __name__ == '__main__':
    unittest.main()
