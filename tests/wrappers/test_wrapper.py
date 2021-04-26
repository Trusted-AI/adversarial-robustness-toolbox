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

import keras.backend as k
import numpy as np

from art.wrappers.wrapper import ClassifierWrapper
from art.utils import load_mnist

from tests.utils import master_seed, get_image_classifier_kr

logger = logging.getLogger(__name__)

BATCH_SIZE = 10
NB_TRAIN = 500
NB_TEST = 100


class TestMixinWKerasClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        k.clear_session()

        # Get MNIST
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        x_train, y_train, x_test, y_test = x_train[:NB_TRAIN], y_train[:NB_TRAIN], x_test[:NB_TEST], y_test[:NB_TEST]
        cls.mnist = (x_train, y_train), (x_test, y_test)

        # Load small Keras model
        cls.model_mnist = get_image_classifier_kr()

    @classmethod
    def tearDownClass(cls):
        k.clear_session()

    def setUp(self):
        master_seed(seed=1234)

    def test_shapes(self):
        x_test, y_test = self.mnist[1]
        classifier = ClassifierWrapper(self.model_mnist)

        preds = classifier.predict(self.mnist[1][0])
        self.assertEqual(preds.shape, y_test.shape)

        self.assertEqual(classifier.nb_classes, 10)

        class_grads = classifier.class_gradient(x_test[:11])
        self.assertEqual(class_grads.shape, tuple([11, 10] + list(x_test[1].shape)))

        loss_grads = classifier.loss_gradient(x_test[:11], y_test[:11])
        self.assertEqual(loss_grads.shape, x_test[:11].shape)

    def test_class_gradient(self):
        (_, _), (x_test, _) = self.mnist
        classifier = ClassifierWrapper(self.model_mnist)

        # Test all gradients label
        grads = classifier.class_gradient(x_test)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 10, 28, 28, 1)).all())
        self.assertNotEqual(np.sum(grads), 0)

        # Test 1 gradient label = 5
        grads = classifier.class_gradient(x_test, label=5)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 1, 28, 28, 1)).all())
        self.assertNotEqual(np.sum(grads), 0)

        # Test a set of gradients label = array
        label = np.random.randint(5, size=NB_TEST)
        grads = classifier.class_gradient(x_test, label=label)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 1, 28, 28, 1)).all())
        self.assertNotEqual(np.sum(grads), 0)

    def test_loss_gradient(self):
        (_, _), (x_test, y_test) = self.mnist
        classifier = ClassifierWrapper(self.model_mnist)

        # Test gradient
        grads = classifier.loss_gradient(x_test, y_test)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 28, 28, 1)).all())
        self.assertNotEqual(np.sum(grads), 0)

    def test_layers(self):
        (_, _), (x_test, _), _, _ = load_mnist()
        x_test = x_test[:NB_TEST]

        classifier = ClassifierWrapper(self.model_mnist)
        self.assertEqual(len(classifier.layer_names), 3)

        layer_names = classifier.layer_names
        for i, name in enumerate(layer_names):
            act_i = classifier.get_activations(x_test, i, batch_size=128)
            act_name = classifier.get_activations(x_test, name, batch_size=128)
            self.assertAlmostEqual(np.sum(act_name - act_i), 0)

    def test_save(self):
        import os

        path = "tmp"
        filename = "model.h5"
        classifier = ClassifierWrapper(self.model_mnist)
        classifier.save(filename, path=path)
        self.assertTrue(os.path.isfile(os.path.join(path, filename)))

        # Remove saved file
        os.remove(os.path.join(path, filename))


if __name__ == "__main__":
    unittest.main()
