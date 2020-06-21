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
import os
import tempfile
import unittest

import numpy as np
from mxnet import gluon, init
from mxnet.gluon import nn

from art.estimators.classification.mxnet import MXClassifier
from art.utils import Deprecated
from tests.utils import TestBase, master_seed

logger = logging.getLogger(__name__)


class TestMXClassifier(TestBase):
    """
    This class tests the MXNet classifier.
    """

    @classmethod
    def setUpClass(cls):
        master_seed(seed=1234, set_mxnet=True)
        super().setUpClass()

        cls.x_train_mnist = np.swapaxes(cls.x_train_mnist, 1, 3)
        cls.x_test_mnist = np.swapaxes(cls.x_test_mnist, 1, 3)

        # Create a simple CNN - this one comes from the Gluon tutorial
        net = nn.Sequential()
        with net.name_scope():
            net.add(
                nn.Conv2D(channels=6, kernel_size=5, activation="relu"),
                nn.MaxPool2D(pool_size=2, strides=2),
                nn.Conv2D(channels=16, kernel_size=3, activation="relu"),
                nn.MaxPool2D(pool_size=2, strides=2),
                nn.Flatten(),
                nn.Dense(120, activation="relu"),
                nn.Dense(84, activation="relu"),
                nn.Dense(10),
            )
        net.initialize(init=init.Xavier())

        # Create optimizer
        loss = gluon.loss.SoftmaxCrossEntropyLoss()
        trainer = gluon.Trainer(net.collect_params(), "sgd", {"learning_rate": 0.1})

        # Fit classifier
        classifier = MXClassifier(
            model=net, loss=loss, clip_values=(0, 1), input_shape=(1, 28, 28), nb_classes=10, optimizer=trainer
        )
        classifier.fit(cls.x_train_mnist, cls.y_train_mnist, batch_size=128, nb_epochs=2)
        cls.classifier = classifier

        cls.x_train_mnist = np.swapaxes(cls.x_train_mnist, 1, 3)
        cls.x_test_mnist = np.swapaxes(cls.x_test_mnist, 1, 3)

    def setUp(self):
        self.x_train_mnist = np.swapaxes(self.x_train_mnist, 1, 3)
        self.x_test_mnist = np.swapaxes(self.x_test_mnist, 1, 3)
        super().setUp()

    def tearDown(self):
        self.x_train_mnist = np.swapaxes(self.x_train_mnist, 1, 3)
        self.x_test_mnist = np.swapaxes(self.x_test_mnist, 1, 3)
        super().tearDown()

    def test_predict(self):
        preds = self.classifier.predict(self.x_test_mnist)
        acc = np.sum(np.argmax(preds, axis=1) == np.argmax(self.y_test_mnist, axis=1)) / self.n_test
        logger.info("Accuracy after fitting: %.2f%%", (acc * 100))
        self.assertGreater(acc, 0.1)

    def test_fit_generator(self):
        from art.data_generators import MXDataGenerator

        acc = (
            np.sum(
                np.argmax(self.classifier.predict(self.x_test_mnist), axis=1) == np.argmax(self.y_test_mnist, axis=1)
            )
            / self.n_test
        )
        logger.info("Accuracy: %.2f%%", (acc * 100))

        # Create MXNet dataset and loader
        dataset = gluon.data.dataset.ArrayDataset(self.x_train_mnist, self.y_train_mnist)
        data_loader = gluon.data.DataLoader(dataset, batch_size=5, shuffle=True)
        data_gen = MXDataGenerator(data_loader, size=self.n_train, batch_size=5)

        # Fit model with generator
        self.classifier.fit_generator(data_gen, nb_epochs=2)
        acc2 = (
            np.sum(
                np.argmax(self.classifier.predict(self.x_test_mnist), axis=1) == np.argmax(self.y_test_mnist, axis=1)
            )
            / self.n_test
        )
        logger.info("Accuracy: %.2f%%", (acc * 100))

        self.assertGreaterEqual(acc2, 0.8 * acc)

    def test_nb_classes(self):
        self.assertEqual(self.classifier.nb_classes, 10)

    def test_input_shape(self):
        self.assertEqual(self.classifier.input_shape, (1, 28, 28))

    def test_class_gradient(self):
        # Test class grads for all classes
        grads_all = self.classifier.class_gradient(self.x_test_mnist)
        self.assertTrue(np.array(grads_all.shape == (self.n_test, 10, 1, 28, 28)).all())
        self.assertNotEqual(np.sum(grads_all), 0)

        # Test class grads for specified label
        grads = self.classifier.class_gradient(self.x_test_mnist, label=3)
        self.assertTrue(np.array(grads.shape == (self.n_test, 1, 1, 28, 28)).all())
        self.assertNotEqual(np.sum(grads), 0)

        # Assert gradient computed for the same class on same input are equal
        self.assertAlmostEqual(float(np.sum(grads_all[:, 3] - grads)), 0, places=4)

        # Test a set of gradients label = array
        labels = np.random.randint(5, size=self.n_test)
        grads = self.classifier.class_gradient(self.x_test_mnist, label=labels)

        self.assertTrue(np.array(grads.shape == (self.n_test, 1, 1, 28, 28)).all())
        self.assertNotEqual(np.sum(grads), 0)

    def test_loss_gradient(self):
        # Compute loss gradients
        grads = self.classifier.loss_gradient(self.x_test_mnist, self.y_test_mnist)

        self.assertTrue(np.array(grads.shape == (self.n_test, 1, 28, 28)).all())
        self.assertNotEqual(np.sum(grads), 0)

    def test_preprocessing(self):
        # Create classifier
        loss = gluon.loss.SoftmaxCrossEntropyLoss()
        classifier_preproc = MXClassifier(
            model=self.classifier._model,
            loss=loss,
            clip_values=(0, 1),
            input_shape=(1, 28, 28),
            nb_classes=10,
            optimizer=self.classifier._optimizer,
            preprocessing=(1, 2),
        )

        preds = self.classifier.predict((self.x_test_mnist - 1.0) / 2)
        preds_preproc = classifier_preproc.predict(self.x_test_mnist)
        self.assertEqual(np.sum(preds - preds_preproc), 0)

    def test_layers(self):
        self.assertEqual(len(self.classifier.layer_names), 7)

        # layer_names = self.classifier.layer_names
        # for i, name in enumerate(layer_names):
        #     act_i = self.classifier.get_activations(x_test, i)
        #     act_name = self.classifier.get_activations(x_test, name)
        #     self.assertAlmostEqual(np.sum(act_name - act_i), 0)

        self.assertEqual(self.classifier.get_activations(self.x_test_mnist, 0).shape, (self.n_test, 6, 24, 24))
        self.assertEqual(self.classifier.get_activations(self.x_test_mnist, 4).shape, (self.n_test, 784))

    def test_set_learning(self):
        classifier = self.classifier

        self.assertFalse(hasattr(classifier, "_learning_phase"))
        classifier.set_learning_phase(False)
        self.assertFalse(classifier.learning_phase)
        classifier.set_learning_phase(True)
        self.assertTrue(classifier.learning_phase)
        self.assertTrue(hasattr(classifier, "_learning_phase"))

    def test_save(self):
        classifier = self.classifier
        t_file = tempfile.NamedTemporaryFile()
        full_path = t_file.name
        t_file.close()
        base_name = os.path.basename(full_path)
        dir_name = os.path.dirname(full_path)

        classifier.save(base_name, path=dir_name)
        self.assertTrue(os.path.exists(full_path + ".params"))
        os.remove(full_path + ".params")

    def test_repr(self):
        repr_ = repr(self.classifier)
        self.assertIn("art.estimators.classification.mxnet.MXClassifier", repr_)
        self.assertIn("input_shape=(1, 28, 28), nb_classes=10", repr_)
        self.assertIn(
            f"channel_index={Deprecated}, channels_first=True, clip_values=array([0., 1.], dtype=float32)", repr_
        )
        self.assertIn("defences=None, preprocessing=(0, 1)", repr_)


if __name__ == "__main__":
    unittest.main()
