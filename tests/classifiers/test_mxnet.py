from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

import numpy as np
from mxnet import init, gluon
from mxnet.gluon import nn

from art.classifiers import MXClassifier
from art.utils import load_mnist, master_seed

logger = logging.getLogger('testLogger')

NB_TRAIN = 1000
NB_TEST = 20


class TestMXClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Get MNIST
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        x_train, y_train = x_train[:NB_TRAIN], y_train[:NB_TRAIN]
        x_test, y_test = x_test[:NB_TEST], y_test[:NB_TEST]
        x_train = np.swapaxes(x_train, 1, 3)
        x_test = np.swapaxes(x_test, 1, 3)
        cls.mnist = (x_train, y_train), (x_test, y_test)

        # Create a simple CNN - this one comes from the Gluon tutorial
        net = nn.Sequential()
        with net.name_scope():
            net.add(
                nn.Conv2D(channels=6, kernel_size=5, activation='relu'),
                nn.MaxPool2D(pool_size=2, strides=2),
                nn.Conv2D(channels=16, kernel_size=3, activation='relu'),
                nn.MaxPool2D(pool_size=2, strides=2),
                nn.Flatten(),
                nn.Dense(120, activation="relu"),
                nn.Dense(84, activation="relu"),
                nn.Dense(10)
            )
        net.initialize(init=init.Xavier())

        # Create optimizer
        trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

        # Fit classifier
        classifier = MXClassifier((0, 1), net, (1, 28, 28), 10, trainer)
        classifier.fit(x_train, y_train, batch_size=128, nb_epochs=2)
        cls.classifier = classifier

    def setUp(self):
        # Set master seed
        master_seed(1234)

    def test_fit_predict(self):
        (_, _), (x_test, y_test) = self.mnist

        preds = self.classifier.predict(x_test)
        acc = np.sum(np.argmax(preds, axis=1) == np.argmax(y_test, axis=1)) / NB_TEST
        logger.info('Accuracy after fitting: %.2f%%', (acc * 100))
        self.assertGreater(acc, 0.1)

    def test_fit_generator(self):
        import mxnet as mx
        from art.data_generators import MXDataGenerator

        (x_train, y_train), (x_test, y_test) = self.mnist
        acc = np.sum(np.argmax(self.classifier.predict(x_test), axis=1) == np.argmax(y_test, axis=1)) / NB_TEST
        logger.info('Accuracy: %.2f%%', (acc * 100))

        # Create MXNet dataset and loader
        dataset = mx.gluon.data.dataset.ArrayDataset(x_train, y_train)
        data_loader = mx.gluon.data.DataLoader(dataset, batch_size=5, shuffle=True)
        data_gen = MXDataGenerator(data_loader, size=NB_TRAIN, batch_size=5)

        # Fit model with generator
        self.classifier.fit_generator(data_gen, nb_epochs=2)
        acc2 = np.sum(np.argmax(self.classifier.predict(x_test), axis=1) == np.argmax(y_test, axis=1)) / NB_TEST
        logger.info('Accuracy: %.2f%%', (acc * 100))

        self.assertTrue(acc2 >= .8 * acc)

    def test_nb_classes(self):
        self.assertEqual(self.classifier.nb_classes, 10)

    def test_input_shape(self):
        self.assertEqual(self.classifier.input_shape, (1, 28, 28))

    def test_class_gradient(self):
        # Get MNIST
        (_, _), (x_test, _) = self.mnist

        # Test class grads for all classes
        grads_all = self.classifier.class_gradient(x_test)
        self.assertTrue(np.array(grads_all.shape == (NB_TEST, 10, 1, 28, 28)).all())
        self.assertTrue(np.sum(grads_all) != 0)

        # Test class grads for specified label
        grads = self.classifier.class_gradient(x_test, label=3)
        self.assertTrue(np.array(grads.shape == (NB_TEST, 1, 1, 28, 28)).all())
        self.assertTrue(np.sum(grads) != 0)

        # Assert gradient computed for the same class on same input are equal
        self.assertAlmostEqual(np.sum(grads_all[:, 3] - grads), 0, places=6)

        # Test a set of gradients label = array
        labels = np.random.randint(5, size=NB_TEST)
        grads = self.classifier.class_gradient(x_test, label=labels)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 1, 1, 28, 28)).all())
        self.assertTrue(np.sum(grads) != 0)

    def test_loss_gradient(self):
        # Get MNIST
        (_, _), (x_test, y_test) = self.mnist

        # Compute loss gradients
        grads = self.classifier.loss_gradient(x_test, y_test)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 1, 28, 28)).all())
        self.assertTrue(np.sum(grads) != 0)

    def test_preprocessing(self):
        # Get MNIST
        (_, _), (x_test, _) = self.mnist

        # Create classifier
        classifier_preproc = MXClassifier((0, 1), self.classifier._model, (1, 28, 28), 10, self.classifier._optimizer,
                                          preprocessing=(1, 2))

        preds = self.classifier.predict((x_test - 1.) / 2)
        preds_preproc = classifier_preproc.predict(x_test)
        self.assertTrue(np.sum(preds - preds_preproc) == 0)

    def test_layers(self):
        # Get MNIST
        (_, _), (x_test, _) = self.mnist

        self.assertEqual(len(self.classifier.layer_names), 7)

        # layer_names = self.classifier.layer_names
        # for i, name in enumerate(layer_names):
        #     act_i = self.classifier.get_activations(x_test, i)
        #     act_name = self.classifier.get_activations(x_test, name)
        #     self.assertAlmostEqual(np.sum(act_name - act_i), 0)

        self.assertTrue(self.classifier.get_activations(x_test, 0).shape == (NB_TEST, 6, 24, 24))
        self.assertTrue(self.classifier.get_activations(x_test, 4).shape == (NB_TEST, 784))

    def test_set_learning(self):
        classifier = self.classifier

        self.assertFalse(hasattr(classifier, '_learning_phase'))
        classifier.set_learning_phase(False)
        self.assertFalse(classifier.learning_phase)
        classifier.set_learning_phase(True)
        self.assertTrue(classifier.learning_phase)
        self.assertTrue(hasattr(classifier, '_learning_phase'))

    def test_save(self):
        import tempfile
        import os

        classifier = self.classifier
        t_file = tempfile.NamedTemporaryFile()
        full_path = t_file.name
        t_file.close()
        base_name = os.path.basename(full_path)
        dir_name = os.path.dirname(full_path)

        classifier.save(base_name, path=dir_name)
        self.assertTrue(os.path.exists(full_path + ".params"))
        os.remove(full_path + '.params')

    def test_repr(self):
        repr_ = repr(self.classifier)
        self.assertTrue('art.classifiers.mxnet.MXClassifier' in repr_)
        self.assertTrue('clip_values=(0, 1)' in repr_)
        self.assertTrue('input_shape=(1, 28, 28), nb_classes=10' in repr_)
        self.assertTrue('channel_index=1, defences=None, preprocessing=(0, 1)' in repr_)


if __name__ == '__main__':
    unittest.main()
