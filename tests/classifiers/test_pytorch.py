from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from art.classifiers.pytorch import PyTorchClassifier
from art.utils import load_mnist, master_seed

logger = logging.getLogger('testLogger')

NB_TRAIN = 1000
NB_TEST = 20


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(1, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(2304, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = x.view(-1, 2304)
        logit_output = self.fc(x)

        return logit_output


class Flatten(nn.Module):
    def forward(self, x):
        n, _, _, _ = x.size()
        result = x.view(n, -1)

        return result


class TestPyTorchClassifier(unittest.TestCase):
    """
    This class tests the functionalities of the PyTorch-based classifier.
    """

    @classmethod
    def setUpClass(cls):
        # Get MNIST
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        x_train, y_train = x_train[:NB_TRAIN], y_train[:NB_TRAIN]
        x_test, y_test = x_test[:NB_TEST], y_test[:NB_TEST]
        x_train = np.swapaxes(x_train, 1, 3)
        x_test = np.swapaxes(x_test, 1, 3)
        cls.mnist = (x_train, y_train), (x_test, y_test)

        # Define the network
        model = nn.Sequential(nn.Conv2d(1, 16, 5), nn.ReLU(), nn.MaxPool2d(2, 2), Flatten(), nn.Linear(2304, 10))

        # Define a loss function and optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        classifier = PyTorchClassifier(clip_values=(0, 1), model=model, loss=loss_fn, optimizer=optimizer,
                                       input_shape=(1, 28, 28), nb_classes=10)
        classifier.fit(x_train, y_train, batch_size=100, nb_epochs=2)
        cls.seq_classifier = classifier

        # Define the network
        model = Model()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        classifier2 = PyTorchClassifier(clip_values=(0, 1), model=model, loss=loss_fn, optimizer=optimizer,
                                        input_shape=(1, 28, 28), nb_classes=10)
        classifier2.fit(x_train, y_train, batch_size=100, nb_epochs=2)
        cls.module_classifier = classifier2

    def setUp(self):
        # Set master seed
        master_seed(1234)

    def test_fit_predict(self):
        # Get MNIST
        (_, _), (x_test, y_test) = self.mnist

        # Test predict
        preds = self.module_classifier.predict(x_test)
        acc = np.sum(np.argmax(preds, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
        logger.info('Accuracy after fitting: %.2f%%', (acc * 100))
        self.assertGreater(acc, 0.1)

    def test_fit_generator(self):
        import torch
        from torch.utils.data import DataLoader
        from art.data_generators import PyTorchDataGenerator

        (x_train, y_train), (x_test, y_test) = self.mnist
        acc = np.sum(np.argmax(self.module_classifier.predict(x_test), axis=1) == np.argmax(y_test, axis=1)) / NB_TEST
        logger.info('Accuracy: %.2f%%', (acc * 100))

        # Create tensors from data
        x_train_tens = torch.from_numpy(x_train)
        x_train_tens = x_train_tens.float()
        y_train_tens = torch.from_numpy(y_train)

        # Create PyTorch dataset and loader
        dataset = torch.utils.data.TensorDataset(x_train_tens, y_train_tens)
        data_loader = DataLoader(dataset=dataset, batch_size=5, shuffle=True)
        data_gen = PyTorchDataGenerator(data_loader, size=NB_TRAIN, batch_size=5)

        # Fit model with generator
        self.module_classifier.fit_generator(data_gen, nb_epochs=2)
        acc2 = np.sum(np.argmax(self.module_classifier.predict(x_test), axis=1) == np.argmax(y_test, axis=1)) / NB_TEST
        logger.info('Accuracy: %.2f%%', (acc * 100))

        self.assertGreaterEqual(acc2, 0.8 * acc)

    def test_nb_classes(self):
        ptc = self.module_classifier
        self.assertEqual(ptc.nb_classes, 10)

    def test_input_shape(self):
        ptc = self.module_classifier
        self.assertTrue(np.array(ptc.input_shape == (1, 28, 28)).all())

    def test_class_gradient(self):
        # Get MNIST
        (_, _), (x_test, _) = self.mnist

        # Test all gradients label = None
        ptc = self.module_classifier
        grads = ptc.class_gradient(x_test)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 10, 1, 28, 28)).all())
        self.assertNotEqual(np.sum(grads), 0)

        # Test 1 gradient label = 5
        grads = ptc.class_gradient(x_test, label=5)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 1, 1, 28, 28)).all())
        self.assertNotEqual(np.sum(grads), 0)

        # Test a set of gradients label = array
        label = np.random.randint(5, size=NB_TEST)
        grads = ptc.class_gradient(x_test, label=label)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 1, 1, 28, 28)).all())
        self.assertNotEqual(np.sum(grads), 0)

    def test_class_gradient_target(self):
        # Get MNIST
        (_, _), (x_test, _) = self.mnist

        # Test gradient
        ptc = self.module_classifier
        grads = ptc.class_gradient(x_test, label=3)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 1, 1, 28, 28)).all())
        self.assertNotEqual(np.sum(grads), 0)

    def test_loss_gradient(self):
        # Get MNIST
        (_, _), (x_test, y_test) = self.mnist

        # Test gradient
        ptc = self.module_classifier
        grads = ptc.loss_gradient(x_test, y_test)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 1, 28, 28)).all())
        self.assertNotEqual(np.sum(grads), 0)

    def test_layers(self):
        # Get MNIST
        (_, _), (x_test, _) = self.mnist

        # Test and get layers
        ptc = self.seq_classifier

        layer_names = self.seq_classifier.layer_names
        self.assertEqual(layer_names, ['0_Conv2d(1, 16, kernel_size=(5, 5), stride=(1, 1))', '1_ReLU()',
                                       '2_MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)',
                                       '3_Flatten()', '4_Linear(in_features=2304, out_features=10, bias=True)'])

        for i, name in enumerate(layer_names):
            act_i = ptc.get_activations(x_test, i, batch_size=5)
            act_name = ptc.get_activations(x_test, name, batch_size=5)
            self.assertEqual(np.sum(act_name - act_i), 0)

        self.assertEqual(ptc.get_activations(x_test, 0, batch_size=5).shape, (20, 16, 24, 24))
        self.assertEqual(ptc.get_activations(x_test, 1, batch_size=5).shape, (20, 16, 24, 24))
        self.assertEqual(ptc.get_activations(x_test, 2, batch_size=5).shape, (20, 16, 12, 12))
        self.assertEqual(ptc.get_activations(x_test, 3, batch_size=5).shape, (20, 2304))
        self.assertEqual(ptc.get_activations(x_test, 4, batch_size=5).shape, (20, 10))

    def test_set_learning(self):
        ptc = self.module_classifier

        self.assertTrue(ptc._model.training)
        ptc.set_learning_phase(False)
        self.assertFalse(ptc._model.training)
        ptc.set_learning_phase(True)
        self.assertTrue(ptc._model.training)
        self.assertTrue(ptc.learning_phase)

    def test_save(self):
        model = self.module_classifier
        import tempfile
        import os
        t_file = tempfile.NamedTemporaryFile()
        full_path = t_file.name
        t_file.close()
        base_name = os.path.basename(full_path)
        dir_name = os.path.dirname(full_path)
        model.save(base_name, path=dir_name)
        self.assertTrue(os.path.exists(full_path + ".optimizer"))
        self.assertTrue(os.path.exists(full_path + ".model"))
        os.remove(full_path + '.optimizer')
        os.remove(full_path + '.model')

    def test_repr(self):
        repr_ = repr(self.module_classifier)
        self.assertIn('art.classifiers.pytorch.PyTorchClassifier', repr_)
        self.assertIn('input_shape=(1, 28, 28), nb_classes=10, channel_index=1', repr_)
        self.assertIn('clip_values=(0, 1)', repr_)
        self.assertIn('defences=None, preprocessing=(0, 1)', repr_)

    def test_pickle(self):
        import os
        from art import DATA_PATH
        full_path = os.path.join(DATA_PATH, 'my_classifier')
        folder = os.path.split(full_path)[0]
        if not os.path.exists(folder):
            os.makedirs(folder)

        import pickle
        pickle.dump(self.module_classifier, open(full_path, 'wb'))

        # Unpickle:
        with open(full_path, 'rb') as f:
            loaded = pickle.load(f)
            self.assertEqual(self.module_classifier._clip_values, loaded._clip_values)
            self.assertEqual(self.module_classifier._channel_index, loaded._channel_index)
            self.assertEqual(set(self.module_classifier.__dict__.keys()), set(loaded.__dict__.keys()))

        # Get MNIST
        (_, _), (x_test, y_test) = self.mnist

        # Test predict
        preds1 = self.module_classifier.predict(x_test)
        acc1 = np.sum(np.argmax(preds1, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
        preds2 = loaded.predict(x_test)
        acc2 = np.sum(np.argmax(preds2, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
        self.assertEqual(acc1, acc2)


if __name__ == '__main__':
    unittest.main()
