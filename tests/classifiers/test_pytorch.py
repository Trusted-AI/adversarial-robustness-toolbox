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

import os
import logging
import unittest
import tempfile
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from art import DATA_PATH
from art.data_generators import PyTorchDataGenerator
from art.classifiers import PyTorchClassifier
from art.utils import load_dataset, master_seed
from art.utils_test import get_classifier_pt

logger = logging.getLogger('testLogger')

NB_TRAIN = 1000
NB_TEST = 20


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(1, 2, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(288, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = x.view(-1, 288)
        logit_output = self.fc(x)
        return logit_output


class Flatten(nn.Module):
    def forward(self, x):
        n, _, _, _ = x.size()
        result = x.view(n, -1)

        return result


class TestPyTorchClassifier(unittest.TestCase):
    """
    This class tests the PyTorch classifier.
    """

    @classmethod
    def setUpClass(cls):
        (x_train, y_train), (x_test, y_test), _, _ = load_dataset('mnist')

        x_train = np.swapaxes(x_train, 1, 3).astype(np.float32)
        x_test = np.swapaxes(x_test, 1, 3).astype(np.float32)

        cls.x_train = x_train[:NB_TRAIN]
        cls.y_train = y_train[:NB_TRAIN]
        cls.x_test = x_test[:NB_TEST]
        cls.y_test = y_test[:NB_TEST]

        # Define the network
        model = nn.Sequential(nn.Conv2d(1, 2, 5), nn.ReLU(), nn.MaxPool2d(2, 2), Flatten(), nn.Linear(288, 10))

        # Define a loss function and optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        classifier = PyTorchClassifier(model=model, clip_values=(0, 1), loss=loss_fn, optimizer=optimizer,
                                       input_shape=(1, 28, 28), nb_classes=10)
        classifier.fit(cls.x_train, cls.y_train, batch_size=100, nb_epochs=1)
        cls.seq_classifier = classifier

        # Define the network
        model = Model()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        classifier_2 = PyTorchClassifier(model=model, clip_values=(0, 1), loss=loss_fn, optimizer=optimizer,
                                         input_shape=(1, 28, 28), nb_classes=10)
        classifier_2.fit(x_train, y_train, batch_size=100, nb_epochs=1)
        cls.module_classifier = classifier_2

    def setUp(self):
        master_seed(1234)

    def test_fit_predict(self):
        classifier = get_classifier_pt()
        predictions = classifier.predict(self.x_test)
        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(self.y_test, axis=1)) / NB_TEST
        logger.info('Accuracy after fitting: %.2f%%', (accuracy * 100))
        self.assertEqual(accuracy, 0.3)

    def test_fit_generator(self):
        classifier = get_classifier_pt()
        accuracy = np.sum(
            np.argmax(classifier.predict(self.x_test), axis=1) == np.argmax(self.y_test, axis=1)) / NB_TEST
        logger.info('Accuracy: %.2f%%', (accuracy * 100))

        # Create tensors from data
        x_train_tens = torch.from_numpy(self.x_train)
        x_train_tens = x_train_tens.float()
        y_train_tens = torch.from_numpy(self.y_train)

        # Create PyTorch dataset and loader
        dataset = torch.utils.data.TensorDataset(x_train_tens, y_train_tens)
        data_loader = DataLoader(dataset=dataset, batch_size=5, shuffle=True)
        data_gen = PyTorchDataGenerator(data_loader, size=NB_TRAIN, batch_size=5)

        # Fit model with generator
        classifier.fit_generator(data_gen, nb_epochs=2)
        accuracy_2 = np.sum(
            np.argmax(classifier.predict(self.x_test), axis=1) == np.argmax(self.y_test, axis=1)) / NB_TEST
        logger.info('Accuracy: %.2f%%', (accuracy_2 * 100))

        self.assertEqual(accuracy, 0.3)
        self.assertAlmostEqual(accuracy_2, 0.7, delta=0.1)

    def test_nb_classes(self):
        classifier = get_classifier_pt()
        self.assertEqual(classifier.nb_classes(), 10)

    def test_input_shape(self):
        classifier = get_classifier_pt()
        self.assertEqual(classifier.input_shape, (1, 28, 28))

    def test_class_gradient(self):
        classifier = get_classifier_pt()

        # Test all gradients label = None
        gradients = classifier.class_gradient(self.x_test)

        self.assertEqual(gradients.shape, (NB_TEST, 10, 1, 28, 28))

        expected_gradients_1 = np.asarray([-0.00150054, -0.00143262, -0.00084528, 0.00430235,
                                           0.00461788, -0.00041574, 0.00425031, 0.00541148,
                                           0.00873945, -0.00144859, 0.00644154, -0.00305683,
                                           -0.00861008, -0.005047, 0.01267533, 0.00141767,
                                           0.00625692, -0.00056882, -0.00664478, -0.00221345,
                                           0.00451501, 0.00000000, 0.00000000, 0.00000000,
                                           0.00000000, 0.00000000, 0.00000000, 0.00000000])
        np.testing.assert_array_almost_equal(gradients[0, 5, 0, :, 14], expected_gradients_1, decimal=4)

        expected_gradients_2 = np.asarray([-0.00451381, -0.00108492, -0.00154162, -0.0025684,
                                           0.00280577, -0.00045833, 0.00133553, 0.00000000,
                                           0.00000000, -0.00713912, -0.00096829, 0.00775198,
                                           -0.00388135, 0.00725403, 0.01267533, 0.00680012,
                                           0.00754666, 0.00996057, 0.00415468, -0.00202466,
                                           -0.00184498, 0.00614046, -0.00652217, 0.00000000,
                                           0.00000000, 0.00000000, 0.00000000, 0.00000000])
        np.testing.assert_array_almost_equal(gradients[0, 5, 0, 14, :], expected_gradients_2, decimal=4)

        # Test 1 gradient label = 5
        gradients = classifier.class_gradient(self.x_test, label=5)

        self.assertEqual(gradients.shape, (NB_TEST, 1, 1, 28, 28))

        expected_gradients_1 = np.asarray([-0.00150054, -0.00143262, -0.00084528, 0.00430235,
                                           0.00461788, -0.00041574, 0.00425031, 0.00541148,
                                           0.00873945, -0.00144859, 0.00644154, -0.00305683,
                                           -0.00861008, -0.005047, 0.01267533, 0.00141767,
                                           0.00625692, -0.00056882, -0.00664478, -0.00221345,
                                           0.00451501, 0.00000000, 0.00000000, 0.00000000,
                                           0.00000000, 0.00000000, 0.00000000, 0.00000000])
        np.testing.assert_array_almost_equal(gradients[0, 0, 0, :, 14], expected_gradients_1, decimal=4)

        expected_gradients_2 = np.asarray([-0.00451381, -0.00108492, -0.00154162, -0.0025684,
                                           0.00280577, -0.00045833, 0.00133553, 0.00000000,
                                           0.00000000, -0.00713912, -0.00096829, 0.00775198,
                                           -0.00388135, 0.00725403, 0.01267533, 0.00680012,
                                           0.00754666, 0.00996057, 0.00415468, -0.00202466,
                                           -0.00184498, 0.00614046, -0.00652217, 0.00000000,
                                           0.00000000, 0.00000000, 0.00000000, 0.00000000])
        np.testing.assert_array_almost_equal(gradients[0, 0, 0, 14, :], expected_gradients_2, decimal=4)

        # Test a set of gradients label = array
        label = np.random.randint(5, size=NB_TEST)
        gradients = classifier.class_gradient(self.x_test, label=label)

        self.assertEqual(gradients.shape, (NB_TEST, 1, 1, 28, 28))

        expected_gradients_1 = np.asarray([-2.4956216e-03, -2.3826526e-03, -2.5130073e-03, -7.8883994e-04,
                                           5.7082525e-03, 5.8527971e-03, 3.7764946e-03, 4.5487015e-03,
                                           5.7944758e-03, -2.0162186e-03, 2.8324679e-03, 1.8695745e-03,
                                           -3.1888916e-04, 1.4913176e-03, 6.9031012e-03, 1.4145866e-03,
                                           1.5333943e-03, 2.6929809e-05, -3.3573490e-03, -1.1134071e-03,
                                           2.2711302e-03, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
                                           0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00])
        np.testing.assert_array_almost_equal(gradients[0, 0, 0, :, 14], expected_gradients_1, decimal=4)

        expected_gradients_2 = np.asarray([-0.003264, 0.00116614, 0.00425005, 0.00325745,
                                           0.00083933, 0.00457561, 0.00499626, 0.00000000,
                                           0.00000000, -0.00372264, -0.00048806, 0.0039387,
                                           -0.00194692, 0.00372874, 0.0069031, 0.00504249,
                                           0.00077568, -0.0002209, 0.00204011, 0.00011565,
                                           0.00013972, -0.00219941, 0.00097462, 0.00000000,
                                           0.00000000, 0.00000000, 0.00000000, 0.00000000])
        np.testing.assert_array_almost_equal(gradients[0, 0, 0, 14, :], expected_gradients_2, decimal=4)

    def test_class_gradient_target(self):
        classifier = get_classifier_pt()
        gradients = classifier.class_gradient(self.x_test, label=3)

        self.assertEqual(gradients.shape, (NB_TEST, 1, 1, 28, 28))

        expected_gradients_1 = np.asarray([-2.4956216e-03, -2.3826526e-03, -2.5130073e-03, -7.8883994e-04,
                                           5.7082525e-03, 5.8527971e-03, 3.7764946e-03, 4.5487015e-03,
                                           5.7944758e-03, -2.0162186e-03, 2.8324679e-03, 1.8695745e-03,
                                           -3.1888916e-04, 1.4913176e-03, 6.9031012e-03, 1.4145866e-03,
                                           1.5333943e-03, 2.6929809e-05, -3.3573490e-03, -1.1134071e-03,
                                           2.2711302e-03, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
                                           0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00])
        np.testing.assert_array_almost_equal(gradients[0, 0, 0, :, 14], expected_gradients_1, decimal=4)

        expected_gradients_2 = np.asarray([-0.003264, 0.00116614, 0.00425005, 0.00325745,
                                           0.00083933, 0.00457561, 0.00499626, 0.00000000,
                                           0.00000000, -0.00372264, -0.00048806, 0.0039387,
                                           -0.00194692, 0.00372874, 0.0069031, 0.00504249,
                                           0.00077568, -0.0002209, 0.00204011, 0.00011565,
                                           0.00013972, -0.00219941, 0.00097462, 0.00000000,
                                           0.00000000, 0.00000000, 0.00000000, 0.00000000])
        np.testing.assert_array_almost_equal(gradients[0, 0, 0, 14, :], expected_gradients_2, decimal=4)

    def test_loss_gradient(self):
        classifier = get_classifier_pt()
        gradients = classifier.loss_gradient(self.x_test, self.y_test)

        self.assertEqual(gradients.shape, (NB_TEST, 1, 28, 28))

        expected_gradients_1 = np.asarray([1.15053124e-04, 1.09845030e-04, 1.15488816e-04, 3.37422716e-05,
                                           1.84526041e-04, 1.60381125e-04, 3.05866299e-04, 1.81207652e-04,
                                           6.27528992e-04, -8.84818073e-05, 4.38439834e-04, 9.20038365e-05,
                                           -4.22611454e-04, -8.74137331e-05, 1.77365995e-03, 6.17997837e-04,
                                           -1.99291739e-04, 1.21479861e-04, -8.62729270e-04, -2.84300098e-04,
                                           5.79916057e-04, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                           0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00])
        np.testing.assert_array_almost_equal(gradients[0, 0, :, 14], expected_gradients_1, decimal=4)

        expected_gradients_2 = np.asarray([-6.5131334e-04, 3.0298581e-04, 1.0413901e-03, 8.3430990e-04,
                                           1.2461779e-04, 1.0898571e-03, 1.1422117e-03, 0.0000000e+00,
                                           0.0000000e+00, -6.2288583e-04, -1.2216593e-04, 9.0777961e-04,
                                           -5.1075104e-04, 8.0109172e-04, 1.7736600e-03, 1.5165898e-03,
                                           1.3210204e-03, 6.4311037e-04, 1.2632636e-03, 2.8421549e-04,
                                           -1.2425387e-04, 7.5472635e-05, -3.4554966e-04, 0.0000000e+00,
                                           0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00])
        np.testing.assert_array_almost_equal(gradients[0, 0, 14, :], expected_gradients_2, decimal=4)

    def test_layers(self):
        ptc = self.seq_classifier
        layer_names = self.seq_classifier.layer_names
        print(layer_names)
        self.assertEqual(layer_names, ['0_Conv2d(1, 2, kernel_size=(5, 5), stride=(1, 1))', '1_ReLU()',
                                       '2_MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)',
                                       '3_Flatten()', '4_Linear(in_features=288, out_features=10, bias=True)'])

        for i, name in enumerate(layer_names):
            activation_i = ptc.get_activations(self.x_test, i, batch_size=5)
            activation_name = ptc.get_activations(self.x_test, name, batch_size=5)
            np.testing.assert_array_equal(activation_name, activation_i)

        self.assertEqual(ptc.get_activations(self.x_test, 0, batch_size=5).shape, (20, 2, 24, 24))
        self.assertEqual(ptc.get_activations(self.x_test, 1, batch_size=5).shape, (20, 2, 24, 24))
        self.assertEqual(ptc.get_activations(self.x_test, 2, batch_size=5).shape, (20, 2, 12, 12))
        self.assertEqual(ptc.get_activations(self.x_test, 3, batch_size=5).shape, (20, 288))
        self.assertEqual(ptc.get_activations(self.x_test, 4, batch_size=5).shape, (20, 10))

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
        full_path = os.path.join(DATA_PATH, 'my_classifier')
        folder = os.path.split(full_path)[0]
        if not os.path.exists(folder):
            os.makedirs(folder)

        pickle.dump(self.module_classifier, open(full_path, 'wb'))

        # Unpickle:
        with open(full_path, 'rb') as f:
            loaded = pickle.load(f)
            self.assertEqual(self.module_classifier._clip_values, loaded._clip_values)
            self.assertEqual(self.module_classifier._channel_index, loaded._channel_index)
            self.assertEqual(set(self.module_classifier.__dict__.keys()), set(loaded.__dict__.keys()))

        # Test predict
        predictions_1 = self.module_classifier.predict(self.x_test)
        accuracy_1 = np.sum(np.argmax(predictions_1, axis=1) == np.argmax(self.y_test, axis=1)) / len(self.y_test)
        predictions_2 = loaded.predict(self.x_test)
        accuracy_2 = np.sum(np.argmax(predictions_2, axis=1) == np.argmax(self.y_test, axis=1)) / len(self.y_test)
        self.assertEqual(accuracy_1, accuracy_2)


if __name__ == '__main__':
    unittest.main()
