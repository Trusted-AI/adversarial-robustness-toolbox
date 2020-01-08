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

from art.config import ART_DATA_PATH
from art.data_generators import PyTorchDataGenerator
from art.classifiers import PyTorchClassifier
from art.utils import load_dataset, master_seed
from tests.utils_test import get_classifier_pt

logger = logging.getLogger(__name__)

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

        x_train = np.reshape(x_train, (x_train.shape[0], 1, 28, 28)).astype(np.float32)
        x_test = np.reshape(x_test, (x_test.shape[0], 1, 28, 28)).astype(np.float32)

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
        self.assertEqual(accuracy, 0.4)

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

        self.assertEqual(accuracy, 0.4)
        self.assertAlmostEqual(accuracy_2, 0.75, delta=0.1)

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

        expected_gradients_1 = np.asarray([-0.00367321, -0.0002892, 0.00037825, -0.00053344, 0.00192121, 0.00112047,
                                           0.0023135, 0.0, 0.0, -0.00391743, -0.0002264, 0.00238103,
                                           -0.00073711, 0.00270405, 0.00389043, 0.00440818, -0.00412769, -0.00441795,
                                           0.00081916, -0.00091284, 0.00119645, -0.00849089, 0.00547925, 0.0,
                                           0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(gradients[0, 5, 0, :, 14], expected_gradients_1, decimal=4)

        expected_gradients_2 = np.asarray([-1.0557442e-03, -1.0079540e-03, -7.7426381e-04, 1.7387437e-03,
                                           2.1773505e-03, 5.0880131e-05, 1.6497375e-03, 2.6113102e-03,
                                           6.0904315e-03, 4.1080985e-04, 2.5268074e-03, -3.6661496e-04,
                                           -3.0568994e-03, -1.1665225e-03, 3.8904310e-03, 3.1726388e-04,
                                           1.3203262e-03, -1.1720933e-04, -1.4315107e-03, -4.7676827e-04,
                                           9.7251305e-04, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
                                           0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00])
        np.testing.assert_array_almost_equal(gradients[0, 5, 0, 14, :], expected_gradients_2, decimal=4)

        # Test 1 gradient label = 5
        gradients = classifier.class_gradient(self.x_test, label=5)

        self.assertEqual(gradients.shape, (NB_TEST, 1, 1, 28, 28))

        expected_gradients_1 = np.asarray([-0.00367321, -0.0002892, 0.00037825, -0.00053344, 0.00192121, 0.00112047,
                                           0.0023135, 0.0, 0.0, -0.00391743, -0.0002264, 0.00238103,
                                           -0.00073711, 0.00270405, 0.00389043, 0.00440818, -0.00412769, -0.00441795,
                                           0.00081916, -0.00091284, 0.00119645, -0.00849089, 0.00547925, 0.0,
                                           0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(gradients[0, 0, 0, :, 14], expected_gradients_1, decimal=4)

        expected_gradients_2 = np.asarray([-1.0557442e-03, -1.0079540e-03, -7.7426381e-04, 1.7387437e-03,
                                           2.1773505e-03, 5.0880131e-05, 1.6497375e-03, 2.6113102e-03,
                                           6.0904315e-03, 4.1080985e-04, 2.5268074e-03, -3.6661496e-04,
                                           -3.0568994e-03, -1.1665225e-03, 3.8904310e-03, 3.1726388e-04,
                                           1.3203262e-03, -1.1720933e-04, -1.4315107e-03, -4.7676827e-04,
                                           9.7251305e-04, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
                                           0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00])
        np.testing.assert_array_almost_equal(gradients[0, 0, 0, 14, :], expected_gradients_2, decimal=4)

        # Test a set of gradients label = array
        label = np.random.randint(5, size=NB_TEST)
        gradients = classifier.class_gradient(self.x_test, label=label)

        self.assertEqual(gradients.shape, (NB_TEST, 1, 1, 28, 28))

        expected_gradients_1 = np.asarray([-0.00195835, -0.00134457, -0.00307221, -0.00340564, 0.00175022, -0.00239714,
                                           -0.00122619, 0.0, 0.0, -0.00520899, -0.00046105, 0.00414874,
                                           -0.00171095, 0.00429184, 0.0075138, 0.00792443, 0.0019566, 0.00035517,
                                           0.00504575, -0.00037397, 0.00022343, -0.00530035, 0.0020528, 0.0,
                                           0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(gradients[0, 0, 0, :, 14], expected_gradients_1, decimal=4)

        expected_gradients_2 = np.asarray([5.0867130e-03, 4.8564533e-03, 6.1040395e-03, 8.6531248e-03,
                                           -6.0958802e-03, -1.4114541e-02, -7.1085966e-04, -5.0330797e-04,
                                           1.2943064e-02, 8.2416134e-03, -1.9859453e-04, -9.8110031e-05,
                                           -3.8902226e-03, -1.2945874e-03, 7.5138002e-03, 1.7720887e-03,
                                           3.1399354e-04, 2.3657191e-04, -3.0891625e-03, -1.0211228e-03,
                                           2.0828887e-03, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
                                           0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00])
        np.testing.assert_array_almost_equal(gradients[0, 0, 0, 14, :], expected_gradients_2, decimal=4)

    def test_class_gradient_target(self):
        classifier = get_classifier_pt()
        gradients = classifier.class_gradient(self.x_test, label=3)

        self.assertEqual(gradients.shape, (NB_TEST, 1, 1, 28, 28))

        expected_gradients_1 = np.asarray([-0.00195835, -0.00134457, -0.00307221, -0.00340564, 0.00175022, -0.00239714,
                                           -0.00122619, 0.0, 0.0, -0.00520899, -0.00046105, 0.00414874,
                                           -0.00171095, 0.00429184, 0.0075138, 0.00792443, 0.0019566, 0.00035517,
                                           0.00504575, -0.00037397, 0.00022343, -0.00530035, 0.0020528, 0.0,
                                           0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(gradients[0, 0, 0, :, 14], expected_gradients_1, decimal=4)

        expected_gradients_2 = np.asarray([5.0867130e-03, 4.8564533e-03, 6.1040395e-03, 8.6531248e-03,
                                           -6.0958802e-03, -1.4114541e-02, -7.1085966e-04, -5.0330797e-04,
                                           1.2943064e-02, 8.2416134e-03, -1.9859453e-04, -9.8110031e-05,
                                           -3.8902226e-03, -1.2945874e-03, 7.5138002e-03, 1.7720887e-03,
                                           3.1399354e-04, 2.3657191e-04, -3.0891625e-03, -1.0211228e-03,
                                           2.0828887e-03, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
                                           0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00])
        np.testing.assert_array_almost_equal(gradients[0, 0, 0, 14, :], expected_gradients_2, decimal=4)

    def test_loss_gradient(self):
        classifier = get_classifier_pt()
        gradients = classifier.loss_gradient(self.x_test, self.y_test)

        self.assertEqual(gradients.shape, (NB_TEST, 1, 28, 28))

        expected_gradients_1 = np.asarray([3.6839640e-05, 3.2549749e-05, 7.7749821e-05, 8.3091691e-05,
                                           -3.7349419e-05, 6.3347623e-05, 3.8059810e-05, 0.0000000e+00,
                                           0.0000000e+00, -8.7319646e-04, -9.1992842e-05, 7.8577449e-04,
                                           -3.5397310e-04, 7.8797276e-04, 1.6001392e-03, 1.9111208e-03,
                                           1.0337514e-03, 2.0264980e-04, 1.5017156e-03, 2.5167916e-04,
                                           -4.8513880e-06, -8.3324237e-04, 2.1826664e-04, 0.0000000e+00,
                                           0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00])
        np.testing.assert_array_almost_equal(gradients[0, 0, :, 14], expected_gradients_1, decimal=4)

        expected_gradients_2 = np.asarray([8.3541102e-04, 7.9759455e-04, 9.6892234e-04, 1.1802778e-03,
                                           -6.0561800e-04, -1.6849663e-03, 2.7197969e-04, 4.3571385e-05,
                                           1.2168724e-03, 4.9924687e-04, 4.7540435e-04, -3.6275905e-04,
                                           -1.1702902e-03, -7.0383825e-04, 1.6001392e-03, 6.1103603e-04,
                                           -5.1674922e-04, 1.6046617e-04, -6.3084543e-04, -2.0675475e-04,
                                           4.2173881e-04, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
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
        full_path = os.path.join(ART_DATA_PATH, 'my_classifier')
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
