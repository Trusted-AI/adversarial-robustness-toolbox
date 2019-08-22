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

        x_train = np.swapaxes(x_train, 1, 3)
        x_test = np.swapaxes(x_test, 1, 3)

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
        self.assertAlmostEqual(accuracy_2, 0.55, delta=0.05)

    def test_nb_classes(self):
        classifier = get_classifier_pt()
        self.assertEqual(classifier.nb_classes, 10)

    def test_input_shape(self):
        classifier = get_classifier_pt()
        self.assertEqual(classifier.input_shape, (1, 28, 28))

    def test_class_gradient(self):
        classifier = get_classifier_pt()

        # Test all gradients label = None
        gradients = classifier.class_gradient(self.x_test)

        self.assertEqual(gradients.shape, (NB_TEST, 10, 1, 28, 28))

        expected_gradients_1 = np.asarray([-1.5104107e-04, -1.4420391e-04, -8.5643369e-05, 4.2904957e-04,
                                           4.6917787e-04, -3.3433505e-05, 4.3190207e-04, 5.4948201e-04,
                                           9.0309686e-04, -1.3793766e-04, 6.4290554e-04, -2.8910500e-04,
                                           -8.5047574e-04, -4.8866379e-04, 1.3007881e-03, 1.5724849e-04,
                                           6.1079778e-04, -5.2589108e-05, -6.7920942e-04, -2.2616469e-04,
                                           4.6133125e-04, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
                                           0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00])
        np.testing.assert_array_almost_equal(gradients[0, 5, 0, :, 14], expected_gradients_1, decimal=4)

        expected_gradients_2 = np.asarray([-4.7446613e-04, -9.8460718e-05, -1.1919734e-04, -2.2912446e-04,
                                           2.8542569e-04, -8.9842360e-06, 1.7257492e-04, 0.0000000e+00,
                                           0.0000000e+00, -7.2422711e-04, -9.8898512e-05, 7.9051330e-04,
                                           -3.9680302e-04, 7.3898572e-04, 1.3007881e-03, 7.1805675e-04,
                                           7.7588821e-04, 9.9411258e-04, 4.4845918e-04, -1.9343558e-04,
                                           -1.8357937e-04, 5.9276586e-04, -6.4391940e-04, 0.0000000e+00,
                                           0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00])
        np.testing.assert_array_almost_equal(gradients[0, 5, 0, 14, :], expected_gradients_2, decimal=4)

        # Test 1 gradient label = 5
        gradients = classifier.class_gradient(self.x_test, label=5)

        self.assertEqual(gradients.shape, (NB_TEST, 1, 1, 28, 28))

        expected_gradients_1 = np.asarray([-1.5104107e-04, -1.4420391e-04, -8.5643369e-05, 4.2904957e-04,
                                           4.6917787e-04, -3.3433505e-05, 4.3190207e-04, 5.4948201e-04,
                                           9.0309686e-04, -1.3793766e-04, 6.4290554e-04, -2.8910500e-04,
                                           -8.5047574e-04, -4.8866379e-04, 1.3007881e-03, 1.5724849e-04,
                                           6.1079778e-04, -5.2589108e-05, -6.7920942e-04, -2.2616469e-04,
                                           4.6133125e-04, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
                                           0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00])
        np.testing.assert_array_almost_equal(gradients[0, 0, 0, :, 14], expected_gradients_1, decimal=4)

        expected_gradients_2 = np.asarray([-4.7446613e-04, -9.8460718e-05, -1.1919734e-04, -2.2912446e-04,
                                           2.8542569e-04, -8.9842360e-06, 1.7257492e-04, 0.0000000e+00,
                                           0.0000000e+00, -7.2422711e-04, -9.8898512e-05, 7.9051330e-04,
                                           -3.9680302e-04, 7.3898572e-04, 1.3007881e-03, 7.1805675e-04,
                                           7.7588821e-04, 9.9411258e-04, 4.4845918e-04, -1.9343558e-04,
                                           -1.8357937e-04, 5.9276586e-04, -6.4391940e-04, 0.0000000e+00,
                                           0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00])
        np.testing.assert_array_almost_equal(gradients[0, 0, 0, 14, :], expected_gradients_2, decimal=4)

        # Test a set of gradients label = array
        label = np.random.randint(5, size=NB_TEST)
        gradients = classifier.class_gradient(self.x_test, label=label)

        self.assertEqual(gradients.shape, (NB_TEST, 1, 1, 28, 28))

        expected_gradients_1 = np.asarray([-2.39315428e-04, -2.28482357e-04, -2.39842790e-04, -6.74667899e-05,
                                           5.54567552e-04, 5.59428358e-04, 3.71058501e-04, 4.47539205e-04,
                                           5.91437332e-04, -1.86359655e-04, 2.78284366e-04, 1.86634657e-04,
                                           -3.62139835e-05, 1.46655992e-04, 7.07449333e-04, 1.51086148e-04,
                                           1.42195524e-04, 5.60022090e-06, -3.43588385e-04, -1.13900256e-04,
                                           2.32334001e-04, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                           0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00])
        np.testing.assert_array_almost_equal(gradients[0, 0, 0, :, 14], expected_gradients_1, decimal=4)

        expected_gradients_2 = np.asarray([-3.3879644e-04, 1.1766610e-04, 4.3186071e-04, 3.2926403e-04,
                                           8.9179150e-05, 4.6644508e-04, 5.1162497e-04, 0.0000000e+00,
                                           0.0000000e+00, -3.7474511e-04, -4.9881939e-05, 4.0110818e-04,
                                           -1.9942035e-04, 3.7868350e-04, 7.0744933e-04, 5.2536512e-04,
                                           1.0788649e-04, -3.9121151e-06, 2.3214625e-04, 1.5785350e-05,
                                           1.0586554e-05, -2.1646731e-04, 8.7655055e-05, 0.0000000e+00,
                                           0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00])
        np.testing.assert_array_almost_equal(gradients[0, 0, 0, 14, :], expected_gradients_2, decimal=4)

    def test_class_gradient_target(self):
        classifier = get_classifier_pt()
        gradients = classifier.class_gradient(self.x_test, label=3)

        self.assertEqual(gradients.shape, (NB_TEST, 1, 1, 28, 28))

        expected_gradients_1 = np.asarray([-2.39315428e-04, -2.28482357e-04, -2.39842790e-04, -6.74667899e-05,
                                           5.54567552e-04, 5.59428358e-04, 3.71058501e-04, 4.47539205e-04,
                                           5.91437332e-04, -1.86359655e-04, 2.78284366e-04, 1.86634657e-04,
                                           -3.62139835e-05, 1.46655992e-04, 7.07449333e-04, 1.51086148e-04,
                                           1.42195524e-04, 5.60022090e-06, -3.43588385e-04, -1.13900256e-04,
                                           2.32334001e-04, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                           0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00])
        np.testing.assert_array_almost_equal(gradients[0, 0, 0, :, 14], expected_gradients_1, decimal=4)

        expected_gradients_2 = np.asarray([-3.3879644e-04, 1.1766610e-04, 4.3186071e-04, 3.2926403e-04,
                                           8.9179150e-05, 4.6644508e-04, 5.1162497e-04, 0.0000000e+00,
                                           0.0000000e+00, -3.7474511e-04, -4.9881939e-05, 4.0110818e-04,
                                           -1.9942035e-04, 3.7868350e-04, 7.0744933e-04, 5.2536512e-04,
                                           1.0788649e-04, -3.9121151e-06, 2.3214625e-04, 1.5785350e-05,
                                           1.0586554e-05, -2.1646731e-04, 8.7655055e-05, 0.0000000e+00,
                                           0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00])
        np.testing.assert_array_almost_equal(gradients[0, 0, 0, 14, :], expected_gradients_2, decimal=4)

    def test_loss_gradient(self):
        classifier = get_classifier_pt()
        gradients = classifier.loss_gradient(self.x_test, self.y_test)

        self.assertEqual(gradients.shape, (NB_TEST, 1, 28, 28))

        expected_gradients_1 = np.asarray([1.3249977e-05, 1.2650192e-05, 1.3294257e-05, 3.8435855e-06,
                                           2.1136035e-05, 1.8405482e-05, 3.5095429e-05, 2.0741667e-05,
                                           7.1921611e-05, -1.0196869e-05, 5.0357849e-05, 1.0525340e-05,
                                           -4.8562586e-05, -1.0082738e-05, 2.0365241e-04, 7.0968767e-05,
                                           -2.2902255e-05, 1.3952388e-05, -9.9061588e-05, -3.2644271e-05,
                                           6.6587862e-05, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
                                           0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00])
        np.testing.assert_array_almost_equal(gradients[0, 0, :, 14], expected_gradients_1, decimal=4)

        expected_gradients_2 = np.asarray([-7.47286831e-05, 3.47994683e-05, 1.19584052e-04, 9.58199744e-05,
                                           1.42759436e-05, 1.25136401e-04, 1.31127046e-04, 0.00000000e+00,
                                           0.00000000e+00, -7.15001079e-05, -1.40273305e-05, 1.04227758e-04,
                                           -5.86470087e-05, 9.19737795e-05, 2.03652409e-04, 1.74126384e-04,
                                           1.51715751e-04, 7.38586386e-05, 1.45048572e-04, 3.26555528e-05,
                                           -1.42736426e-05, 8.72509190e-06, -3.97102558e-05, 0.00000000e+00,
                                           0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00])
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
