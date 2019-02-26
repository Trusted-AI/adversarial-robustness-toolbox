from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from art.classifiers.pytorch import PyTorchClassifier
from art.classifiers.detector_classifier import DetectorClassifier
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


class TestDetectorClassifier(unittest.TestCase):
    """
    This class tests the functionalities of the DetectorClassifier.
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

        # Define the internal classifier
        model = Model()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        classifier = PyTorchClassifier((0, 1), model, loss_fn, optimizer, (1, 28, 28), 10)
        classifier.fit(x_train, y_train, batch_size=100, nb_epochs=2)

        # Define the internal detector
        model = nn.Sequential(nn.Conv2d(1, 16, 5), nn.ReLU(), nn.MaxPool2d(2, 2), Flatten(), nn.Linear(2304, 1))
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        detector = PyTorchClassifier((0, 1), model, loss_fn, optimizer, (1, 28, 28), 1)

        # Define the detector-classifier
        cls.detector_classifier = DetectorClassifier(classifier=classifier, detector=detector)

    def setUp(self):
        # Set master seed
        master_seed(1234)

    def test_predict(self):
        # Get MNIST
        (_, _), (x_test, y_test) = self.mnist

        # Test predict logits
        preds = self.detector_classifier.predict(x=x_test, logits=True)
        self.assertTrue(preds.shape == (NB_TEST, 11))

        # Test predict softmax
        preds = self.detector_classifier.predict(x=x_test, logits=False)
        self.assertTrue(np.sum(preds) == NB_TEST)

    def test_nb_classes(self):
        dc = self.detector_classifier
        self.assertTrue(dc.nb_classes == 11)

    def test_input_shape(self):
        dc = self.detector_classifier
        self.assertTrue(np.array(dc.input_shape == (1, 28, 28)).all())

    def test_class_gradient(self):
        # Get MNIST
        (_, _), (x_test, _) = self.mnist

        # Test all gradients label = None
        ptc = self.module_classifier
        grads = ptc.class_gradient(x_test)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 10, 1, 28, 28)).all())
        self.assertTrue(np.sum(grads) != 0)

        # Test 1 gradient label = 5
        grads = ptc.class_gradient(x_test, label=5)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 1, 1, 28, 28)).all())
        self.assertTrue(np.sum(grads) != 0)

        # Test a set of gradients label = array
        label = np.random.randint(5, size=NB_TEST)
        grads = ptc.class_gradient(x_test, label=label)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 1, 1, 28, 28)).all())
        self.assertTrue(np.sum(grads) != 0)

    def test_class_gradient_target(self):
        # Get MNIST
        (_, _), (x_test, _) = self.mnist

        # Test gradient
        ptc = self.module_classifier
        grads = ptc.class_gradient(x_test, label=3)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 1, 1, 28, 28)).all())
        self.assertTrue(np.sum(grads) != 0)

    def test_loss_gradient(self):
        # Get MNIST
        (_, _), (x_test, y_test) = self.mnist

        # Test gradient
        ptc = self.module_classifier
        grads = ptc.loss_gradient(x_test, y_test)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 1, 28, 28)).all())
        self.assertTrue(np.sum(grads) != 0)

    def test_layers(self):
        # Get MNIST
        (_, _), (x_test, _) = self.mnist

        # Test and get layers
        ptc = self.seq_classifier

        layer_names = self.seq_classifier.layer_names
        self.assertTrue(layer_names == ['0_Conv2d(1, 16, kernel_size=(5, 5), stride=(1, 1))', '1_ReLU()',
                                        '2_MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)',
                                        '3_Flatten()', '4_Linear(in_features=2304, out_features=10, bias=True)'])

        for i, name in enumerate(layer_names):
            act_i = ptc.get_activations(x_test, i, batch_size=5)
            act_name = ptc.get_activations(x_test, name, batch_size=5)
            self.assertTrue(np.sum(act_name-act_i) == 0)

        self.assertTrue(ptc.get_activations(x_test, 0, batch_size=5).shape == (20, 16, 24, 24))
        self.assertTrue(ptc.get_activations(x_test, 1, batch_size=5).shape == (20, 16, 24, 24))
        self.assertTrue(ptc.get_activations(x_test, 2, batch_size=5).shape == (20, 16, 12, 12))
        self.assertTrue(ptc.get_activations(x_test, 3, batch_size=5).shape == (20, 2304))
        self.assertTrue(ptc.get_activations(x_test, 4, batch_size=5).shape == (20, 10))

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
                                       

if __name__ == '__main__':
    unittest.main()
