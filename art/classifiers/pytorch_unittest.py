from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from art.classifiers.pytorch import PyTorchImageClassifier, PyTorchTextClassifier
from art.utils import load_mnist, load_imdb
from art.utils import to_categorical


NB_TRAIN = 1000
NB_TEST = 5


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
        n = x.size()
        result = x.view(n[0], -1)

        return result


class TestPyTorchImageClassifier(unittest.TestCase):
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
        classifier = PyTorchImageClassifier((0, 1), model, loss_fn, optimizer, (1, 28, 28), 10)
        classifier.fit(x_train, y_train, batch_size=100, nb_epochs=2)
        cls.seq_classifier = classifier

        # Define the network
        model = Model()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        classifier2 = PyTorchImageClassifier((0, 1), model, loss_fn, optimizer, (1, 28, 28), 10)
        classifier2.fit(x_train, y_train, batch_size=100, nb_epochs=2)
        cls.module_classifier = classifier2

    def test_fit_predict(self):
        # Get MNIST
        (_, _), (x_test, y_test) = self.mnist

        # Test predict
        preds = self.module_classifier.predict(x_test)
        acc = np.sum(np.argmax(preds, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
        print("\nAccuracy: %.2f%%" % (acc * 100))
        self.assertGreater(acc, 0.1)

    def test_nb_classes(self):
        ptc = self.module_classifier
        self.assertTrue(ptc.nb_classes == 10)

    def test_input_shape(self):
        ptc = self.module_classifier
        self.assertTrue(np.array(ptc.input_shape == (1, 28, 28)).all())

    def test_class_gradient(self):
        # Get MNIST
        (_, _), (x_test, y_test) = self.mnist

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
        (_, _), (x_test, y_test) = self.mnist

        # Test and get layers
        ptc = self.seq_classifier

        layer_names = self.seq_classifier.layer_names
        self.assertTrue(layer_names == ['0_Conv2d(1, 16, kernel_size=(5, 5), stride=(1, 1))', '1_ReLU()',
                                        '2_MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)',
                                        '3_Flatten()', '4_Linear(in_features=2304, out_features=10, bias=True)'])

        for i, name in enumerate(layer_names):
            act_i = ptc.get_activations(x_test, i)
            act_name = ptc.get_activations(x_test, name)
            self.assertTrue(np.sum(act_name-act_i) == 0)

        self.assertTrue(ptc.get_activations(x_test, 0).shape == (NB_TEST, 16, 24, 24))
        self.assertTrue(ptc.get_activations(x_test, 1).shape == (NB_TEST, 16, 24, 24))
        self.assertTrue(ptc.get_activations(x_test, 2).shape == (NB_TEST, 16, 12, 12))
        self.assertTrue(ptc.get_activations(x_test, 3).shape == (NB_TEST, 2304))
        self.assertTrue(ptc.get_activations(x_test, 4).shape == (NB_TEST, 10))


##################################################################################################
##Test Text
##################################################################################################
# class TextModel(nn.Module):
#     def __init__(self):
#         super(TextModel, self).__init__()
#
#         self.embeddings = nn.Embedding(1000, 32)
#         self.linear1 = nn.Linear(16000, 8)
#         self.linear2 = nn.Linear(8, 2)
#
#     def forward(self, inputs):
#         embeds = self.embeddings(inputs).view((-1, 16000))
#         out = F.relu(self.linear1(embeds))
#         out = self.linear2(out)
#
#         return out


class TestTFTextClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load IMDB
        (x_train, y_train), (x_test, y_test), ids = load_imdb(nb_words=1000, max_length=500)
        x_train = x_train.astype(np.int64)
        x_test = x_test.astype(np.int64)
        ids = list(ids.values())
        ids = [0] + [id for id in ids if id < 1000]

        x_train, y_train, x_test, y_test = x_train[:NB_TRAIN], y_train[:NB_TRAIN], x_test[:NB_TEST], y_test[:NB_TEST]
        cls.imdb = (x_train, y_train), (x_test, y_test)
        cls.word_ids = ids

        # Define the network
        model = nn.Sequential(nn.Embedding(1000, 32, sparse=False), Flatten(), nn.Linear(16000, 8), nn.Linear(8, 2))
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # Create classifier
        cls.classifier = PyTorchTextClassifier(model, 0, ids, loss_fn, optimizer, 2)

    def test_fit_predict(self):
        (x_train, y_train), (x_test, y_test) = self.imdb
        y_train = to_categorical(y_train, nb_classes=2)
        y_test = to_categorical(y_test, nb_classes=2)

        self.classifier.fit(x_train, y_train, nb_epochs=1, batch_size=10)
        acc = np.sum(np.argmax(self.classifier.predict(x_test), axis=1) == np.argmax(y_test, axis=1)) / x_test.shape[0]
        print("\nAccuracy: %.2f%%" % (acc * 100))

        self.assertTrue(acc >= 0)

    def test_nb_classes(self):
        # Start to test
        self.assertTrue(self.classifier.nb_classes == 2)

    def test_class_gradient(self):
        # Get IMDB
        (_, _), (x_test, y_test) = self.imdb
        y_test = to_categorical(y_test, nb_classes=2)

        # Test all gradients label = None
        grads = self.classifier.class_gradient(x_test)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 2, 500, 32)).all())
        self.assertTrue(np.sum(grads) != 0)

        # Test 1 gradient label = 5
        grads = self.classifier.class_gradient(x_test, label=1)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 1, 500, 32)).all())
        self.assertTrue(np.sum(grads) != 0)

        # Test a set of gradients label = array
        label = np.random.randint(2, size=NB_TEST)
        grads = self.classifier.class_gradient(x_test, label=label)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 1, 500, 32)).all())
        self.assertTrue(np.sum(grads) != 0)

    def test_loss_gradient(self):
        # Get IMDB
        (_, _), (x_test, y_test) = self.imdb
        y_test = to_categorical(y_test, nb_classes=2)

        # Test gradient
        grads = self.classifier.loss_gradient(x_test, y_test)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 500, 32)).all())
        self.assertTrue(np.sum(grads) != 0)

    def test_layers(self):
        # Get IMDB
        (_, _), (x_test, y_test) = self.imdb
        y_test = to_categorical(y_test, nb_classes=2)

        # Test and get layers
        layer_names = self.classifier.layer_names
        print(layer_names)
        self.assertTrue(layer_names == ['0_Embedding(1000, 32)', '1_Flatten()',
                                        '2_Linear(in_features=16000, out_features=8, bias=True)',
                                        '3_Linear(in_features=8, out_features=2, bias=True)'])

        for i, name in enumerate(layer_names):
            act_i = self.classifier.get_activations(x_test, i)
            act_name = self.classifier.get_activations(x_test, name)
            self.assertAlmostEqual(np.sum(act_name - act_i), 0)

        print(self.classifier.get_activations(x_test, 0).shape)
        print(self.classifier.get_activations(x_test, 1).shape)
        print(self.classifier.get_activations(x_test, 2).shape)
        print(self.classifier.get_activations(x_test, 3).shape)
        self.assertTrue(self.classifier.get_activations(x_test, 0).shape == (NB_TEST, 500, 32))
        self.assertTrue(self.classifier.get_activations(x_test, 1).shape == (NB_TEST, 16000))
        self.assertTrue(self.classifier.get_activations(x_test, 2).shape == (NB_TEST, 8))
        self.assertTrue(self.classifier.get_activations(x_test, 3).shape == (NB_TEST, 2))

    def test_embedding(self):
        # Get IMDB
        (x_train, y_train), (x_test, y_test) = self.imdb
        y_train = to_categorical(y_train, nb_classes=2)

        # Test to embedding
        x_emb = self.classifier.to_embedding(x_test)
        self.assertTrue(x_emb.shape == (NB_TEST, 500, 32))

        # Test predict_from_embedding
        acc = np.sum(np.argmax(self.classifier.predict_from_embedding(x_emb), axis=1) == y_test) / x_test.shape[0]
        print('\nAccuracy: %.2f%%' % (acc * 100))
        self.assertTrue(acc >= 0)

        # Test to id
        x_id = self.classifier.to_id(x_emb)
        print(x_id, x_test)
        self.assertTrue((x_id == x_test).all())


if __name__ == '__main__':
    unittest.main()







