# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
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
import os
import logging
import numpy as np
import pytest
import pickle
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from art.data_generators import PyTorchDataGenerator
from art.estimators.classification.pytorch import PyTorchClassifier

from tests.utils import ExpectedValue

from tests.classifiersFrameworks.utils import (
    backend_test_fit_generator,
    backend_test_loss_gradient,
    backend_test_class_gradient,
    backend_test_layers,
    backend_test_repr,
)

logger = logging.getLogger(__name__)


@pytest.mark.only_with_platform("pytorch")
def test_device():
    class Flatten(nn.Module):
        def forward(self, x):
            n, _, _, _ = x.size()
            result = x.view(n, -1)

            return result

    # Define the network
    model = nn.Sequential(nn.Conv2d(1, 2, 5), nn.ReLU(), nn.MaxPool2d(2, 2), Flatten(), nn.Linear(288, 10))

    # Define a loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # First test cpu
    classifier_cpu = PyTorchClassifier(
        model=model,
        clip_values=(0, 1),
        loss=loss_fn,
        optimizer=optimizer,
        input_shape=(1, 28, 28),
        nb_classes=10,
        device_type="cpu",
    )

    assert classifier_cpu._device == torch.device("cpu")
    assert classifier_cpu._device != torch.device("cuda")

    # Then test gpu
    if torch.cuda.device_count() >= 2:
        with torch.cuda.device(0):
            classifier_gpu0 = PyTorchClassifier(
                model=model,
                clip_values=(0, 1),
                loss=loss_fn,
                optimizer=optimizer,
                input_shape=(1, 28, 28),
                nb_classes=10,
            )
            assert classifier_gpu0._device == torch.device("cuda:0")
            assert classifier_gpu0._device != torch.device("cuda:1")

        with torch.cuda.device(1):
            classifier_gpu1 = PyTorchClassifier(
                model=model,
                clip_values=(0, 1),
                loss=loss_fn,
                optimizer=optimizer,
                input_shape=(1, 28, 28),
                nb_classes=10,
            )
            assert classifier_gpu1._device == torch.device("cuda:1")
            assert classifier_gpu1._device != torch.device("cuda:0")


@pytest.mark.only_with_platform("pytorch")
def test_pickle(get_default_mnist_subset, get_image_classifier_list):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_default_mnist_subset

    classifier, _ = get_image_classifier_list(one_classifier=True)

    from art.config import ART_DATA_PATH
    full_path = os.path.join(ART_DATA_PATH, "my_classifier")
    folder = os.path.split(full_path)[0]
    if not os.path.exists(folder):
        os.makedirs(folder)

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

    # Define the network
    model = Model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    classifier_2 = PyTorchClassifier(
        model=model, clip_values=(0, 1), loss=loss_fn, optimizer=optimizer, input_shape=(1, 28, 28), nb_classes=10
    )
    classifier_2.fit(x_train_mnist, y_train_mnist, batch_size=100, nb_epochs=1)
    module_classifier = classifier_2

    classifier = module_classifier

    # TODO doesn't seem like the issue is with the model since here the same model doesn't work either
    pickle.dump(classifier, open(full_path, "wb"))

    # Unpickle:
    with open(full_path, "rb") as f:
        loaded = pickle.load(f)
        np.testing.assert_equal(classifier._clip_values, loaded._clip_values)
        assert classifier._channel_index == loaded._channel_index
        assert set(classifier.__dict__.keys()) == set(loaded.__dict__.keys())

    # Test predict
    predictions_1 = classifier.predict(x_test_mnist)
    accuracy_1 = np.sum(np.argmax(predictions_1, axis=1) == np.argmax(y_test_mnist, axis=1)) / y_test_mnist.shape[0]
    predictions_2 = loaded.predict(x_test_mnist)
    accuracy_2 = np.sum(np.argmax(predictions_2, axis=1) == np.argmax(y_test_mnist, axis=1)) / y_test_mnist.shape[0]
    assert accuracy_1 == accuracy_2


@pytest.mark.only_with_platform("pytorch")
def test_save(get_image_classifier_list):
    classifier, _ = get_image_classifier_list(one_classifier=True)
    t_file = tempfile.NamedTemporaryFile()
    full_path = t_file.name
    t_file.close()
    base_name = os.path.basename(full_path)
    dir_name = os.path.dirname(full_path)
    classifier.save(base_name, path=dir_name)
    assert os.path.exists(full_path + ".optimizer")
    assert os.path.exists(full_path + ".model")
    os.remove(full_path + ".optimizer")
    os.remove(full_path + ".model")


@pytest.mark.only_with_platform("pytorch")
def test_set_learning(get_image_classifier_list):
    classifier, _ = get_image_classifier_list(one_classifier=True)
    assert classifier._model.training
    classifier.set_learning_phase(False)
    assert classifier._model.training is False
    classifier.set_learning_phase(True)
    assert classifier._model.training
    assert classifier.learning_phase


# TODO refactor this with the test_layers
# @pytest.mark.only_with_platform("pytorch")
# def test_repr(self):
#
#     repr_ = repr(self.module_classifier)
#     self.assertIn("art.estimators.classification.pytorch.PyTorchClassifier", repr_)
#     self.assertIn("input_shape=(1, 28, 28), nb_classes=10, channel_index=1", repr_)
#     self.assertIn("clip_values=array([0., 1.], dtype=float32)", repr_)
#     self.assertIn("defences=None, preprocessing=(0, 1)", repr_)

@pytest.mark.only_with_platform("pytorch")
def test_layers(get_image_classifier_list, get_default_mnist_subset):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_default_mnist_subset

    # classifier, _ = get_image_classifier_list(one_classifier=True)

    class Flatten(nn.Module):
        def forward(self, x):
            n, _, _, _ = x.size()
            result = x.view(n, -1)

            return result

    def create_model():
        # Define the network
        model = nn.Sequential(nn.Conv2d(1, 2, 5), nn.ReLU(), nn.MaxPool2d(2, 2), Flatten(), nn.Linear(288, 10))

        # Define a loss function and optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        classifier = PyTorchClassifier(
            model=model, clip_values=(0, 1), loss=loss_fn, optimizer=optimizer, input_shape=(1, 28, 28), nb_classes=10
        )
        classifier.fit(x_train_mnist, y_train_mnist, batch_size=100, nb_epochs=1)
        return classifier

    # TODO this should be using the default get_image_classifier_list
    ptc = create_model()
    # ptc = self.seq_classifier
    layer_names = ptc.layer_names
    backend_test_repr(ptc, [
        "0_Conv2d(1, 2, kernel_size=(5, 5), stride=(1, 1))",
        "1_ReLU()",
        "2_MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)",
        "3_Flatten()",
        "4_Linear(in_features=288, out_features=10, bias=True)",
    ])

    #
    # self.assertEqual(
    #     layer_names,
    #     [
    #         "0_Conv2d(1, 2, kernel_size=(5, 5), stride=(1, 1))",
    #         "1_ReLU()",
    #         "2_MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)",
    #         "3_Flatten()",
    #         "4_Linear(in_features=288, out_features=10, bias=True)",
    #     ],
    # )

    for i, name in enumerate(layer_names):
        activation_i = ptc.get_activations(x_test_mnist, i, batch_size=5)
        activation_name = ptc.get_activations(x_test_mnist, name, batch_size=5)
        np.testing.assert_array_equal(activation_name, activation_i)

    assert ptc.get_activations(x_test_mnist, 0, batch_size=5).shape == (100, 2, 24, 24)
    assert ptc.get_activations(x_test_mnist, 1, batch_size=5).shape == (100, 2, 24, 24)
    assert ptc.get_activations(x_test_mnist, 2, batch_size=5).shape == (100, 2, 12, 12)
    assert ptc.get_activations(x_test_mnist, 3, batch_size=5).shape == (100, 288)
    assert ptc.get_activations(x_test_mnist, 4, batch_size=5).shape == (100, 10)


@pytest.mark.only_with_platform("pytorch")
def test_fit_predict(get_image_classifier_list, get_default_mnist_subset):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_default_mnist_subset
    classifier, _ = get_image_classifier_list(one_classifier=True)
    predictions = classifier.predict(x_test_mnist)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test_mnist, axis=1)) / x_test_mnist.shape[0]
    logger.info("Accuracy after fitting: %.2f%%", (accuracy * 100))
    assert accuracy == 0.32


@pytest.mark.only_with_platform("pytorch")
def test_fit_image_generator(get_image_classifier_list, image_data_generator, get_default_mnist_subset):
    classifier, _ = get_image_classifier_list(one_classifier=True)

    expected_values = {"pre_fit_accuracy": ExpectedValue(0.32, 0.06), "post_fit_accuracy": ExpectedValue(0.73, 0.06)}

    data_gen = image_data_generator()
    backend_test_fit_generator(expected_values, classifier, data_gen, get_default_mnist_subset, nb_epochs=2)


@pytest.mark.only_with_platform("pytorch")
def test_class_gradient(get_image_classifier_list, get_default_mnist_subset, framework):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_default_mnist_subset

    classifier_logits, _ = get_image_classifier_list(one_classifier=True, from_logits=True)

    expected_gradients_1_all_labels = np.asarray(
        [
            -0.00367321,
            -0.0002892,
            0.00037825,
            -0.00053344,
            0.00192121,
            0.00112047,
            0.0023135,
            0.0,
            0.0,
            -0.00391743,
            -0.0002264,
            0.00238103,
            -0.00073711,
            0.00270405,
            0.00389043,
            0.00440818,
            -0.00412769,
            -0.00441795,
            0.00081916,
            -0.00091284,
            0.00119645,
            -0.00849089,
            0.00547925,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    expected_gradients_2_all_labels = np.asarray(
        [
            -1.0557442e-03,
            -1.0079540e-03,
            -7.7426381e-04,
            1.7387437e-03,
            2.1773505e-03,
            5.0880131e-05,
            1.6497375e-03,
            2.6113102e-03,
            6.0904315e-03,
            4.1080985e-04,
            2.5268074e-03,
            -3.6661496e-04,
            -3.0568994e-03,
            -1.1665225e-03,
            3.8904310e-03,
            3.1726388e-04,
            1.3203262e-03,
            -1.1720933e-04,
            -1.4315107e-03,
            -4.7676827e-04,
            9.7251305e-04,
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
        ]
    )

    expected_gradients_1_label5 = np.asarray(
        [
            -0.00367321,
            -0.0002892,
            0.00037825,
            -0.00053344,
            0.00192121,
            0.00112047,
            0.0023135,
            0.0,
            0.0,
            -0.00391743,
            -0.0002264,
            0.00238103,
            -0.00073711,
            0.00270405,
            0.00389043,
            0.00440818,
            -0.00412769,
            -0.00441795,
            0.00081916,
            -0.00091284,
            0.00119645,
            -0.00849089,
            0.00547925,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    expected_gradients_2_label5 = np.asarray(
        [
            -1.0557442e-03,
            -1.0079540e-03,
            -7.7426381e-04,
            1.7387437e-03,
            2.1773505e-03,
            5.0880131e-05,
            1.6497375e-03,
            2.6113102e-03,
            6.0904315e-03,
            4.1080985e-04,
            2.5268074e-03,
            -3.6661496e-04,
            -3.0568994e-03,
            -1.1665225e-03,
            3.8904310e-03,
            3.1726388e-04,
            1.3203262e-03,
            -1.1720933e-04,
            -1.4315107e-03,
            -4.7676827e-04,
            9.7251305e-04,
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
        ]
    )

    expected_gradients_1_label_array = np.asarray(
        [
            -0.00195835,
            -0.00134457,
            -0.00307221,
            -0.00340564,
            0.00175022,
            -0.00239714,
            -0.00122619,
            0.0,
            0.0,
            -0.00520899,
            -0.00046105,
            0.00414874,
            -0.00171095,
            0.00429184,
            0.0075138,
            0.00792443,
            0.0019566,
            0.00035517,
            0.00504575,
            -0.00037397,
            0.00022343,
            -0.00530035,
            0.0020528,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    expected_gradients_2_label_array = np.asarray(
        [
            5.0867130e-03,
            4.8564533e-03,
            6.1040395e-03,
            8.6531248e-03,
            -6.0958802e-03,
            -1.4114541e-02,
            -7.1085966e-04,
            -5.0330797e-04,
            1.2943064e-02,
            8.2416134e-03,
            -1.9859453e-04,
            -9.8110031e-05,
            -3.8902226e-03,
            -1.2945874e-03,
            7.5138002e-03,
            1.7720887e-03,
            3.1399354e-04,
            2.3657191e-04,
            -3.0891625e-03,
            -1.0211228e-03,
            2.0828887e-03,
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
            0.0000000e00,
        ]
    )

    decimal_precision = 4

    expected_values = {
        "expected_gradients_1_all_labels": ExpectedValue(
            expected_gradients_1_all_labels,
            decimal_precision,
        ),
        "expected_gradients_2_all_labels": ExpectedValue(
            expected_gradients_2_all_labels,
            decimal_precision,
        ),
        "expected_gradients_1_label5": ExpectedValue(
            expected_gradients_1_label5,
            decimal_precision,
        ),
        "expected_gradients_2_label5": ExpectedValue(
            expected_gradients_2_label5,
            decimal_precision,
        ),
        "expected_gradients_1_labelArray": ExpectedValue(
            expected_gradients_1_label_array,
            decimal_precision,
        ),
        "expected_gradients_2_labelArray": ExpectedValue(
            expected_gradients_2_label_array,
            decimal_precision,
        ),
    }

    labels = np.random.randint(5, size=x_test_mnist.shape[0])
    backend_test_class_gradient(framework, get_default_mnist_subset, classifier_logits, expected_values, labels)
