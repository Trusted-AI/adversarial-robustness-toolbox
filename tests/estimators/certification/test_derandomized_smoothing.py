# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2022
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
import pytest

import numpy as np

from art.utils import load_dataset
from art.estimators.certification.derandomized_smoothing import (
    PyTorchDeRandomizedSmoothing,
    TensorFlowV2DeRandomizedSmoothing,
)
from tests.utils import ARTTestException


@pytest.fixture()
def fix_get_mnist_data():
    """
    Get the first 128 samples of the mnist test set with channels first format

    :return: First 128 sample/label pairs of the MNIST test dataset.
    """
    nb_test = 128

    (_, _), (x_test, y_test), _, _ = load_dataset("mnist")
    x_test = np.squeeze(x_test).astype(np.float32)
    x_test = np.expand_dims(x_test, axis=1)
    y_test = np.argmax(y_test, axis=1)

    x_test, y_test = x_test[:nb_test], y_test[:nb_test]
    return x_test, y_test


@pytest.fixture()
def fix_get_cifar10_data():
    """
    Get the first 128 samples of the cifar10 test set

    :return: First 128 sample/label pairs of the cifar10 test dataset.
    """
    nb_test = 128

    (_, _), (x_test, y_test), _, _ = load_dataset("cifar10")
    y_test = np.argmax(y_test, axis=1)
    x_test, y_test = x_test[:nb_test], y_test[:nb_test]
    x_test = np.transpose(x_test, (0, 3, 1, 2))  # return in channels first format
    return x_test.astype(np.float32), y_test


@pytest.mark.skip_framework("mxnet", "non_dl_frameworks", "tensorflow1", "keras", "kerastf", "tensorflow2")
def test_pytorch_training(art_warning, fix_get_mnist_data, fix_get_cifar10_data):
    """
    Check that the training loop for pytorch does not result in errors
    """
    import torch
    import torch.optim as optim
    import torch.nn as nn

    device = "cuda" if torch.cuda.is_available() else "cpu"

    class SmallMNISTModel(nn.Module):
        def __init__(self):
            super(SmallMNISTModel, self).__init__()

            self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(4, 4), dilation=(1, 1), stride=(2, 2))
            self.max_pool = nn.MaxPool2d(2, stride=2)
            self.fc1 = nn.Linear(in_features=1152, out_features=100)

            self.fc2 = nn.Linear(in_features=100, out_features=10)

            self.relu = nn.ReLU()

        def forward(self, x):
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).to(device)
            x = self.relu(self.conv1(x))
            x = self.max_pool(x)
            x = torch.flatten(x, 1)
            x = self.relu(self.fc1(x))
            return self.fc2(x)

    class SmallCIFARModel(nn.Module):
        def __init__(self):
            super(SmallCIFARModel, self).__init__()

            self.conv1 = nn.Conv2d(in_channels=6, out_channels=32, kernel_size=(4, 4), dilation=(1, 1), stride=(2, 2))
            self.max_pool = nn.MaxPool2d(2, stride=2)
            self.fc1 = nn.Linear(in_features=1568, out_features=100)

            self.fc2 = nn.Linear(in_features=100, out_features=10)

            self.relu = nn.ReLU()

        def forward(self, x):
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).to(device)
            x = self.relu(self.conv1(x))
            x = self.max_pool(x)
            x = torch.flatten(x, 1)
            x = self.relu(self.fc1(x))
            return self.fc2(x)

    for dataset, dataset_name in zip([fix_get_mnist_data, fix_get_cifar10_data], ["mnist", "cifar"]):
        if dataset_name == "mnist":
            ptc = SmallMNISTModel().to(device)
            input_shape = (1, 28, 28)
        else:
            ptc = SmallCIFARModel().to(device)
            input_shape = (3, 32, 32)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(ptc.parameters(), lr=0.01, momentum=0.9)
        try:
            for ablation_type in ["column", "row", "block"]:
                classifier = PyTorchDeRandomizedSmoothing(
                    model=ptc,
                    clip_values=(0, 1),
                    loss=criterion,
                    optimizer=optimizer,
                    input_shape=input_shape,
                    nb_classes=10,
                    ablation_type=ablation_type,
                    ablation_size=5,
                    threshold=0.3,
                    algorithm="levine2020",
                    logits=True,
                )
                classifier.fit(x=dataset[0], y=dataset[1], nb_epochs=1)
        except ARTTestException as e:
            art_warning(e)


@pytest.mark.skip_framework("mxnet", "non_dl_frameworks", "tensorflow1", "keras", "kerastf", "pytorch")
def test_tf2_training(art_warning, fix_get_mnist_data, fix_get_cifar10_data):
    """
    Check that the training loop for tensorflow2 does not result in errors
    """
    import tensorflow as tf

    def build_model(input_shape):
        img_inputs = tf.keras.Input(shape=(input_shape[0], input_shape[1], input_shape[2] * 2))
        x = tf.keras.layers.Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), activation="relu")(img_inputs)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(x)
        # tensorflow uses channels last and we are loading weights from an originally trained pytorch model
        x = tf.transpose(x, (0, 3, 1, 2))
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(100, activation="relu")(x)
        x = tf.keras.layers.Dense(10)(x)
        return tf.keras.Model(inputs=img_inputs, outputs=x)

    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=0.01)

    for dataset, dataset_name in zip([fix_get_mnist_data, fix_get_cifar10_data], ["mnist", "cifar"]):
        if dataset_name == "mnist":
            input_shape = (28, 28, 1)
        else:
            input_shape = (32, 32, 3)
        net = build_model(input_shape=input_shape)

        try:
            for ablation_type in ["column", "row", "block"]:
                ablation_size = 5
                classifier = TensorFlowV2DeRandomizedSmoothing(
                    model=net,
                    clip_values=(0, 1),
                    loss_object=loss_object,
                    optimizer=optimizer,
                    input_shape=input_shape,
                    nb_classes=10,
                    ablation_type=ablation_type,
                    ablation_size=ablation_size,
                    threshold=0.3,
                    logits=True,
                )
                x = np.transpose(np.copy(dataset[0]), (0, 2, 3, 1))  # put channels last
                classifier.fit(x=x, y=dataset[1], nb_epochs=1)
        except ARTTestException as e:
            art_warning(e)


@pytest.mark.skip_framework("mxnet", "non_dl_frameworks", "tensorflow1", "keras", "kerastf", "tensorflow2")
def test_pytorch_mnist_certification(art_warning, fix_get_mnist_data):
    """
    Assert that the correct number of certifications are given for the MNIST dataset
    """
    import torch
    import torch.optim as optim
    import torch.nn as nn

    device = "cuda" if torch.cuda.is_available() else "cpu"

    class SmallMNISTModel(nn.Module):
        def __init__(self):
            super(SmallMNISTModel, self).__init__()

            self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(4, 4), dilation=(1, 1), stride=(2, 2))
            self.max_pool = nn.MaxPool2d(2, stride=2)
            self.fc1 = nn.Linear(in_features=1152, out_features=100)

            self.fc2 = nn.Linear(in_features=100, out_features=10)

            self.relu = nn.ReLU()

        def forward(self, x):
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).to(device)
            x = self.relu(self.conv1(x))
            x = self.max_pool(x)
            x = torch.flatten(x, 1)
            x = self.relu(self.fc1(x))
            return self.fc2(x)

        def load_weights(self):
            fpath = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "../../utils/resources/models/certification/derandomized/"
            )
            self.conv1.weight = nn.Parameter(torch.from_numpy(np.load(fpath + "W_CONV2D1_MNIST.npy")).float())
            self.conv1.bias = nn.Parameter(torch.from_numpy(np.load(fpath + "B_CONV2D1_MNIST.npy")).float())

            self.fc1.weight = nn.Parameter(torch.from_numpy(np.load(fpath + "W_DENSE1_MNIST.npy")).float())
            self.fc1.bias = nn.Parameter(torch.from_numpy(np.load(fpath + "B_DENSE1_MNIST.npy")).float())

            self.fc2.weight = nn.Parameter(torch.from_numpy(np.load(fpath + "W_DENSE2_MNIST.npy")).float())
            self.fc2.bias = nn.Parameter(torch.from_numpy(np.load(fpath + "B_DENSE2_MNIST.npy")).float())

    ptc = SmallMNISTModel()
    ptc.load_weights()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ptc.parameters(), lr=0.01)

    try:
        for ablation_type in ["column", "block"]:
            if ablation_type == "column":
                size_to_certify = 5
                ablation_size = 2
            else:
                """
                the model was trained on column ablations, so make the block task simpler so that a
                degree of certification is obtained.
                """
                size_to_certify = 1
                ablation_size = 5

            classifier = PyTorchDeRandomizedSmoothing(
                model=ptc,
                clip_values=(0, 1),
                loss=criterion,
                optimizer=optimizer,
                input_shape=(1, 28, 28),
                nb_classes=10,
                ablation_type=ablation_type,
                ablation_size=ablation_size,
                threshold=0.3,
                algorithm="levine2020",
                logits=True,
            )

            preds = classifier.predict(np.copy(fix_get_mnist_data[0]))
            cert, cert_and_correct, top_predicted_class_argmax = classifier.ablator.certify(
                preds, label=fix_get_mnist_data[1], size_to_certify=size_to_certify
            )
            if ablation_type == "column":
                assert np.sum(cert.cpu().numpy()) == 52
            else:
                assert np.sum(cert.cpu().numpy()) == 22
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("mxnet", "non_dl_frameworks", "tensorflow1", "keras", "kerastf", "pytorch")
def test_tf2_mnist_certification(art_warning, fix_get_mnist_data):
    """
    Assert that the correct number of certifications are given for the MNIST dataset
    """

    import tensorflow as tf

    def build_model(input_shape):
        img_inputs = tf.keras.Input(shape=(input_shape[0], input_shape[1], input_shape[2] * 2))
        x = tf.keras.layers.Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), activation="relu")(img_inputs)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(x)
        # tensorflow uses channels last and we are loading weights from an originally trained pytorch model
        x = tf.transpose(x, (0, 3, 1, 2))
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(100, activation="relu")(x)
        x = tf.keras.layers.Dense(10)(x)
        return tf.keras.Model(inputs=img_inputs, outputs=x)

    def get_weights():
        fpath = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "../../utils/resources/models/certification/derandomized/"
        )
        weight_names = [
            "W_CONV2D1_MNIST.npy",
            "B_CONV2D1_MNIST.npy",
            "W_DENSE1_MNIST.npy",
            "B_DENSE1_MNIST.npy",
            "W_DENSE2_MNIST.npy",
            "B_DENSE2_MNIST.npy",
        ]
        weight_list = []
        for name in weight_names:
            w = np.load(fpath + name)
            if "W_CONV" in name:
                w = np.transpose(w, (2, 3, 1, 0))
            if "W_DENSE" in name:
                w = np.transpose(w)
            weight_list.append(w)
        return weight_list

    net = build_model(input_shape=(28, 28, 1))
    net.set_weights(get_weights())

    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=0.01)

    try:
        for ablation_type in ["column", "block"]:
            if ablation_type == "column":
                size_to_certify = 5
                ablation_size = 2
            else:
                """
                the model was trained on column ablations, so make the block task simpler so that a
                degree of certification is obtained.
                """
                size_to_certify = 1
                ablation_size = 5

            classifier = TensorFlowV2DeRandomizedSmoothing(
                model=net,
                clip_values=(0, 1),
                loss_object=loss_object,
                optimizer=optimizer,
                input_shape=(28, 28, 1),
                nb_classes=10,
                ablation_type=ablation_type,
                ablation_size=ablation_size,
                threshold=0.3,
                logits=True,
            )

            x = np.copy(fix_get_mnist_data[0])
            x = np.squeeze(x)
            x = np.expand_dims(x, axis=-1)
            preds = classifier.predict(x)
            cert, cert_and_correct, top_predicted_class_argmax = classifier.ablator.certify(
                preds, label=fix_get_mnist_data[1], size_to_certify=size_to_certify
            )

            if ablation_type == "column":
                assert np.sum(cert) == 52
            else:
                assert np.sum(cert) == 22

    except ARTTestException as e:
        art_warning(e)
