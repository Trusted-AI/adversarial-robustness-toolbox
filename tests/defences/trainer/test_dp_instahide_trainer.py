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
from __future__ import absolute_import, division, print_function, unicode_literals

import pytest
import logging

from art.defences.preprocessor import Mixup, Cutout
from art.defences.trainer import DPInstaHideTrainer
from art.estimators.classification import PyTorchClassifier, TensorFlowV2Classifier, KerasClassifier
from art.utils import load_dataset
from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.fixture()
def mnist_data():
    """
    Get the first 100 samples of the MNIST train set with channels last format

    :return: First 100 sample/label pairs of the MNIST train dataset.
    """
    nb_samples = 100

    (x_train, y_train), (_, _), _, _ = load_dataset("mnist")
    x_train, y_train = x_train[:nb_samples], y_train[:nb_samples]
    return x_train, y_train


@pytest.fixture()
def get_classifier(framework):
    def _get_classifier():
        if framework == "pytorch":
            import torch

            class Model(torch.nn.Module):
                def __init__(self):
                    super(Model, self).__init__()
                    self.conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=7)
                    self.relu = torch.nn.ReLU()
                    self.pool = torch.nn.MaxPool2d(4, 4)
                    self.fc = torch.nn.Linear(25, 10)

                def forward(self, x):
                    x = torch.permute(x, (0, 3, 1, 2)).float()
                    x = self.conv(x)
                    x = self.relu(x)
                    x = self.pool(x)
                    x = torch.flatten(x, 1)
                    x = self.fc(x)
                    return x

            model = Model()
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            classifier = PyTorchClassifier(
                model,
                loss=criterion,
                optimizer=optimizer,
                input_shape=(28, 28, 1),
                nb_classes=10,
            )

        elif framework == "tensorflow2":
            import tensorflow as tf
            from tensorflow.keras import layers, Sequential

            model = Sequential()
            model.add(layers.Conv2D(1, kernel_size=(7, 7), activation="relu", input_shape=(28, 28, 1)))
            model.add(layers.MaxPooling2D(pool_size=(4, 4)))
            model.add(layers.Flatten())
            model.add(layers.Dense(10))

            loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

            def train_step(model, images, labels):
                with tf.GradientTape() as tape:
                    predictions = model(images, training=True)
                    loss = loss_object(labels, predictions)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            classifier = TensorFlowV2Classifier(
                model, nb_classes=10, input_shape=(28, 28, 1), loss_object=loss_object, train_step=train_step
            )

        elif framework in ("keras", "kerastf"):
            import tensorflow as tf
            from tensorflow.keras import layers, Sequential

            if tf.__version__[0] == "2":
                tf.compat.v1.disable_eager_execution()

            model = Sequential()
            model.add(layers.Conv2D(1, kernel_size=(7, 7), activation="relu", input_shape=(28, 28, 1)))
            model.add(layers.MaxPooling2D(pool_size=(4, 4)))
            model.add(layers.Flatten())
            model.add(layers.Dense(10))
            model.compile(optimizer="adam", loss="categorical_crossentropy")
            classifier = KerasClassifier(model, clip_values=(0, 1), use_logits=True)

        else:
            classifier = None

        return classifier

    return _get_classifier


@pytest.mark.only_with_platform("pytorch", "tensorflow2", "keras", "kerastf")
@pytest.mark.parametrize("noise", ["gaussian", "laplacian", "exponential"])
def test_dp_instahide_single_aug(art_warning, get_classifier, mnist_data, noise):
    classifier = get_classifier()
    x, y = mnist_data
    cutout = Cutout(length=8, channels_first=False)

    try:
        trainer = DPInstaHideTrainer(classifier, augmentations=cutout, noise=noise, loc=0, scale=0.1)
        trainer.fit(x, y, nb_epochs=1)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch", "tensorflow2", "keras", "kerastf")
@pytest.mark.parametrize("noise", ["gaussian", "laplacian", "exponential"])
def test_dp_instahide_multiple_aug(art_warning, get_classifier, mnist_data, noise):
    classifier = get_classifier()
    x, y = mnist_data
    mixup = Mixup(num_classes=10)
    cutout = Cutout(length=8, channels_first=False)

    try:
        trainer = DPInstaHideTrainer(classifier, augmentations=[mixup, cutout], noise=noise, loc=0, scale=0.1)
        trainer.fit(x, y, nb_epochs=1)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch", "tensorflow2", "keras", "kerastf")
@pytest.mark.parametrize("noise", ["gaussian", "laplacian", "exponential"])
def test_dp_instahide_generator(art_warning, get_classifier, mnist_data, noise):
    from art.data_generators import NumpyDataGenerator

    classifier = get_classifier()
    x, y = mnist_data
    generator = NumpyDataGenerator(x, y, batch_size=len(x))
    cutout = Cutout(length=8, channels_first=False)

    try:
        trainer = DPInstaHideTrainer(classifier, augmentations=cutout, noise=noise, loc=0, scale=0.1)
        trainer.fit_generator(generator, nb_epochs=1)
    except ARTTestException as e:
        art_warning(e)
