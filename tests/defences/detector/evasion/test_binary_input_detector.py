# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2023
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

import logging
import pytest
import numpy as np

from art.attacks.evasion.fast_gradient import FastGradientMethod
from art.estimators.classification import PyTorchClassifier, TensorFlowV2Classifier, KerasClassifier
from art.defences.detector.evasion import BinaryInputDetector

from tests.utils import ARTTestException, get_image_classifier_tf, get_image_classifier_kr, get_image_classifier_pt

logger = logging.getLogger(__name__)


@pytest.mark.only_with_platform("keras", "kerastf")
def test_binary_input_detector_keras(art_warning, get_default_mnist_subset):
    from tensorflow.keras import layers, Sequential

    (x_train, _), (x_test, _) = get_default_mnist_subset

    # Keras classifier
    classifier = get_image_classifier_kr(loss_name="categorical_crossentropy", from_logits=True)

    # Generate adversarial samples:
    attacker = FastGradientMethod(classifier, eps=0.1)
    x_train_adv = attacker.generate(x_train)
    x_test_adv = attacker.generate(x_test)

    # Compile training data for detector:
    x_train_detector = np.concatenate((x_train, x_train_adv), axis=0)
    y_train_detector = np.concatenate((np.array([[1, 0]] * len(x_train)), np.array([[0, 1]] * len(x_train))), axis=0)

    # Create a simple CNN for the detector
    model = Sequential()
    model.add(layers.Conv2D(4, kernel_size=(5, 5), activation="relu", input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(2, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    detector_classifier = KerasClassifier(model=model, clip_values=(0, 1), use_logits=False)

    try:
        detector = BinaryInputDetector(detector_classifier)
        detector.fit(x_train_detector, y_train_detector, nb_epochs=2, batch_size=128)

        # Apply detector on clean and adversarial test data:
        _, test_detection = detector.detect(x_test)
        _, test_adv_detection = detector.detect(x_test_adv)

        # Assert there is at least one true positive and negative:
        nb_true_positives = np.sum(test_adv_detection)
        nb_true_negatives = len(test_detection) - np.sum(test_detection)
        assert nb_true_positives > 0
        assert nb_true_negatives > 0
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("tensorflow2")
def test_binary_input_detector_tensorflow(art_warning, get_default_mnist_subset):
    import tensorflow as tf
    from tensorflow.keras import layers, Sequential

    (x_train, _), (x_test, _) = get_default_mnist_subset

    # TensorFlow classifier
    classifier, _ = get_image_classifier_tf(from_logits=True)

    # Generate adversarial samples:
    attacker = FastGradientMethod(classifier, eps=0.1)
    x_train_adv = attacker.generate(x_train)
    x_test_adv = attacker.generate(x_test)

    # Compile training data for detector:
    x_train_detector = np.concatenate((x_train, x_train_adv), axis=0)
    y_train_detector = np.concatenate((np.array([[1, 0]] * len(x_train)), np.array([[0, 1]] * len(x_train))), axis=0)

    # Create a simple CNN for the detector
    model = Sequential()
    model.add(layers.Conv2D(4, kernel_size=(5, 5), activation="relu", input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(2, activation="softmax"))

    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    def train_step(model, images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    detector_classifier = TensorFlowV2Classifier(
        model, nb_classes=2, input_shape=(28, 28, 1), loss_object=loss_object, train_step=train_step
    )

    try:
        detector = BinaryInputDetector(detector_classifier)
        detector.fit(x_train_detector, y_train_detector, nb_epochs=2, batch_size=128)

        # Apply detector on clean and adversarial test data:
        _, test_detection = detector.detect(x_test)
        _, test_adv_detection = detector.detect(x_test_adv)

        # Assert there is at least one true positive and negative:
        nb_true_positives = np.sum(test_adv_detection)
        nb_true_negatives = len(test_detection) - np.sum(test_detection)
        assert nb_true_positives > 0
        assert nb_true_negatives > 0
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_binary_input_detector_pytorch(art_warning, get_default_mnist_subset):
    import torch

    (x_train, _), (x_test, _) = get_default_mnist_subset

    # PyTorch classifier
    classifier = get_image_classifier_pt(from_logits=True)

    # Generate adversarial samples:
    attacker = FastGradientMethod(classifier, eps=0.1)
    x_train_adv = attacker.generate(x_train)
    x_test_adv = attacker.generate(x_test)

    # Compile training data for detector:
    x_train_detector = np.concatenate((x_train, x_train_adv), axis=0)
    y_train_detector = np.concatenate((np.array([[1, 0]] * len(x_train)), np.array([[0, 1]] * len(x_train))), axis=0)

    # Create a simple CNN for the detector
    model = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5),
        torch.nn.MaxPool2d(2, 2),
        torch.nn.Flatten(),
        torch.nn.Linear(576, 2),
    )
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    detector_classifier = PyTorchClassifier(
        model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(1, 28, 28),
        nb_classes=2,
    )

    try:
        detector = BinaryInputDetector(detector_classifier)
        detector.fit(x_train_detector, y_train_detector, nb_epochs=2, batch_size=128)

        # Apply detector on clean and adversarial test data:
        _, test_detection = detector.detect(x_test)
        _, test_adv_detection = detector.detect(x_test_adv)

        # Assert there is at least one true positive and negative:
        nb_true_positives = np.sum(test_adv_detection)
        nb_true_negatives = len(test_detection) - np.sum(test_detection)
        assert nb_true_positives > 0
        assert nb_true_negatives > 0
    except ARTTestException as e:
        art_warning(e)
