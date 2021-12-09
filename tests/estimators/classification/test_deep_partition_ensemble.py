# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2021
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
import os
import unittest

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from art.utils import load_dataset
from art.estimators.classification.deep_partition_ensemble import DeepPartitionEnsemble
from art.estimators.classification import TensorFlowV2Classifier, PyTorchClassifier, KerasClassifier

from tests.utils import master_seed

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
logger = logging.getLogger(__name__)

BATCH_SIZE = 100
NB_TRAIN = 1000
NB_TEST = 10
ENSEMBLE_SIZE = 5


class TestDeepPartitionEnsemble(unittest.TestCase):
    """
    This class tests the deep partition ensemble classifier.
    """

    @classmethod
    def setUpClass(cls):
        # Get MNIST
        (x_train, y_train), (x_test, y_test), _, _ = load_dataset("mnist")
        x_train, y_train = x_train[:NB_TRAIN], y_train[:NB_TRAIN]
        x_test, y_test = x_test[:NB_TEST], y_test[:NB_TEST]
        cls.mnist = (x_train, y_train), (x_test, y_test)

    def setUp(self):
        master_seed(seed=1234)

    def test_1_tf(self):
        """
        Test with a TensorFlow Classifier.
        :return:
        """
        tf_version = list(map(int, tf.__version__.lower().split("+")[0].split(".")))
        if tf_version[0] == 2:

            # Get MNIST
            (x_train, y_train), (x_test, y_test) = self.mnist

            # Create a model from scratch
            from tensorflow.keras import Model
            from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D

            class TensorFlowModel(Model):
                """
                Standard TensorFlow model for unit testing.
                """

                def __init__(self):
                    super(TensorFlowModel, self).__init__()
                    self.conv1 = Conv2D(filters=4, kernel_size=5, activation="relu")
                    self.conv2 = Conv2D(filters=10, kernel_size=5, activation="relu")
                    self.maxpool = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="valid", data_format=None)
                    self.flatten = Flatten()
                    self.dense1 = Dense(100, activation="relu")
                    self.logits = Dense(10, activation="linear")

                def call(self, x):
                    """
                    Call function to evaluate the model.

                    :param x: Input to the model
                    :return: Prediction of the model
                    """
                    x = self.conv1(x)
                    x = self.maxpool(x)
                    x = self.conv2(x)
                    x = self.maxpool(x)
                    x = self.flatten(x)
                    x = self.dense1(x)
                    x = self.logits(x)
                    return x

            optimizer = Adam(learning_rate=0.01)

            def train_step(model, images, labels):
                with tf.GradientTape() as tape:
                    predictions = model(images, training=True)
                    loss = loss_object(labels, predictions)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            model = TensorFlowModel()
            loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

            classifier = TensorFlowV2Classifier(
                model=model,
                loss_object=loss_object,
                train_step=train_step,
                nb_classes=10,
                input_shape=(28, 28, 1),
                clip_values=(0, 1),
            )

            # Initialize DPA Classifier
            dpa = DeepPartitionEnsemble(
                classifiers=classifier,
                ensemble_size=ENSEMBLE_SIZE,
                channels_first=classifier.channels_first,
                clip_values=classifier.clip_values,
                preprocessing_defences=classifier.preprocessing_defences,
                postprocessing_defences=classifier.postprocessing_defences,
                preprocessing=classifier.preprocessing,
            )

            # Check basic functionality of DPA Classifier
            # check predict
            y_test_dpa = dpa.predict(x=x_test)
            self.assertEqual(y_test_dpa.shape, y_test.shape)
            self.assertTrue((np.sum(y_test_dpa, axis=1) <= ENSEMBLE_SIZE * np.ones((NB_TEST,))).all())

            # loss gradient
            grad = dpa.loss_gradient(x=x_test, y=y_test, sampling=True)
            assert grad.shape == (10, 28, 28, 1)

            # fit
            dpa.fit(x=x_train, y=y_train)

    def test_2_pt(self):
        """
        Test with a PyTorch Classifier.
        :return:
        """

        # Get MNIST
        (x_train, y_train), (x_test, y_test) = self.mnist

        x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
        x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)

        # Create a model from scratch
        class PyTorchModel(nn.Module):
            def __init__(self):
                super(PyTorchModel, self).__init__()
                self.conv_1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, stride=1)
                self.conv_2 = nn.Conv2d(in_channels=4, out_channels=10, kernel_size=5, stride=1)
                self.fc_1 = nn.Linear(in_features=4 * 4 * 10, out_features=100)
                self.fc_2 = nn.Linear(in_features=100, out_features=10)

            def forward(self, x):
                x = F.relu(self.conv_1(x))
                x = F.max_pool2d(x, 2, 2)
                x = F.relu(self.conv_2(x))
                x = F.max_pool2d(x, 2, 2)
                x = x.view(-1, 4 * 4 * 10)
                x = F.relu(self.fc_1(x))
                x = self.fc_2(x)
                return x

        # Step 2a: Define the loss function and the optimizer
        model = PyTorchModel()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # Step 3: Create the ART classifier

        classifier = PyTorchClassifier(
            model=model,
            clip_values=(0, 1),
            loss=criterion,
            optimizer=optimizer,
            input_shape=(1, 28, 28),
            nb_classes=10,
        )

        # Initialize DPA Classifier
        dpa = DeepPartitionEnsemble(
            classifiers=classifier,
            ensemble_size=ENSEMBLE_SIZE,
            channels_first=classifier.channels_first,
            clip_values=classifier.clip_values,
            preprocessing_defences=classifier.preprocessing_defences,
            postprocessing_defences=classifier.postprocessing_defences,
            preprocessing=classifier.preprocessing,
        )

        # Check basic functionality of DPA Classifier
        # check predict
        y_test_dpa = dpa.predict(x=x_test)
        self.assertEqual(y_test_dpa.shape, y_test.shape)
        self.assertTrue((np.sum(y_test_dpa, axis=1) <= ENSEMBLE_SIZE * np.ones((NB_TEST,))).all())

        # loss gradient
        grad = dpa.loss_gradient(x=x_test, y=y_test, sampling=True)
        assert grad.shape == (10, 1, 28, 28)

        # fit
        dpa.fit(x=x_train, y=y_train)

    def test_3_kr(self):
        """
        Test with a Keras Classifier.
        :return:
        """
        tf.compat.v1.disable_eager_execution()

        # Get MNIST
        (x_train, y_train), (x_test, y_test) = self.mnist

        # Create a model from scratch
        model = Sequential()
        model.add(Conv2D(filters=4, kernel_size=(5, 5), strides=1, activation="relu", input_shape=(28, 28, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=10, kernel_size=(5, 5), strides=1, activation="relu", input_shape=(23, 23, 4)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(100, activation="relu"))
        model.add(Dense(10, activation="softmax"))

        model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=0.01), metrics=["accuracy"])

        classifier = KerasClassifier(model=model, clip_values=(0, 1), use_logits=False)

        # Initialize DPA Classifier
        dpa = DeepPartitionEnsemble(
            classifiers=classifier,
            ensemble_size=ENSEMBLE_SIZE,
            channels_first=classifier.channels_first,
            clip_values=classifier.clip_values,
            preprocessing_defences=classifier.preprocessing_defences,
            postprocessing_defences=classifier.postprocessing_defences,
            preprocessing=classifier.preprocessing,
        )

        # Check basic functionality of DPA Classifier
        # check predict
        y_test_dpa = dpa.predict(x=x_test)
        self.assertEqual(y_test_dpa.shape, y_test.shape)
        self.assertTrue((np.sum(y_test_dpa, axis=1) <= ENSEMBLE_SIZE * np.ones((NB_TEST,))).all())

        # loss gradient
        grad = dpa.loss_gradient(x=x_test, y=y_test, sampling=True)
        assert grad.shape == (10, 28, 28, 1)

        # fit
        dpa.fit(x=x_train, y=y_train)
