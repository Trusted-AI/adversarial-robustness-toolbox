from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import os
import shutil
import unittest
import logging
import keras
import keras.backend as k
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

from art.classifiers.keras import KerasClassifier
from art.detection.features import MeanClassDist, SaliencyMap, AttentionMap
from art.utils import load_mnist, master_seed

logger = logging.getLogger('testLogger')

BATCH_SIZE, NB_TRAIN, NB_TEST = 100, 1000, 10


class TestFeatures(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dir_path = './tests'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # Initialize a tf session
        session = tf.Session()
        k.set_session(session)

        # Get MNIST
        nb_train = 1000
        (x_train, y_train), (_, _), _, _ = load_mnist()
        x_train, y_train = x_train[:nb_train], y_train[:nb_train]

        input_shape = x_train.shape[1:]
        nb_classes = 10

        # Create simple CNN
        cls.model = Sequential()
        cls.model.add(Conv2D(4, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
        cls.model.add(MaxPooling2D(pool_size=(2, 2)))
        cls.model.add(Flatten())
        cls.model.add(Dense(nb_classes, activation='softmax'))
        cls.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.01),
                      metrics=['accuracy'])
        cls.model.fit(x_train, y_train, nb_epoch=5, batch_size=128)


    def setUp(self):
        # Set master seed
        master_seed(1234)

    def test_saliency_map(self):
        """
        Testing feature workflow for SaliencyMap
        """

        # Get MNIST
        nb_train, nb_test = 1000, 10
        (x_train, y_train), (_, _), _, _ = load_mnist()
        x_train, y_train = x_train[:nb_train], y_train[:nb_train]
        nb_classes = 10

        # compute the class gradients using Keras
        model = self.model
        grad_f = [k.function([model.layers[0].input], k.gradients([model.layers[-1].output[:, i]],
                                                                 [model.layers[0].input])) for i in range(nb_classes)]
        fv_keras = np.max(np.abs(np.asarray([grad_f[i]([x_train])[0] for i in range(nb_classes)])), axis=0)

        # compute the class gradients using ART
        classifier = KerasClassifier((0, 1), model, use_logits=False)
        df = SaliencyMap(classifier=classifier)
        fv_art = df.extract(x_train)
        self.assertTrue(np.all(fv_keras == fv_art))

    def test_mean_class_dist_fv(self):
        """
        Testing feature workflow for Meanclassdist
        :return:
        """

        # Get MNIST
        nb_train, nb_test = 1000, 10
        (x_train, y_train), (_, _), _, _ = load_mnist()
        x_train, y_train = x_train[:nb_train], y_train[:nb_train]

        nb_classes = 10
        layer_id = 2

        # compute the features using Keras
        model = self.model
        f = k.function([model.layers[0].input], [model.layers[layer_id].output])

        # get the per class layer output for the training set
        layer_output = f([x_train])[0]
        layer_output = layer_output.reshape(layer_output.shape[0], -1)
        layer_output_per_class_keras = [layer_output[np.where(np.argmax(y_train, axis=1) == c)[0]]
                                        for c in range(nb_classes)]

        dists_ = []
        norms2_x = np.sum(layer_output ** 2, 1)[:, None]

        for c in range(nb_classes):

            norms2_y = np.sum(layer_output_per_class_keras[c] ** 2, 1)[None, :]
            pw_dists = norms2_x - 2 * np.matmul(layer_output, layer_output_per_class_keras[c].T) + norms2_y
            dists_.append(np.mean(pw_dists, axis=1))

        fv_keras = np.stack(dists_).T

        # compute the features using classifier from ART
        classifier = KerasClassifier((0, 1), model, use_logits=False)
        df = MeanClassDist(classifier=classifier, x=x_train, y=y_train, layer=2)
        fv_art = df.extract(x=x_train)

        self.assertTrue(np.all(fv_keras == fv_art))

    def test_attention_map(self):
        """
        Testing feature workflow for AttentionMap
        :return:
        """

        # Get MNIST
        nb_train, nb_test = 2, 2
        (x_train, y_train), (_, _), _, _ = load_mnist()
        x_train, y_train = x_train[:nb_train], y_train[:nb_train]

        # compute the attention map using only Keras
        model = self.model

        strides = 4
        window_width = 8

        predictions = []
        for image in x_train:
            images = []
            for i in range(0, image.shape[0], strides):
                for j in range(0, image.shape[1], strides):
                    img = np.copy(image)
                    start_x = np.maximum(0, i - window_width + 1)
                    end_x = np.minimum(image.shape[0], i + window_width)
                    start_y = np.maximum(0, j - window_width + 1)
                    end_y = np.minimum(image.shape[1], j + window_width)
                    img[start_x:end_x, start_y:end_y, :] = 0.5
                    images.append(img)
            predictions.append(model.predict(np.array(images)))

        fv_keras = np.array(predictions).reshape((x_train.shape[0], np.arange(0, image.shape[0], strides).shape[0],
                                              np.arange(0, image.shape[1], strides).shape[0], -1))

        # compute the class gradients using ART
        classifier = KerasClassifier((0, 1), model, use_logits=False)
        df = AttentionMap(classifier=classifier, window_width=window_width, strides=strides)
        fv_art = df.extract(x_train)
        self.assertTrue(np.all(np.isclose(fv_keras, fv_art)))
