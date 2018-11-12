from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import os
import shutil
import unittest
import keras
import keras.backend as k
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

from art.classifiers.keras import KerasClassifier
from art.detection.features import MeanClassDist, SaliencyMap, AttentionMap
from art.utils import load_mnist


BATCH_SIZE, NB_TRAIN, NB_TEST = 100, 1000, 10

class TestFeatures(unittest.TestCase):

    def setUp(self):
        dir_path = './tests'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # Initialize a tf session
        session = tf.Session()
        k.set_session(session)

        # Get MNIST
        NB_TRAIN = 1000
        (x_train, y_train), (_,_), _, _ = load_mnist()
        x_train, y_train = x_train[:NB_TRAIN], y_train[:NB_TRAIN]

        input_shape = x_train.shape[1:]
        nb_classes = 10

        # Create simple CNN
        model = Sequential()
        model.add(Conv2D(4, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(nb_classes, activation='softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.01),
                      metrics=['accuracy'])

        model.fit(x_train,y_train,nb_epoch=5,batch_size=128)
        model.save('./tests/model.h5')


    def tearDown(self):
        shutil.rmtree('./tests/')



    def test_mean_class_dist_fv(self):
        """
        Testing feature workflow with a Mean Class Dist
        :return:
        """
        # Initialize a tf session
        session = tf.Session()
        k.set_session(session)

        # Get MNIST
        NB_TRAIN, NB_TEST = 1000, 10
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        x_train, y_train = x_train[:NB_TRAIN], y_train[:NB_TRAIN]
        x_test, y_test = x_test[:NB_TEST], y_test[:NB_TEST]

        nb_classes = 10

        layer_id = 2

        # compute the features using Keras
        model = load_model('./tests/model.h5')
        f = k.function([model.layers[0].input],
                                  [model.layers[layer_id].output])

        # get the per class layer output for the training set
        layer_output = f([x_train])[0]
        layer_output = layer_output.reshape(layer_output.shape[0],-1)
        layer_output_per_class_keras = [layer_output[np.where(np.argmax(y_train, axis=1) == c)[0]]
                              for c in range(nb_classes)]


        dists_ = []
        norms2_x = np.sum(layer_output ** 2, 1)[:, None]

        for c in range(nb_classes):

            norms2_y = np.sum(layer_output_per_class_keras[c] ** 2, 1)[None, :]
            pw_dists = norms2_x - 2 * np.matmul(layer_output, layer_output_per_class_keras[c].T) + norms2_y
            dists_.append(np.mean(pw_dists, axis=1))

        fv_keras = np.stack(dists_).T

        #compute the features using classifier from ART
        classifier = KerasClassifier((0, 1), model, use_logits=False)
        df = MeanClassDist(classifier=classifier,x=x_train,y=y_train,layerid=2)
        fv_art = df.extract(x=x_train)

        self.assertTrue(np.all(fv_keras==fv_art))

    def test_attention_map(self):
        """
        Testing feature workflow for AttentionMap
        :return:
        """
        # Initialize a tf session
        session = tf.Session()
        k.set_session(session)

        # Get MNIST
        NB_TRAIN, NB_TEST = 2, 2
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        x_train, y_train = x_train[:NB_TRAIN], y_train[:NB_TRAIN]
        x_test, y_test = x_test[:NB_TEST], y_test[:NB_TEST]

        nb_classes = 10

        # compute the class gradients using Keras
        model = load_model('./tests/model.h5')
        # grad_f = [k.function([model.layers[0].input],k.gradients([model.layers[-1].output[:,i]], [model.layers[0].input])) for i in range(nb_classes)]
        # fv_keras = np.max(np.abs(np.asarray([grad_f[i]([x_train])[0] for i in range(nb_classes)])),axis=0)

        # compute the class gradients using ART
        classifier = KerasClassifier((0, 1), model, use_logits=False)
        df = AttentionMap(classifier=classifier,window_width=8,strides=4)
        fv_art = df.extract(x_train)
        print(fv_art.shape)
        # self.assertTrue(np.all(fv_keras==fv_art))