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

import logging
import unittest

import keras.backend as k
import numpy as np
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential

from art.classifiers import KerasClassifier
from art.poison_detection import ActivationDefence
from art.utils import load_mnist

logger = logging.getLogger('testLogger')

NB_TRAIN, NB_TEST, BATCH_SIZE = 300, 10, 128


class TestActivationDefence(unittest.TestCase):

    # python -m unittest discover art/ -p 'activation_defence_unittest.py'

    @classmethod
    def setUpClass(cls):

        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        x_train, y_train = x_train[:NB_TRAIN], y_train[:NB_TRAIN]
        cls.mnist = (x_train, y_train), (x_test, y_test)

        k.set_learning_phase(1)
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:]))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        cls.classifier = KerasClassifier((0, 1), model=model)
        cls.classifier.fit(x_train, y_train, nb_epochs=2, batch_size=128)

        cls.defence = ActivationDefence(cls.classifier, x_train, y_train)

    @unittest.expectedFailure
    def test_wrong_parameters_1(self):
        self.defence.set_params(nb_clusters=0)

    @unittest.expectedFailure
    def test_wrong_parameters_2(self):
        self.defence.set_params(clustering_method='what')

    @unittest.expectedFailure
    def test_wrong_parameters_3(self):
        self.defence.set_params(reduce='what')

    @unittest.expectedFailure
    def test_wrong_parameters_4(self):
        self.defence.set_params(cluster_analysis='what')

    def test_activations(self):
        (x_train, _), (_, _) = self.mnist
        activations = self.defence._get_activations()
        self.assertEqual(len(x_train), len(activations))

    def test_output_clusters(self):
        # Get MNIST
        (x_train, _), (_, _) = self.mnist

        n_classes = self.classifier.nb_classes
        for nb_clusters in range(2, 5):
            clusters_by_class, red_activations_by_class = self.defence.cluster_activations(nb_clusters=nb_clusters)

            # Verify expected number of classes
            self.assertEqual(np.shape(clusters_by_class)[0], n_classes)
            # Check we get the expected number of clusters:
            found_clusters = len(np.unique(clusters_by_class[0]))
            self.assertEqual(found_clusters, nb_clusters)
            # Check right amount of data
            n_dp = 0
            for i in range(0, n_classes):
                n_dp += len(clusters_by_class[i])
            self.assertEqual(len(x_train), n_dp)

    def test_detect_poison(self):
        # Get MNIST
        (x_train, _), (_, _) = self.mnist

        confidence_level, is_clean_lst = self.defence.detect_poison(nb_clusters=2, nb_dims=10, reduce='PCA')
        sum_clean1 = sum(is_clean_lst)

        # Check number of items in is_clean
        self.assertEqual(len(x_train), len(is_clean_lst))
        self.assertEqual(len(x_train), len(confidence_level))

        # Test right number of clusters
        found_clusters = len(np.unique(self.defence.clusters_by_class[0]))
        self.assertEqual(found_clusters, 2)

        confidence_level, is_clean_lst = self.defence.detect_poison(nb_clusters=3, nb_dims=10, reduce='PCA',
                                                                    cluster_analysis='distance')
        self.assertEqual(len(x_train), len(is_clean_lst))
        self.assertEqual(len(x_train), len(confidence_level))

        # Test change of state to new number of clusters:
        found_clusters = len(np.unique(self.defence.clusters_by_class[0]))
        self.assertEqual(found_clusters, 3)

        # Test clean data has changed
        sum_clean2 = sum(is_clean_lst)
        self.assertNotEqual(sum_clean1, sum_clean2)

        confidence_level, is_clean_lst = self.defence.detect_poison(nb_clusters=2, nb_dims=10, reduce='PCA',
                                                                    cluster_analysis='distance')
        sum_dist = sum(is_clean_lst)
        confidence_level, is_clean_lst = self.defence.detect_poison(nb_clusters=2, nb_dims=10, reduce='PCA',
                                                                    cluster_analysis='smaller')
        sum_size = sum(is_clean_lst)
        self.assertNotEqual(sum_dist, sum_size)

    def test_analyze_cluster(self):
        # Get MNIST
        (x_train, _), (_, _) = self.mnist

        dist_clean_by_class = self.defence.analyze_clusters(cluster_analysis='distance')

        n_classes = self.classifier.nb_classes
        self.assertEqual(n_classes, len(dist_clean_by_class))

        # Check right amount of data
        n_dp = 0
        for i in range(0, n_classes):
            n_dp += len(dist_clean_by_class[i])
        self.assertEqual(len(x_train), n_dp)

        sz_clean_by_class = self.defence.analyze_clusters(cluster_analysis='smaller')
        n_classes = self.classifier.nb_classes
        self.assertEqual(n_classes, len(sz_clean_by_class))

        # Check right amount of data
        n_dp = 0
        sum_sz = 0
        sum_dis = 0

        for i in range(0, n_classes):
            n_dp += len(sz_clean_by_class[i])
            sum_sz += sum(sz_clean_by_class[i])
            sum_dis += sum(dist_clean_by_class[i])
        self.assertEqual(len(x_train), n_dp)

        # Very unlikely that they are the same
        self.assertNotEqual(sum_dis, sum_sz, msg='This is very unlikely to happen... there may be an error')

    if __name__ == '__main__':
        unittest.main()
