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

import unittest
import keras.backend as k
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout
import numpy as np

from art.poison_detection import ActivationDefence
from art.classifiers import KerasClassifier
from art.utils import load_dataset


class TestActivationDefence(unittest.TestCase):

    # python -m unittest discover art/ -p 'activation_defence_unittest.py'

    def setUp(self):

        (self.x_train, self.y_train), (x_test, y_test), min_, max_ = load_dataset(str('mnist'))
        self.x_train = self.x_train[:300]
        self.y_train = self.y_train[:300]

        k.set_learning_phase(1)
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                         input_shape=self.x_train.shape[1:]))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.classifier = KerasClassifier((min_, max_), model=model)
        self.classifier.fit(self.x_train, self.y_train, nb_epochs=1, batch_size=128)

        self.defence = ActivationDefence(self.classifier, self.x_train, self.y_train)

    # def tearDown(self):
    #     self.classifier.dispose()
    #     self.x_train.dispose()
    #     self.y_train.dispose()

    @unittest.expectedFailure
    def test_wrong_parameters_1(self):
        self.defence.set_params(n_clusters=0)

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
        activations = self.defence._get_activations()
        self.assertEqual(len(self.x_train), len(activations))

    def test_output_clusters(self):
        n_classes = self.classifier.nb_classes
        for n_clusters in range(2, 5):
            clusters_by_class, red_activations_by_class = self.defence.cluster_activations(n_clusters=n_clusters)

            # Verify expected number of classes
            self.assertEqual(np.shape(clusters_by_class)[0], n_classes)
            # Check we get the expected number of clusters:
            found_clusters = len(np.unique(clusters_by_class[0]))
            self.assertEqual(found_clusters, n_clusters)
            # Check right amount of data
            n_dp = 0
            for i in range(0, n_classes):
                n_dp += len(clusters_by_class[i])
            self.assertEqual(len(self.x_train), n_dp)

    def test_detect_poison(self):

        confidence_level, is_clean_lst = self.defence.detect_poison(n_clusters=2,
                                                                    ndims=10,
                                                                    reduce='PCA')
        sum_clean1 = sum(is_clean_lst)

        # Check number of items in is_clean
        self.assertEqual(len(self.x_train), len(is_clean_lst))
        self.assertEqual(len(self.x_train), len(confidence_level))
        # Test right number of clusters
        found_clusters = len(np.unique(self.defence.clusters_by_class[0]))
        self.assertEqual(found_clusters, 2)

        confidence_level, is_clean_lst = self.defence.detect_poison(n_clusters=3,
                                                                    ndims=10,
                                                                    reduce='PCA',
                                                                    cluster_analysis='distance'
                                                                    )
        self.assertEqual(len(self.x_train), len(is_clean_lst))
        self.assertEqual(len(self.x_train), len(confidence_level))
        # Test change of state to new number of clusters:
        found_clusters = len(np.unique(self.defence.clusters_by_class[0]))
        self.assertEqual(found_clusters, 3)
        # Test clean data has changed
        sum_clean2 = sum(is_clean_lst)
        self.assertNotEqual(sum_clean1, sum_clean2)

        confidence_level, is_clean_lst = self.defence.detect_poison(n_clusters=2,
                                                                    ndims=10,
                                                                    reduce='PCA',
                                                                    cluster_analysis='distance'
                                                                    )
        sum_dist = sum(is_clean_lst)
        confidence_level, is_clean_lst = self.defence.detect_poison(n_clusters=2,
                                                                    ndims=10,
                                                                    reduce='PCA',
                                                                    cluster_analysis='smaller'
                                                                    )
        sum_size = sum(is_clean_lst)
        self.assertNotEqual(sum_dist, sum_size)

    def test_analyze_cluster(self):
        dist_clean_by_class = self.defence.analyze_clusters(cluster_analysis='distance')

        n_classes = self.classifier.nb_classes
        self.assertEqual(n_classes, len(dist_clean_by_class))

        # Check right amount of data
        n_dp = 0
        for i in range(0, n_classes):
            n_dp += len(dist_clean_by_class[i])
        self.assertEqual(len(self.x_train), n_dp)

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
        self.assertEqual(len(self.x_train), n_dp)

        # Very unlikely that they are the same
        self.assertNotEqual(sum_dis, sum_sz, msg='This is very unlikely to happen... there may be an error')

    if __name__ == '__main__':
        unittest.main()
