# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
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

from keras_preprocessing.image import ImageDataGenerator
import numpy as np

from art.data_generators import KerasDataGenerator
from art.defences.detector.poison import ActivationDefence
from art.utils import load_mnist
from art.visualization import convert_to_rgb

from tests.utils import master_seed

logger = logging.getLogger(__name__)

NB_TRAIN, NB_TEST, BATCH_SIZE = 300, 10, 128


class TestActivationDefence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        (x_train, y_train), (x_test, y_test), min_, max_ = load_mnist()
        x_train, y_train = x_train[:NB_TRAIN], y_train[:NB_TRAIN]
        cls.mnist = (x_train, y_train), (x_test, y_test), (min_, max_)

        # Create simple keras model
        import tensorflow as tf

        tf_version = [int(v) for v in tf.__version__.split(".")]
        if tf_version[0] == 2 and tf_version[1] >= 3:
            tf.compat.v1.disable_eager_execution()
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
        else:
            from keras.models import Sequential
            from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=x_train.shape[1:]))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Flatten())
        model.add(Dense(10, activation="softmax"))

        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        from art.estimators.classification.keras import KerasClassifier

        cls.classifier = KerasClassifier(model=model, clip_values=(min_, max_))

        cls.classifier.fit(x_train, y_train, nb_epochs=1, batch_size=128)

        cls.defence = ActivationDefence(cls.classifier, x_train, y_train)

        datagen = ImageDataGenerator()
        datagen.fit(x_train)

        data_gen = KerasDataGenerator(
            datagen.flow(x_train, y_train, batch_size=NB_TRAIN), size=NB_TRAIN, batch_size=NB_TRAIN
        )

        cls.defence_gen = ActivationDefence(cls.classifier, None, None, generator=data_gen)

    def setUp(self):
        # Set master seed
        master_seed(1234)

    @unittest.expectedFailure
    def test_wrong_parameters_1(self):
        self.defence.set_params(nb_clusters=0)

    @unittest.expectedFailure
    def test_wrong_parameters_2(self):
        self.defence.set_params(clustering_method="what")

    @unittest.expectedFailure
    def test_wrong_parameters_3(self):
        self.defence.set_params(reduce="what")

    @unittest.expectedFailure
    def test_wrong_parameters_4(self):
        self.defence.set_params(cluster_analysis="what")

    def test_activations(self):
        (x_train, _), (_, _), (_, _) = self.mnist
        activations = self.defence._get_activations()
        self.assertEqual(len(x_train), len(activations))

    def test_output_clusters(self):
        # Get MNIST
        (x_train, _), (_, _), (_, _) = self.mnist

        n_classes = self.classifier.nb_classes
        for nb_clusters in range(2, 5):
            clusters_by_class, _ = self.defence.cluster_activations(nb_clusters=nb_clusters)

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
        (x_train, _), (_, _), (_, _) = self.mnist

        _, is_clean_lst = self.defence.detect_poison(nb_clusters=2, nb_dims=10, reduce="PCA")
        sum_clean1 = sum(is_clean_lst)

        _, is_clean_lst_gen = self.defence_gen.detect_poison(nb_clusters=2, nb_dims=10, reduce="PCA")
        sum_clean1_gen = sum(is_clean_lst_gen)

        # Check number of items in is_clean
        self.assertEqual(len(x_train), len(is_clean_lst))
        self.assertEqual(len(x_train), len(is_clean_lst_gen))

        # Test right number of clusters
        found_clusters = len(np.unique(self.defence.clusters_by_class[0]))
        found_clusters_gen = len(np.unique(self.defence_gen.clusters_by_class[0]))
        self.assertEqual(found_clusters, 2)
        self.assertEqual(found_clusters_gen, 2)

        _, is_clean_lst = self.defence.detect_poison(
            nb_clusters=3, nb_dims=10, reduce="PCA", cluster_analysis="distance"
        )
        _, is_clean_lst_gen = self.defence_gen.detect_poison(
            nb_clusters=3, nb_dims=10, reduce="PCA", cluster_analysis="distance"
        )
        self.assertEqual(len(x_train), len(is_clean_lst))
        self.assertEqual(len(x_train), len(is_clean_lst_gen))

        # Test change of state to new number of clusters:
        found_clusters = len(np.unique(self.defence.clusters_by_class[0]))
        found_clusters_gen = len(np.unique(self.defence_gen.clusters_by_class[0]))
        self.assertEqual(found_clusters, 3)
        self.assertEqual(found_clusters_gen, 3)

        # Test clean data has changed
        sum_clean2 = sum(is_clean_lst)
        sum_clean2_gen = sum(is_clean_lst_gen)
        self.assertNotEqual(sum_clean1, sum_clean2)
        self.assertNotEqual(sum_clean1_gen, sum_clean2_gen)

        kwargs = {"nb_clusters": 2, "nb_dims": 10, "reduce": "PCA", "cluster_analysis": "distance"}
        _, is_clean_lst = self.defence.detect_poison(**kwargs)
        _, is_clean_lst_gen = self.defence_gen.detect_poison(**kwargs)
        sum_dist = sum(is_clean_lst)
        sum_dist_gen = sum(is_clean_lst_gen)
        kwargs = {"nb_clusters": 2, "nb_dims": 10, "reduce": "PCA", "cluster_analysis": "smaller"}
        _, is_clean_lst = self.defence.detect_poison(**kwargs)
        _, is_clean_lst_gen = self.defence_gen.detect_poison(**kwargs)
        sum_size = sum(is_clean_lst)
        sum_size_gen = sum(is_clean_lst_gen)
        self.assertNotEqual(sum_dist, sum_size)
        self.assertNotEqual(sum_dist_gen, sum_size_gen)

    def test_evaluate_defense(self):
        # Get MNIST
        (x_train, _), (_, _), (_, _) = self.mnist

        kwargs = {"nb_clusters": 2, "nb_dims": 10, "reduce": "PCA"}
        _, _ = self.defence.detect_poison(**kwargs)
        _, _ = self.defence_gen.detect_poison(**kwargs)
        is_clean = np.zeros(len(x_train))
        self.defence.evaluate_defence(is_clean)
        self.defence_gen.evaluate_defence(is_clean)

    def test_analyze_cluster(self):
        # Get MNIST
        (x_train, _), (_, _), (_, _) = self.mnist

        self.defence.analyze_clusters(cluster_analysis="relative-size")
        self.defence_gen.analyze_clusters(cluster_analysis="relative-size")

        self.defence.analyze_clusters(cluster_analysis="silhouette-scores")
        self.defence_gen.analyze_clusters(cluster_analysis="silhouette-scores")

        report, dist_clean_by_class = self.defence.analyze_clusters(cluster_analysis="distance")
        report_gen, dist_clean_by_class_gen = self.defence_gen.analyze_clusters(cluster_analysis="distance")
        n_classes = self.classifier.nb_classes
        self.assertEqual(n_classes, len(dist_clean_by_class))
        self.assertEqual(n_classes, len(dist_clean_by_class_gen))

        # Check right amount of data
        n_dp = 0
        n_dp_gen = 0
        for i in range(0, n_classes):
            n_dp += len(dist_clean_by_class[i])
            n_dp_gen += len(dist_clean_by_class_gen[i])
        self.assertEqual(len(x_train), n_dp)
        self.assertEqual(len(x_train), n_dp_gen)

        report, sz_clean_by_class = self.defence.analyze_clusters(cluster_analysis="smaller")
        report_gen, sz_clean_by_class_gen = self.defence_gen.analyze_clusters(cluster_analysis="smaller")
        n_classes = self.classifier.nb_classes
        self.assertEqual(n_classes, len(sz_clean_by_class))
        self.assertEqual(n_classes, len(sz_clean_by_class_gen))

        # Check right amount of data
        n_dp = 0
        n_dp_gen = 0
        sum_sz = 0
        sum_sz_gen = 0
        sum_dis = 0
        sum_dis_gen = 0

        for i in range(0, n_classes):
            n_dp += len(sz_clean_by_class[i])
            n_dp_gen += len(sz_clean_by_class_gen[i])
            sum_sz += sum(sz_clean_by_class[i])
            sum_sz_gen += sum(sz_clean_by_class_gen[i])
            sum_dis += sum(dist_clean_by_class[i])
            sum_dis_gen += sum(dist_clean_by_class_gen[i])
        self.assertEqual(len(x_train), n_dp)
        self.assertEqual(len(x_train), n_dp_gen)

        # Very unlikely that they are the same
        self.assertNotEqual(sum_dis, sum_sz, msg="This is very unlikely to happen... there may be an error")
        self.assertNotEqual(sum_dis_gen, sum_sz_gen, msg="This is very unlikely to happen... there may be an error")

    def test_plot_clusters(self):
        self.defence.detect_poison(nb_clusters=2, nb_dims=10, reduce="PCA")
        self.defence_gen.detect_poison(nb_clusters=2, nb_dims=10, reduce="PCA")
        self.defence.plot_clusters(save=False)
        self.defence_gen.plot_clusters(save=False)

    def test_pickle(self):

        # Test pickle and unpickle:
        filename = "test_pickle.h5"
        ActivationDefence._pickle_classifier(self.classifier, filename)
        loaded = ActivationDefence._unpickle_classifier(filename)

        np.testing.assert_equal(self.classifier._clip_values, loaded._clip_values)
        self.assertEqual(self.classifier._channels_first, loaded._channels_first)
        self.assertEqual(self.classifier._use_logits, loaded._use_logits)
        self.assertEqual(self.classifier._input_layer, loaded._input_layer)

        ActivationDefence._remove_pickle(filename)

    def test_fix_relabel_poison(self):
        (x_train, y_train), (_, _), (_, _) = self.mnist
        x_poison = x_train[:100]
        y_fix = y_train[:100]

        test_set_split = 0.7
        n_train = int(len(x_poison) * test_set_split)
        x_test = x_poison[n_train:]
        y_test = y_fix[n_train:]

        predictions = np.argmax(self.classifier.predict(x_test), axis=1)
        ini_miss = 1 - np.sum(predictions == np.argmax(y_test, axis=1)) / y_test.shape[0]

        improvement, new_classifier = ActivationDefence.relabel_poison_ground_truth(
            self.classifier,
            x_poison,
            y_fix,
            test_set_split=test_set_split,
            tolerable_backdoor=0.01,
            max_epochs=5,
            batch_epochs=10,
        )

        predictions = np.argmax(new_classifier.predict(x_test), axis=1)
        final_miss = 1 - np.sum(predictions == np.argmax(y_test, axis=1)) / y_test.shape[0]

        self.assertEqual(improvement, ini_miss - final_miss)

        # Other method (since it's cross validation we can't assert to a concrete number).
        improvement, _ = ActivationDefence.relabel_poison_cross_validation(
            self.classifier, x_poison, y_fix, n_splits=2, tolerable_backdoor=0.01, max_epochs=5, batch_epochs=10
        )
        self.assertGreaterEqual(improvement, 0)

    def test_visualizations(self):
        # test that visualization doesn't error in grayscale and RGB settings
        (x_train, _), (_, _), (_, _) = self.mnist
        self.defence.visualize_clusters(x_train)

        x_train_rgb = convert_to_rgb(x_train)
        self.defence.visualize_clusters(x_train_rgb)


if __name__ == "__main__":
    unittest.main()
