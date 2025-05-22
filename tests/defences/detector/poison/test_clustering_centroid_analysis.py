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

import json
from unittest.mock import MagicMock, patch

import tensorflow as tf
from sklearn.base import ClusterMixin
from tensorflow.keras.metrics import Precision, Recall, AUC

import logging
import unittest
from typing import Union

import numpy as np
import pandas as pd

from art.estimators.classification import KerasClassifier, TensorFlowV2Classifier
from sklearn.cluster import DBSCAN
from sklearn.decomposition import FastICA, PCA
from tensorflow.keras import Model, Sequential, Input
from tensorflow.keras.layers import Dense
from umap import UMAP

from art.defences.detector.poison.clustering_centroid_analysis import ClusteringCentroidAnalysis, _calculate_centroid, _class_clustering, _feature_extraction, _cluster_classes, _encode_labels
from art.defences.detector.poison.utils import ReducerType, ClustererType

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# TODO: add a better formatter for the logger. Eliminate date

def _create_mlp_model(input_dim: int, model_name: str) -> Model:

    # Define a small DNN with one hidden layer
    base_model = Sequential(name=model_name, layers=[
        Dense(64, activation="relu", name="input_layer", input_shape=(input_dim,)), # FIXME: 28, 1, 1 to input_dim,
        Dense(64, activation="relu", name="hidden_layer"),
        Dense(1, activation="sigmoid", name="output_layer")
    ])
    base_model.compile(optimizer="adam",
                       loss="binary_crossentropy",
                       metrics=["accuracy",
                                Precision(name="precision"),
                                Recall(name="recall"),
                                AUC(name="auc")])

    base_model.summary(print_fn=logger.info)
    return base_model


def train_art_keras_classifier(x_train: Union[pd.DataFrame, np.ndarray] , y_train: Union[pd.DataFrame, np.ndarray], model_name: str) -> KerasClassifier:
    """Trains a KerasClassifier using the ART wrapper."""

    # Create the Keras model
    mlp_model = _create_mlp_model(x_train.shape[1], model_name)

    # Create the ART KerasClassifier wrapper
    mlp_classifier = TensorFlowV2Classifier(
        model=mlp_model,
        loss_object=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        nb_classes=2,  # Ensure this is set correctly
        input_shape=(x_train.shape[1],)
    )


    # Requires ndarrays, so the dataframes are transformed
    x_values = x_train.values if type(x_train) == pd.DataFrame else x_train
    y_values = np.squeeze(y_train.values) if type(y_train) == pd.DataFrame else y_train

    # Train the model
    mlp_classifier.fit(x_values, y_values, batch_size=512, nb_epochs=100, verbose=True)

    return mlp_classifier

class MockClusterer(ClusterMixin):
    """
    A mock ClusterMixin for testing purposes.  This avoids using a real clustering
    algorithm and allows us to control the output for our tests.
    """
    def __init__(self, cluster_labels_to_return):
        self.cluster_labels_to_return = cluster_labels_to_return

    def fit_predict(self, x, **kwargs):
        return self.cluster_labels_to_return


class TestInitialization(unittest.TestCase):
    """
    Unit tests for the ClusteringCentroidAnalysis class, focusing on
    __init__, _get_benign_data, and _extract_submodels.
    """

    def setUp(self):
        # Create mock data and objects for testing
        self.x_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        self.y_train = np.array([0, 1, 0, 1])
        self.benign_indices = np.array([0, 2])
        self.final_feature_layer_name = 'dense_2'
        self.non_relu_intermediate_layer_name = 'dense_3'
        self.misclassification_threshold = 0.05

        # Define a simple Keras model for testing _extract_submodels
        self.mock_model = tf.keras.Sequential([
            tf.keras.layers.Dense(4, activation='relu', input_shape=(10,), name='input_layer'),
            tf.keras.layers.Dense(8, activation='tanh', name='dense_1'),
            tf.keras.layers.Dense(6, activation='relu', name='dense_2'),
            tf.keras.layers.Dense(2, activation='sigmoid', name='dense_3'),
            tf.keras.layers.Dense(1, activation='sigmoid', name="output_layer")
        ])

        # Generate some dummy training data
        self.x_train = np.random.rand(100, 10)
        self.y_train = np.random.randint(0, 2, size=(100,)) # For M.E multi-class

        # Compile and train the model
        self.mock_model.compile(optimizer='adam', loss='binary_crossentropy')
        self.mock_model.fit(self.x_train, self.y_train, epochs=1) # Train for a few steps

        self.mock_classifier = TensorFlowV2Classifier(
            model=self.mock_model,
            loss_object=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            nb_classes=2, input_shape=(10,)
        )

    def is_valid_scaler(self, obj):
        """Check if an object is a valid scaler."""
        self.assertTrue(hasattr(obj, 'fit_transform'))
        self.assertTrue(callable(getattr(obj, 'fit_transform')))

    def is_valid_reducer(self, obj):
        """Check if an object is a valid reducer."""
        self.assertTrue(hasattr(obj, 'fit_transform'))
        self.assertTrue(callable(getattr(obj, 'fit_transform')))

    def is_valid_clusterer(self, obj):
        """Check if an object is a valid clusterer."""
        self.assertTrue(hasattr(obj, 'fit_predict'))
        self.assertTrue(callable(getattr(obj, 'fit_predict')))

    def test_get_benign_data_basic(self):
        """Test _get_benign_data with a simple example."""
        cca = ClusteringCentroidAnalysis(
            classifier=self.mock_classifier,
            x_train=self.x_train,
            y_train=self.y_train,
            benign_indices=self.benign_indices,
            final_feature_layer_name=self.final_feature_layer_name,
            misclassification_threshold=self.misclassification_threshold
        )
        x_benign, y_benign = cca._get_benign_data()
        self.assertTrue(np.array_equal(x_benign, self.x_train[[0, 2]]))
        self.assertTrue(np.array_equal(y_benign, self.y_train[[0, 2]]))

    def test_extract_submodels_valid_layer(self):
        """Test _extract_submodels with a valid layer name."""
        cca = ClusteringCentroidAnalysis(
            classifier=self.mock_classifier,
            x_train=self.x_train,
            y_train=self.y_train,
            benign_indices=self.benign_indices,
            final_feature_layer_name=self.final_feature_layer_name,
            misclassification_threshold=self.misclassification_threshold
        )
        feature_model, classify_model = cca.feature_representation_model, cca.classifying_submodel

        # Verify model types and names
        self.assertIsInstance(feature_model, Model)
        self.assertIsInstance(classify_model, Sequential)
        self.assertEqual('feature_representation_model', feature_model.name)
        self.assertEqual('classifying_submodel', classify_model.name)

        # Create a test input and get reference output
        sample_input = np.random.rand(1, 10)

        sample_output = self.mock_classifier.model.predict(sample_input)

        # Get feature representation
        feature_value = feature_model.predict(sample_input)

        # Verify intermediate feature shape is compatible with classifier input
        self.assertEqual(classify_model.input_shape[1], feature_value.shape[1],
                         "Feature sub model output and classifying sub model input do not match.")

        # Predict with classifier submodel
        final_value = classify_model.predict(feature_value)

        # Verify output shapes match
        self.assertEqual(sample_output.shape, final_value.shape)

        # Due to TensorFlow non-eager mode, we might have numerical differences
        # We test that the outputs are approximately equal with a reasonable tolerance
        np.testing.assert_allclose(sample_output, final_value, rtol=1e-3, atol=1e-3)


    def test_extract_submodels_invalid_layer(self):
        """Test _extract_submodels with an invalid layer name.  Check for error"""
        cca = ClusteringCentroidAnalysis(
            classifier=self.mock_classifier,
            x_train=self.x_train,
            y_train=self.y_train,
            benign_indices=self.benign_indices,
            final_feature_layer_name=self.final_feature_layer_name,
            misclassification_threshold=self.misclassification_threshold
        )
        with self.assertRaises(ValueError):  # Expect a ValueError
            cca._extract_submodels('invalid_layer_name')

    def test_init_basic(self):
        """Test __init__ with valid inputs."""
        cca = ClusteringCentroidAnalysis(
            classifier=self.mock_classifier,
            x_train=self.x_train,
            y_train=self.y_train,
            benign_indices=self.benign_indices,
            final_feature_layer_name=self.final_feature_layer_name,
            misclassification_threshold=self.misclassification_threshold
        )
        self.assertEqual(self.mock_classifier, cca.classifier)
        self.assertTrue(np.array_equal(cca.x_train, self.x_train))
        self.assertTrue(np.array_equal(cca.y_train, self.y_train))
        self.assertTrue(np.array_equal(cca.benign_indices, self.benign_indices))
        self.assertEqual(self.misclassification_threshold, cca.misclassification_threshold)
        self.is_valid_reducer(cca.reducer)
        self.is_valid_clusterer(cca.clusterer)
        self.assertTrue(np.array_equal(cca.x_benign, self.x_train[[0, 2]]))
        self.assertTrue(np.array_equal(cca.y_benign, self.y_train[[0, 2]]))
        self.assertIsInstance(cca.feature_representation_model, Model)
        self.assertIsInstance(cca.classifying_submodel, Sequential)
        self.assertEqual({0, 1}, cca.unique_classes)

    def test_init_empty_benign_indices(self):
        """Test __init__ with empty benign indices."""
        with self.assertRaises(ValueError) as e:
            ClusteringCentroidAnalysis(
                classifier=self.mock_classifier,
                x_train=self.x_train,
                y_train=self.y_train,
                benign_indices=np.array([]),
                final_feature_layer_name=self.final_feature_layer_name,
                misclassification_threshold=self.misclassification_threshold
            )
            self.assertEqual('Benign indices passed (0) are not enough to run the algorithm', str(e.exception))

    def test_init_invalid_layer_name(self):
        """Test __init__ with an invalid layer name. Check that it raises error."""
        with self.assertRaises(ValueError) as e:
            ClusteringCentroidAnalysis(
                classifier=self.mock_classifier,
                x_train=self.x_train,
                y_train=self.y_train,
                benign_indices=self.benign_indices,
                final_feature_layer_name='invalid_layer',
                misclassification_threshold=self.misclassification_threshold
            )
            self.assertEqual(f"Layer with name 'invalid_layer' not found in the model.", str(e.exception))

    def test_init_invalid_layer_non_relu(self):
        """Test __init__ with an invalid layer that does not have ReLu activation. Check that it raises error."""
        with self.assertWarns(UserWarning) as w:
            ClusteringCentroidAnalysis(
                classifier=self.mock_classifier,
                x_train=self.x_train,
                y_train=self.y_train,
                benign_indices=self.benign_indices,
                final_feature_layer_name=self.non_relu_intermediate_layer_name,
                misclassification_threshold=self.misclassification_threshold
            )
            self.assertEqual(1, len(w.warnings))
            self.assertEqual(f"Final feature layer '{self.non_relu_intermediate_layer_name}' must have a ReLU activation.",
                             str(w.warnings[0].message))

class TestEncodeLabels(unittest.TestCase):

    def test_encode_binary_labels(self):
        y = np.array([1, 0, 0, 1, 0, 1, 0, 1])
        y_encoded, unique_classes, label_mapping, reverse_mapping = _encode_labels(y)

        np.testing.assert_array_equal(y, y_encoded)
        self.assertEqual({0, 1}, unique_classes)
        np.testing.assert_array_equal(np.array([0, 1]), label_mapping)
        self.assertEqual({0: 0, 1: 1}, reverse_mapping)

    def test_encode_multi_labels(self):
        y = np.array(['A', 'B', 'C', 'B', 'B', 'C', 'A', 'D'])
        y_encoded, unique_classes, label_mapping, reverse_mapping = _encode_labels(y)

        np.testing.assert_array_equal(np.array([0, 1, 2, 1, 1, 2, 0, 3]), y_encoded)
        self.assertEqual({0, 1, 2, 3}, unique_classes)
        np.testing.assert_array_equal(np.array(['A', 'B', 'C', 'D']), label_mapping)
        self.assertEqual({'A': 0, 'B': 1, 'C': 2, 'D': 3}, reverse_mapping)


class TestCalculateCentroid(unittest.TestCase):

    def setUp(self):
        # Example feature data for testing
        self.features = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15]
        ])

    def test_empty_indices(self):
        """Test with an empty array of selected indices."""
        selected_indices = np.array([], dtype=int)
        centroid = _calculate_centroid(selected_indices, self.features)
        self.assertTrue(np.all(np.isnan(centroid)), "Centroid of empty selection should be NaN")

    def test_single_index(self):
        """Test with a single selected index."""
        selected_indices = np.array([0])
        centroid = _calculate_centroid(selected_indices, self.features)
        self.assertTrue(np.array_equal(centroid, self.features[0]), "Centroid should be the feature itself")

    def test_multiple_indices(self):
        """Test with multiple selected indices."""
        selected_indices = np.array([0, 2, 4])
        expected_centroid = np.array([7, 8, 9])
        centroid = _calculate_centroid(selected_indices, self.features)
        self.assertTrue(np.array_equal(centroid, expected_centroid), "Centroid calculation incorrect")

    def test_all_indices(self):
        """Test with all indices selected."""
        selected_indices = np.array([0, 1, 2, 3, 4])
        expected_centroid = np.array([7, 8, 9])
        centroid = _calculate_centroid(selected_indices, self.features)
        self.assertTrue(np.allclose(centroid, expected_centroid), "Centroid should be the mean of all features")

    def test_non_contiguous_indices(self):
        """Test with non-contiguous selected indices."""
        selected_indices = np.array([1, 3])
        expected_centroid = np.array([7, 8, 9])
        centroid = _calculate_centroid(selected_indices, self.features)
        self.assertTrue(np.array_equal(centroid, expected_centroid), "Centroid calculation incorrect for non-contiguous indices")

    def test_float_features(self):
        """Test with float feature values."""
        float_features = self.features.astype(float)
        selected_indices = np.array([0, 2, 4])
        expected_centroid = np.array([7., 8., 9.])
        centroid = _calculate_centroid(selected_indices, float_features)
        self.assertTrue(np.allclose(centroid, expected_centroid), "Centroid calculation incorrect for float features")


class TestClassClustering(unittest.TestCase):

    def test_class_clustering_basic(self):
        """Test with a simple scenario with one class present."""
        y = np.array([0, 0, 0, 0])
        features = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        label = 0
        clusterer = MockClusterer(np.array([0, 0, 1, 1]))  # Mock cluster labels

        cluster_labels, selected_indices = _class_clustering(y, features, label, clusterer)

        self.assertTrue(np.array_equal(cluster_labels, np.array([0, 0, 1, 1])))
        self.assertTrue(np.array_equal(selected_indices, np.array([0, 1, 2, 3])))

    def test_class_clustering_multiple_classes(self):
        """Test with multiple classes, checking label selection."""
        y = np.array([0, 1, 0, 1, 0])
        features = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        label = 1
        clusterer = MockClusterer(np.array([0, 1]))  # Mock cluster labels for class 1

        cluster_labels, selected_indices = _class_clustering(y, features, label, clusterer)

        self.assertTrue(np.array_equal(cluster_labels, np.array([0, 1])))
        self.assertTrue(np.array_equal(selected_indices, np.array([1, 3])))

    def test_class_clustering_no_matching_label(self):
        """Test when the label is not present in y."""
        y = np.array([0, 0, 0])
        features = np.array([[1, 2], [3, 4], [5, 6]])
        label = 1  # Label not in y
        clusterer = MockClusterer(np.array([]))

        cluster_labels, selected_indices = _class_clustering(y, features, label, clusterer)

        self.assertTrue(np.array_equal(cluster_labels, np.array([])))
        self.assertTrue(np.array_equal(selected_indices, np.array([])))

    def test_class_clustering_different_clusterer(self):
        """Test with a different (but still mocking) clusterer."""
        y = np.array([0, 0, 0])
        features = np.array([[1, 2], [3, 4], [5, 6]])
        label = 0
        clusterer = MockClusterer(np.array([2, 2, 2]))  # All in one cluster

        cluster_labels, selected_indices = _class_clustering(y, features, label, clusterer)

        self.assertTrue(np.array_equal(cluster_labels, np.array([2, 2, 2])))
        self.assertTrue(np.array_equal(selected_indices, np.array([0, 1, 2])))

    def test_class_clustering_complex_labels(self):
        """Test with non-integer labels."""
        y = np.array(["a", "b", "a", "b"])
        features = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        label = "b"
        clusterer = MockClusterer(np.array([0, 1]))

        cluster_labels, selected_indices = _class_clustering(y, features, label, clusterer)

        self.assertTrue(np.array_equal(cluster_labels, np.array([0, 1])))
        self.assertTrue(np.array_equal(selected_indices, np.array([1, 3])))

class TestClusterClasses(unittest.TestCase):
    """Unit tests for the _cluster_classes function."""

    def setUp(self):
        """Set up test fixtures for each test."""
        # Create mock features for testing
        self.features = np.array([
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
            [9, 10],
            [11, 12]
        ])

    def test_single_target_class(self):
        """Test clustering with a single target class."""
        # Setup
        y_train = np.array([0, 0, 0, 0, 0, 0])  # All samples belong to class 0
        unique_classes = {0}  # Only one class

        # Mock clusterer that assigns samples to two clusters (0 and 1) with no outliers
        clusterer = MockClusterer(np.array([0, 0, 0, 1, 1, 1]))

        # Execute
        class_cluster_labels, cluster_class_mapping = _cluster_classes(
            y_train, unique_classes, self.features, clusterer)

        # Assert
        self.assertEqual(6, len(class_cluster_labels))
        self.assertEqual(2, len(cluster_class_mapping))

        # All cluster labels should be either 0 or 1
        self.assertTrue(np.all(np.isin(class_cluster_labels, [0, 1])))

        # Both clusters should be mapped to class 0
        self.assertEqual(0, cluster_class_mapping[0])
        self.assertEqual(0, cluster_class_mapping[1])

        # Check cluster distribution
        self.assertEqual(3, np.sum(class_cluster_labels == 0))  # First 3 samples in cluster 0
        self.assertEqual(3, np.sum(class_cluster_labels == 1))  # Last 3 samples in cluster 1

    def test_binary_target_classes(self):
        """Test clustering with binary (two) target classes."""
        # Setup
        y_train = np.array([0, 0, 0, 1, 1, 1])  # Class 0 and 1
        unique_classes = {0, 1}

        # Mock clusterer that will return:
        # - For class 0: [0, 0, 1]
        # - For class 1: [0, 1, 1]
        class CustomMockClusterer(MockClusterer):
            def fit_predict(self, x, **kwargs):
                return np.array([0, 0, 1])

        clusterer = CustomMockClusterer(None)  # The parameter is ignored due to overridden fit_predict

        # Execute
        class_cluster_labels, cluster_class_mapping = _cluster_classes(
            y_train, unique_classes, self.features, clusterer)

        np.testing.assert_array_equal(np.array([0, 0, 1, 2, 2, 3]), class_cluster_labels)

        # Check cluster-to-class mapping
        self.assertEqual(4, len(cluster_class_mapping))  # 4 clusters total
        self.assertEqual(0, cluster_class_mapping[0])  # Cluster 0 -> Class 0
        self.assertEqual(0, cluster_class_mapping[1])  # Cluster 1 -> Class 0
        self.assertEqual(1, cluster_class_mapping[2])  # Cluster 2 -> Class 1
        self.assertEqual(1, cluster_class_mapping[3])  # Cluster 3 -> Class 1

    def test_multiple_target_classes(self):
        """Test clustering with multiple (more than two) target classes."""
        # Setup
        y_train = np.array([0, 0, 1, 1, 2, 2])  # Three classes: 0, 1, 2
        unique_classes = {0, 1, 2}

        # Mock clusterer that returns a single cluster for each class
        class ThreeClassMockClusterer(MockClusterer):
            def fit_predict(self, x, **kwargs):
                # Always return [0, 0] for any input of size 2
                return np.array([0, 0])

        clusterer = ThreeClassMockClusterer(None)

        # Execute
        class_cluster_labels, cluster_class_mapping = _cluster_classes(
            y_train, unique_classes, self.features, clusterer)

        # Assert
        self.assertEqual(len(class_cluster_labels), 6)

        # We should have one cluster per class, with increasing IDs
        expected_labels = np.array([0, 0, 1, 1, 2, 2])
        self.assertTrue(np.array_equal(class_cluster_labels, expected_labels))

        # Check cluster-to-class mapping
        self.assertEqual(3, len(cluster_class_mapping))  # 3 clusters total
        self.assertEqual(0, cluster_class_mapping[0])  # Cluster 0 -> Class 0
        self.assertEqual(1, cluster_class_mapping[1])  # Cluster 1 -> Class 1
        self.assertEqual(2, cluster_class_mapping[2])  # Cluster 2 -> Class 2

    def test_with_outliers(self):
        """Test clustering that detects outliers (marked as -1)."""
        # Setup
        y_train = np.array([0, 0, 0, 1, 1, 1])
        unique_classes = {0, 1}

        # Mock clusterer that returns outliers (-1) for some samples
        class OutlierMockClusterer(MockClusterer):
            def fit_predict(self, x, **kwargs):
                return np.array([0, -1, -1])

        clusterer = OutlierMockClusterer(None)

        # Execute
        class_cluster_labels, cluster_class_mapping = _cluster_classes(
            y_train, unique_classes, self.features, clusterer)

        np.testing.assert_array_equal(np.array([0, -1, -1, 1, -1, -1]), class_cluster_labels)

        # Check cluster-to-class mapping (should only have non-outlier clusters)
        self.assertEqual(len(cluster_class_mapping), 2)
        self.assertEqual(cluster_class_mapping[0], 0)  # Cluster 0 -> Class 0
        self.assertEqual(cluster_class_mapping[1], 1)  # Cluster 1 -> Class 1

    def test_empty_target_classes(self):
        """Test with an empty set of target classes."""
        # Setup
        y_train = np.array([0, 0, 1, 1])
        unique_classes = set()  # Empty set

        clusterer = MockClusterer(np.array([0, 0]))

        # Execute
        class_cluster_labels, cluster_class_mapping = _cluster_classes(
            y_train, unique_classes, self.features, clusterer)

        # Assert
        self.assertEqual(len(class_cluster_labels), 4)

        # With no classes to process, all elements should be "unassigned"
        # The array was initialized with np.empty, so we can't easily test values
        # but the cluster mapping should be empty
        self.assertEqual(len(cluster_class_mapping), 0)

    def test_all_samples_in_one_cluster(self):
        """Test when all samples of a class are assigned to a single cluster."""
        # Setup
        y_train = np.array([0, 0, 0, 1, 1, 1])
        unique_classes = {0, 1}

        # All samples in one cluster per class
        clusterer = MockClusterer(np.array([0, 0, 0]))

        # Execute
        class_cluster_labels, cluster_class_mapping = _cluster_classes(
            y_train, unique_classes, self.features, clusterer)

        np.testing.assert_array_equal(np.array([0, 0, 0, 1, 1, 1]), class_cluster_labels)

        # Check cluster-to-class mapping
        self.assertEqual(len(cluster_class_mapping), 2)
        self.assertEqual(cluster_class_mapping[0], 0)  # Cluster 0 -> Class 0
        self.assertEqual(cluster_class_mapping[1], 1)  # Cluster 1 -> Class 1

    def test_all_outliers(self):
        """Test when all samples are detected as outliers."""
        # Setup
        y_train = np.array([0, 0, 1, 1])
        unique_classes = {0, 1}

        # All samples are outliers (-1)
        class AllOutliersMockClusterer(MockClusterer):
            def fit_predict(self, x, **kwargs):
                return np.full(x.shape[0], -1)

        clusterer = AllOutliersMockClusterer(None)

        # Execute
        class_cluster_labels, cluster_class_mapping = _cluster_classes(
            y_train, unique_classes, self.features, clusterer)

        np.testing.assert_array_equal(np.array([-1, -1, -1, -1]), class_cluster_labels)

        # No cluster should be mapped to any class
        self.assertEqual(len(cluster_class_mapping), 0)

class TestFeatureExtraction(unittest.TestCase):

    """Unit tests for the _feature_extraction function."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a simple model for testing
        self.input_shape = (10,)
        inputs = Input(shape=self.input_shape)
        x = Dense(20, activation='relu')(inputs)
        outputs = Dense(5, activation='relu')(x)
        self.model = Model(inputs=inputs, outputs=outputs)

        # Create sample data
        self.x_train = np.random.rand(100, 10)

        # Mock feature output for consistent testing
        self.mock_features = np.random.rand(100, 5)


    def test_integration_with_real_model(self):
        """Integration test with a real model and no mocking."""
        # Create a small real model
        model = Sequential([
            Dense(20, activation='relu', input_shape=self.input_shape),
            Dense(10, activation='relu'),
            Dense(5, activation='relu')
        ])
        model.compile(optimizer='adam', loss='mse')

        # Execute
        result = _feature_extraction(self.x_train, model)

        # Assert
        self.assertEqual((100, 5), result.shape)
        self.assertIsInstance(result, np.ndarray)

class TestEvaluateDefence(unittest.TestCase):
    """
    Unit tests for the evaluate_defence method of the ClusteringCentroidAnalysis class.
    """

    def setUp(self):
        """
        Set up a mock ClusteringCentroidAnalysis object and its necessary attributes.
        """
        self.mock_classifier = MagicMock()
        self.mock_classifier.model = MagicMock()

        # Dummy data for constructor - these values might not be directly used by
        # evaluate_defence but are needed for instantiation.
        x_train_dummy = np.array([[1, 2], [3, 4], [5, 6]])
        y_train_constructor_dummy = np.array(['A', 'B', 'A'])
        benign_indices_dummy = np.array([0, 2])
        final_feature_layer_name_dummy = "mock_feature_layer"
        misclassification_threshold_dummy = 0.1

        # Patch _extract_submodels to avoid complex model setup if it's problematic
        # and not relevant to evaluate_defence
        with patch('art.defences.detector.poison.clustering_centroid_analysis.ClusteringCentroidAnalysis._extract_submodels',
                   return_value=(MagicMock(), MagicMock())) as _:
            self.defence = ClusteringCentroidAnalysis(
                classifier=self.mock_classifier,
                x_train=x_train_dummy,
                y_train=y_train_constructor_dummy, # Used by _encode_labels in __init__
                benign_indices=benign_indices_dummy,
                final_feature_layer_name=final_feature_layer_name_dummy,
                misclassification_threshold=misclassification_threshold_dummy
            )

        # The following attributes are set after instantiation to control the test
        # environment precisely

        self.defence.unique_classes = {0, 1} # e.g., 'A' -> 0, 'B' -> 1
        self.defence.y_train = np.array([0, 0, 1, 1, 0]) # Total 5 samples
        self.defence.is_clean = np.array([1, 0, 1, 0, 1]) # Predictions by the defence

    @patch('art.defences.detector.poison.clustering_centroid_analysis.GroundTruthEvaluator')
    def test_evaluate_defence_basic_case(self, MockGroundTruthEvaluator):
        """
        Test evaluate_defence with a basic scenario of ground truth and predictions.
        """
        # Mock setup
        mock_evaluator_instance = MockGroundTruthEvaluator.return_value
        expected_json_report = json.dumps({"accuracy": 0.6, "class_0_fp": 1, "class_1_fn": 0})
        mock_evaluator_instance.analyze_correctness.return_value = (
            {"some_errors": []},  # errors_by_class (not directly used by evaluate_defence's return)
            expected_json_report    # confusion_matrix_json
        )

        # Ground truth setup
        # This is the `is_clean` array passed as an argument to evaluate_defence
        ground_truth_is_clean = np.array([1, 1, 1, 0, 0]) # Ground truth for the 5 samples

        returned_json_report = self.defence.evaluate_defence(is_clean=ground_truth_is_clean)

        self.assertEqual(returned_json_report, expected_json_report)

        # Verify how analyze_correctness was called
        mock_evaluator_instance.analyze_correctness.assert_called_once()
        call_args = mock_evaluator_instance.analyze_correctness.call_args[1] # Get kwargs

        # Expected segmentation based on self.defence.y_train, self.defence.is_clean, and ground_truth_is_clean
        # self.defence.y_train = np.array([0, 0, 1, 1, 0])
        # self.defence.is_clean (predictions) = np.array([1, 0, 1, 0, 1])
        # ground_truth_is_clean (truth) = np.array([1, 1, 1, 0, 0])

        # Class 0 indices: 0, 1, 4
        # Class 1 indices: 2, 3

        expected_assigned_clean_by_class = [
            self.defence.is_clean[[0, 1, 4]], # Predictions for class 0: [1, 0, 1]
            self.defence.is_clean[[2, 3]]     # Predictions for class 1: [1, 0]
        ]
        expected_is_clean_by_class = [
            ground_truth_is_clean[[0, 1, 4]], # Ground truth for class 0: [1, 1, 0]
            ground_truth_is_clean[[2, 3]]     # Ground truth for class 1: [1, 0]
        ]

        # np.testing.assert_equal doesn't work well for lists of arrays directly in assert_called_with
        # So we compare element by element
        self.assertEqual(len(call_args['assigned_clean_by_class']), len(expected_assigned_clean_by_class))
        for i, arr in enumerate(call_args['assigned_clean_by_class']):
            np.testing.assert_array_equal(arr, expected_assigned_clean_by_class[i])

        self.assertEqual(len(call_args['is_clean_by_class']), len(expected_is_clean_by_class))
        for i, arr in enumerate(call_args['is_clean_by_class']):
            np.testing.assert_array_equal(arr, expected_is_clean_by_class[i])


    @patch('art.defences.detector.poison.clustering_centroid_analysis.GroundTruthEvaluator')
    def test_evaluate_defence_all_predicted_clean_all_truth_clean(self, MockGroundTruthEvaluator):
        """
        Test case: All samples predicted as clean by defence, and all are truly clean.
        """
        mock_evaluator_instance = MockGroundTruthEvaluator.return_value
        expected_json_report = json.dumps({"accuracy": 1.0})
        mock_evaluator_instance.analyze_correctness.return_value = ({}, expected_json_report)

        self.defence.is_clean = np.ones_like(self.defence.y_train) # All predicted clean
        ground_truth_is_clean = np.ones_like(self.defence.y_train) # All truly clean

        returned_json_report = self.defence.evaluate_defence(is_clean=ground_truth_is_clean)
        self.assertEqual(returned_json_report, expected_json_report)

        call_args = mock_evaluator_instance.analyze_correctness.call_args[1]

        # self.defence.y_train = np.array([0, 0, 1, 1, 0])
        # Class 0 indices: 0, 1, 4
        # Class 1 indices: 2, 3
        expected_assigned_clean_by_class = [np.array([1,1,1]), np.array([1,1])]
        expected_is_clean_by_class = [np.array([1,1,1]), np.array([1,1])]

        self.assertEqual(len(call_args['assigned_clean_by_class']), len(expected_assigned_clean_by_class))
        for i, arr in enumerate(call_args['assigned_clean_by_class']):
            np.testing.assert_array_equal(arr, expected_assigned_clean_by_class[i])

        self.assertEqual(len(call_args['is_clean_by_class']), len(expected_is_clean_by_class))
        for i, arr in enumerate(call_args['is_clean_by_class']):
            np.testing.assert_array_equal(arr, expected_is_clean_by_class[i])


    @patch('art.defences.detector.poison.clustering_centroid_analysis.GroundTruthEvaluator')
    def test_evaluate_defence_all_predicted_poisoned_all_truth_poisoned(self, MockGroundTruthEvaluator):
        """
        Test case: All samples predicted as poisoned, and all are truly poisoned.
        """
        mock_evaluator_instance = MockGroundTruthEvaluator.return_value
        expected_json_report = json.dumps({"accuracy": 1.0, "tn_perfect": True}) # Example detail
        mock_evaluator_instance.analyze_correctness.return_value = ({}, expected_json_report)

        self.defence.is_clean = np.zeros_like(self.defence.y_train) # All predicted poisoned
        ground_truth_is_clean = np.zeros_like(self.defence.y_train) # All truly poisoned

        returned_json_report = self.defence.evaluate_defence(is_clean=ground_truth_is_clean)
        self.assertEqual(returned_json_report, expected_json_report)

        call_args = mock_evaluator_instance.analyze_correctness.call_args[1]
        expected_assigned_clean_by_class = [np.array([0,0,0]), np.array([0,0])]
        expected_is_clean_by_class = [np.array([0,0,0]), np.array([0,0])]

        self.assertEqual(len(call_args['assigned_clean_by_class']), len(expected_assigned_clean_by_class))
        for i, arr in enumerate(call_args['assigned_clean_by_class']):
            np.testing.assert_array_equal(arr, expected_assigned_clean_by_class[i])

        self.assertEqual(len(call_args['is_clean_by_class']), len(expected_is_clean_by_class))
        for i, arr in enumerate(call_args['is_clean_by_class']):
            np.testing.assert_array_equal(arr, expected_is_clean_by_class[i])

    @patch('art.defences.detector.poison.clustering_centroid_analysis.GroundTruthEvaluator')
    def test_evaluate_defence_no_samples_for_a_class_in_unique_classes(self, MockGroundTruthEvaluator):
        """
        Test case: A class in unique_classes has no samples in y_train (edge case).
        This shouldn't happen if unique_classes is derived from y_train correctly,
        but tests robustness.
        """
        mock_evaluator_instance = MockGroundTruthEvaluator.return_value
        expected_json_report = json.dumps({"note": "class 2 had no samples"})
        mock_evaluator_instance.analyze_correctness.return_value = ({}, expected_json_report)

        self.defence.unique_classes = {0, 1, 2} # Add class 2
        # self.defence.y_train remains [0, 0, 1, 1, 0] (no samples for class 2)
        self.defence.is_clean = np.array([1, 0, 1, 0, 1])
        ground_truth_is_clean = np.array([1, 1, 1, 0, 0])

        returned_json_report = self.defence.evaluate_defence(is_clean=ground_truth_is_clean)
        self.assertEqual(returned_json_report, expected_json_report)

        call_args = mock_evaluator_instance.analyze_correctness.call_args[1]

        # Class 0 indices: 0, 1, 4
        # Class 1 indices: 2, 3
        # Class 2 indices: []
        expected_assigned_clean_by_class = [
            self.defence.is_clean[[0, 1, 4]],
            self.defence.is_clean[[2, 3]],
            np.array([]) # Empty for class 2
        ]
        expected_is_clean_by_class = [
            ground_truth_is_clean[[0, 1, 4]],
            ground_truth_is_clean[[2, 3]],
            np.array([]) # Empty for class 2
        ]

        self.assertEqual(len(call_args['assigned_clean_by_class']), len(expected_assigned_clean_by_class))
        for i, arr in enumerate(call_args['assigned_clean_by_class']):
            np.testing.assert_array_equal(arr, expected_assigned_clean_by_class[i],
                                          err_msg=f"Mismatch in assigned_clean_by_class at index {i}")

        self.assertEqual(len(call_args['is_clean_by_class']), len(expected_is_clean_by_class))
        for i, arr in enumerate(call_args['is_clean_by_class']):
            np.testing.assert_array_equal(arr, expected_is_clean_by_class[i],
                                          err_msg=f"Mismatch in is_clean_by_class at index {i}")


class TestDetectPoison(unittest.TestCase):
    """
    Unit tests for the detect_poison method in ClusteringCentroidAnalysis
    """

    def setUp(self):
        """Set up test fixtures for each test case."""
        # Create a mock classifier with a simple model architecture
        self.input_shape = (10,)
        inputs = tf.keras.Input(shape=self.input_shape)
        x = tf.keras.layers.Dense(20, activation='relu', name='hidden_layer')(inputs)
        outputs = tf.keras.layers.Dense(2, activation='softmax', name='output_layer')(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy')

        # Mock the classifier
        self.mock_classifier = MagicMock()
        self.mock_classifier.model = self.model

        # Create sample data - 100 samples, 10 features, 2 classes
        np.random.seed(42)  # For reproducibility
        self.x_train = np.random.randn(100, 10)
        self.y_train = np.random.randint(0, 2, 100)

        # Benign indices (assuming first 20 samples are definitely benign)
        self.benign_indices = np.arange(20)

        # Create mock feature representation and classifier models
        self.mock_feature_model = MagicMock()
        self.mock_classifier_model = MagicMock()

        # Setup for mock feature extraction
        self.mock_features = np.random.randn(100, 5)

        # Common patches for all tests
        self.patches = [
            patch('art.defences.detector.poison.clustering_centroid_analysis._feature_extraction',
                  return_value=self.mock_features),
            patch('art.defences.detector.poison.clustering_centroid_analysis._calculate_centroid',
                  side_effect=self._mock_calculate_centroid),
            patch('art.defences.detector.poison.clustering_centroid_analysis._cluster_classes',
                  side_effect=self._mock_cluster_classes)
        ]

        # Apply patches
        for p in self.patches:
            p.start()

    def tearDown(self):
        """Clean up after each test."""
        # Stop all patches
        for p in self.patches:
            p.stop()

    def _mock_calculate_centroid(self, indices, features):
        """Mock implementation of _calculate_centroid."""
        return np.mean(features[indices], axis=0)

    def _mock_cluster_classes(self, y_train, unique_classes, features, clusterer, all_benign=True):
        """
        Mock implementation of _cluster_classes.

        Parameters:
        -----------
        all_benign : bool
            If True, return clusters with no outliers (all benign)
            If False, return clusters with some outliers (some poisoned)
        """
        # Create mock cluster labels
        n_samples = len(y_train)

        if all_benign:
            # No outliers - all samples belong to valid clusters (benign case)
            class_cluster_labels = np.zeros(n_samples)
            # Assign different cluster labels based on the class
            for i, label in enumerate(unique_classes):
                class_cluster_labels[y_train == label] = i
        else:
            # Some outliers (poisoned case) - outliers are marked with -1
            class_cluster_labels = np.zeros(n_samples)
            # Assign different cluster labels based on the class
            for i, label in enumerate(unique_classes):
                class_cluster_labels[y_train == label] = i

            # Mark 10% of samples as outliers/poisoned
            poison_indices = np.random.choice(n_samples, size=n_samples//10, replace=False)
            class_cluster_labels[poison_indices] = -1

        # Create cluster-to-class mapping
        cluster_class_mapping = {}
        for label in unique_classes:
            unique_clusters = np.unique(class_cluster_labels[y_train == label])
            unique_clusters = unique_clusters[unique_clusters != -1]  # Exclude outliers
            for cluster in unique_clusters:
                cluster_class_mapping[cluster] = label

        return class_cluster_labels, cluster_class_mapping

    def test_detect_poison_all_benign(self):
        """
        Test the detect_poison method when all data is benign (no poisoned samples).
        This tests the true negative case.
        """
        # Create the defence with mocked methods
        defence = ClusteringCentroidAnalysis(
            classifier=self.mock_classifier,
            x_train=self.x_train,
            y_train=self.y_train,
            benign_indices=self.benign_indices,
            final_feature_layer_name='hidden_layer',
            misclassification_threshold=0.1
        )

        # Mock the _calculate_misclassification_rate method to return low rates (all benign)
        defence._calculate_misclassification_rate = MagicMock(return_value=0.05)

        # Call detect_poison with our mocked _cluster_classes returning no outliers
        with patch('art.defences.detector.poison.clustering_centroid_analysis._cluster_classes',
                   side_effect=lambda y, u, f, c: self._mock_cluster_classes(y, u, f, c, all_benign=True)):
            report, is_clean = defence.detect_poison()

        self.assertIsInstance(report, dict)
        self.assertEqual(len(self.y_train), len(is_clean[is_clean == 1]))
        # In the all-benign case, no samples should be marked as poisoned
        self.assertEqual(np.sum(is_clean), len(self.y_train))

    def test_detect_poison_with_poisoned_samples_as_outliers(self):
        """
        Test the detect_poison method when some data points are outliers (poisoned).
        This tests detection of poisoned samples as outliers.
        """
        # Create the defence with mocked methods
        defence = ClusteringCentroidAnalysis(
            classifier=self.mock_classifier,
            x_train=self.x_train,
            y_train=self.y_train,
            benign_indices=self.benign_indices,
            final_feature_layer_name='hidden_layer',
            misclassification_threshold=0.1
        )

        # Mock the _calculate_misclassification_rate method to return low rates (all benign)
        defence._calculate_misclassification_rate = MagicMock(return_value=0.05)

        # Call detect_poison with our mocked _cluster_classes returning some outliers
        with patch('art.defences.detector.poison.clustering_centroid_analysis._cluster_classes',
                   side_effect=lambda y, u, f, c: self._mock_cluster_classes(y, u, f, c, all_benign=False)):
            report, is_clean = defence.detect_poison()

        # Assertions
        self.assertIsInstance(report, dict)

        # Some samples should be marked as poisoned (0). FIXME: not robust enough
        self.assertLess(np.sum(is_clean), len(self.y_train))

    def test_detect_poison_with_high_misclassification_rate(self):
        """
        Test the detect_poison method when some clusters have high misclassification rates.
        This tests detection of poisoned samples based on misclassification rates.
        """
        # Create the defence with mocked methods
        defence = ClusteringCentroidAnalysis(
            classifier=self.mock_classifier,
            x_train=self.x_train,
            y_train=self.y_train,
            benign_indices=self.benign_indices,
            final_feature_layer_name='hidden_layer',
            misclassification_threshold=0.1
        )

        # Mock _calculate_misclassification_rate to return high rates for some clusters
        # This simulates a backdoor attack where the deviation causes high misclassification
        def mock_misclass_rate(class_label, deviation):
            # Return a high misclassification rate for class 0, low for class 1
            if class_label == 0:
                return 0.95  # Above threshold (1 - 0.1 = 0.9)
            else:
                return 0.05  # Below threshold

        defence._calculate_misclassification_rate = MagicMock(side_effect=mock_misclass_rate)

        # Call detect_poison with _cluster_classes returning clean clusters
        with patch('art.defences.detector.poison.clustering_centroid_analysis._cluster_classes',
                   side_effect=lambda y, u, f, c: self._mock_cluster_classes(y, u, f, c, all_benign=True)):
            report, is_clean = defence.detect_poison()

        # all elements in class 0 are poisoned. No outliers --> all poisoned elements are class 0
        np.testing.assert_equal(np.where(is_clean == 0), np.where(self.y_train == 0))

    def test_detect_poison_both_mechanisms(self):
        """
        Test the detect_poison method when detection happens through both mechanisms:
        1. Outliers detection
        2. High misclassification rates
        """
        # Create the defence with mocked methods
        defence = ClusteringCentroidAnalysis(
            classifier=self.mock_classifier,
            x_train=self.x_train,
            y_train=self.y_train,
            benign_indices=self.benign_indices,
            final_feature_layer_name='hidden_layer',
            misclassification_threshold=0.1
        )

        # Mock _calculate_misclassification_rate to return high rates for some clusters
        def mock_misclass_rate(class_label, deviation):
            # Only cluster 1 has high misclassification rate
            if class_label == 1:
                return 0.95  # Above threshold (1 - 0.1 = 0.9)
            else:
                return 0.05  # Below threshold

        defence._calculate_misclassification_rate = MagicMock(side_effect=mock_misclass_rate)

        # Call detect_poison with _cluster_classes returning some outliers
        with patch('art.defences.detector.poison.clustering_centroid_analysis._cluster_classes',
                   side_effect=lambda y, u, f, c: self._mock_cluster_classes(y, u, f, c, all_benign=False)):
            report, is_clean = defence.detect_poison()

        self.assertIsInstance(report, dict)
        # all elements in class 1 are poisoned
        self.assertGreater(len(is_clean[is_clean == 0]), len(self.y_train[self.y_train == 1]))
        self.assertTrue(np.all(is_clean[np.where(self.y_train == 1)] == 0))
        # most elements in class 0 are detected as clean. Poisoned ones are outliers. FIXME: can I make this more robust?
        self.assertLess(np.mean(self.y_train[np.where(is_clean == 1)]), 0.2)


if __name__ == "__main__":
    unittest.main()
