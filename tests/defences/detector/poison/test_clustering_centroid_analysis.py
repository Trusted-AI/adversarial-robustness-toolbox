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

from unittest.mock import MagicMock

import tensorflow as tf
from sklearn.base import ClusterMixin
from sklearn.compose import ColumnTransformer
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.metrics import Precision, Recall, Accuracy, AUC

tf.compat.v1.disable_eager_execution()

import logging
import unittest
from typing import Union

import numpy as np
import pandas as pd

from art.attacks.poisoning import PoisoningAttackBackdoor
from art.attacks.poisoning.perturbations import create_flip_perturbation
from art.estimators.classification import KerasClassifier
from art.utils import load_unsw_nb15
from sklearn.cluster import DBSCAN
from sklearn.decomposition import FastICA, PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from tensorflow.python.keras import Model, Sequential, Input
from tensorflow.python.keras.layers import Dense
from umap import UMAP

from art.defences.detector.poison.clustering_centroid_analysis import get_reducer, get_scaler, get_clusterer, \
    ClusteringCentroidAnalysis, _calculate_centroid, _class_clustering
from art.defences.detector.poison.utils import ReducerType, ScalerType, ClustererType

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
    mlp_classifier = KerasClassifier(model=mlp_model, use_logits=False)

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

    def fit_predict(self, X):
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
        self.y_train = np.random.randint(0, 2, size=(100, 1))  # For binary classification

        # Compile and train the model
        self.mock_model.compile(optimizer='adam', loss='binary_crossentropy')
        self.mock_model.fit(self.x_train, self.y_train, epochs=1) # Train for a few steps

        self.mock_classifier = KerasClassifier(model=self.mock_model, use_logits=False)

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

        self.assertIsInstance(feature_model, Model)
        self.assertIsInstance(classify_model, Model)
        self.assertEqual(feature_model.name, 'feature_representation_model') # Check the name.
        self.assertEqual(classify_model.name, 'classifying_submodel')

        sample_input = np.random.rand(1, 10)
        sample_output = self.mock_classifier.model.predict(sample_input)

        feature_value = feature_model.predict(sample_input)
        self.assertEqual(classify_model.input_shape[1], feature_value.shape[1])

        final_value = classify_model.predict(feature_value)
        self.assertEqual(sample_output, final_value)


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
        self.assertEqual(cca.classifier, self.mock_classifier)
        self.assertTrue(np.array_equal(cca.x_train, self.x_train))
        self.assertTrue(np.array_equal(cca.y_train, self.y_train))
        self.assertTrue(np.array_equal(cca.benign_indices, self.benign_indices))
        self.assertEqual(cca.misclassification_threshold, self.misclassification_threshold)
        self.is_valid_reducer(cca.reducer)
        self.is_valid_scaler(cca.scaler)
        self.is_valid_clusterer(cca.clusterer)
        self.assertTrue(np.array_equal(cca.x_benign, self.x_train[[0, 2]]))
        self.assertTrue(np.array_equal(cca.y_benign, self.y_train[[0, 2]]))
        self.assertIsInstance(cca.feature_representation_model, Model)
        self.assertIsInstance(cca.classifying_submodel, Model)
        self.assertEqual(cca.unique_classes, set([0, 1]))

    def test_init_empty_benign_indices(self):
        """Test __init__ with empty benign indices."""
        with self.assertRaises(ValueError) as e:
            cca = ClusteringCentroidAnalysis(
                classifier=self.mock_classifier,
                x_train=self.x_train,
                y_train=self.y_train,
                benign_indices=np.array([]),
                final_feature_layer_name=self.final_feature_layer_name,
                misclassification_threshold=self.misclassification_threshold
            )
            self.assertEqual(str(e.exception), 'Benign indices passed (0) are not enough to run the algorithm')

    def test_init_invalid_layer_name(self):
        """Test __init__ with an invalid layer name. Check that it raises error."""
        with self.assertRaises(ValueError):
            ClusteringCentroidAnalysis(
                classifier=self.mock_classifier,
                x_train=self.x_train,
                y_train=self.y_train,
                benign_indices=self.benign_indices,
                final_feature_layer_name='invalid_layer',
                misclassification_threshold=self.misclassification_threshold
            )

@unittest.skip("Changes were made")
class TestClusteringCentroidAnalysis(unittest.TestCase):

    _FEATURES = ["dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes",
                 "sttl", "dttl", "sload", "dload", "sloss", "dloss" , "sjit", "djit",
                 "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat",
                 "trans_depth", "ct_srv_src", "ct_state_ttl", "ct_dst_ltm",
                 "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd",
                 "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports",
                 "smeansz", "dmeansz", "sintpkt", "dintpkt", "res_bdy_len" # These were changed from original implementation
                 # "rate" FIXME: this one is missing? Is the dataset wrong?
                 ]
    _CATEGORICAL_COLS = ["proto", "state", "service"]

    @classmethod
    def _preprocess_train(cls, x_data: pd.DataFrame, y_data: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        """
        Preprocesses x_train and y_train to be used in the test model
        :param x_data: information used by the model
        :param y_data: target label for the model
        :return: (x_data, y_data)
        """
        # Features filtering
        x_filtered = x_data[cls._FEATURES].copy()
        y_filtered = y_data.copy()
        y_filtered.rename(columns={"label": "intrusion"}, inplace=True)

        numeric_cols = x_filtered.select_dtypes(include='number').columns

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cls._CATEGORICAL_COLS),
            ],
            remainder="passthrough"
        )

        preprocessor.fit(x_filtered)
        x_processed = preprocessor.transform(x_filtered)

        feature_names = list(numeric_cols) + list(preprocessor.named_transformers_['cat'].get_feature_names_out(cls._CATEGORICAL_COLS))
        x_processed_df = pd.DataFrame(x_processed, columns=feature_names, index=x_filtered.index)

        return x_processed_df, y_filtered, preprocessor

    @classmethod
    def _preprocess_test(cls, x_data: pd.DataFrame, y_data: pd.DataFrame, fitted_preprocessor: ColumnTransformer) -> (pd.DataFrame, pd.DataFrame):
        """
        Preprocesses x_train and y_train to be used in the test model
        :param x_data: information used by the model
        :param y_data: target label for the model
        :param fitted_preprocessor: ColumnTransformer used to fit the preprocessor in the train data preprocessing
        :return: (x_data, y_data)
        """
        # Features filtering
        x_filtered = x_data[cls._FEATURES].copy()
        y_filtered = y_data.copy()
        y_filtered.rename(columns={"label": "intrusion"}, inplace=True)

        x_processed = fitted_preprocessor.transform(x_filtered)
        numeric_cols = x_filtered.select_dtypes(include='number').columns
        feature_names = list(numeric_cols) + list(fitted_preprocessor.named_transformers_['cat'].get_feature_names_out(cls._CATEGORICAL_COLS))

        x_processed_df = pd.DataFrame(x_processed, columns=feature_names, index=x_filtered.index)

        return x_processed_df, y_filtered

    @classmethod
    def setUpClass(cls):
        """
        Sets up a shared model, a poisoned dataset, and a ClusteringCentroidAnalysis instance for all tests.
        """
        # Define and apply a backdoor poisoning attack
        backdoor = PoisoningAttackBackdoor([
            create_flip_perturbation([1],0.5)
        ])

        (x_train, y_train), (x_test, y_test) = load_unsw_nb15(frac=0.01, )

        x_train, y_train, fitted_preprocessor = cls._preprocess_train(x_train, y_train)
        x_test, y_test = cls._preprocess_test(x_test, y_test, fitted_preprocessor)

        cls.x_benign = x_train[:500].copy()
        cls.y_benign = y_train[:500].copy()

        # Clean samples, used for negative test results
        cls.x_clean = x_train[500:].copy()
        cls.y_clean = y_train[500:].copy()

        # Poisons the "label" column
        cls.x_poisoned = x_train[500:].copy()
        cls.y_poisoned, _ = backdoor.poison(y_train[500:]["intrusion"].values, np.ndarray([1]))

        is_poisoned = 0
        for i in range(len(cls.y_poisoned)):
            if not cls.y_poisoned[i] and cls.y_clean.iloc[i]['intrusion']:
                is_poisoned += 1

        logger.info(f"x_train:\t{x_train.shape}")
        logger.info(f"y_train:\t{y_train.shape}")
        logger.info(f"x_test:\t{x_test.shape}")
        logger.info(f"y_test:\t{y_test.shape}")
        logger.info(f"x_poisoned:\t{cls.x_poisoned.shape}")
        logger.info(f"y_poisoned:\t{cls.y_poisoned.shape}")
        logger.info(f"x_clean:\t{cls.x_clean.shape}")
        logger.info(f"y_clean:\t{cls.y_clean.shape}")
        logger.info(f"Poisoned entries:\t{is_poisoned}")

        # Wrap the model in an ART classifier and train on poisoned data
        cls.poisoned_classifier = train_art_keras_classifier(cls.x_poisoned, cls.y_poisoned, "mlp_poisoned")
        results = cls.poisoned_classifier.model.evaluate(x_test, y_test, verbose=1)
        metrics = cls.poisoned_classifier.model.metrics_names

        for name, value in zip(metrics, results):
            logger.info(f"{name.capitalize():<10}: {value:.4f}")

        cls.clean_classifier = train_art_keras_classifier(cls.x_clean, cls.y_clean, "mlp_clean")
        results = cls.clean_classifier.model.evaluate(x_test, y_test, verbose=1)
        metrics = cls.clean_classifier.model.metrics_names

        for name, value in zip(metrics, results):
            logger.info(f"{name.capitalize():<10}: {value:.4f}")

        # Represents a poisoned scenario
        cls.clustering_centroid_analysis_poisoned = ClusteringCentroidAnalysis(
            classifier=cls.poisoned_classifier,
            x_train=cls.x_poisoned, # FIXME: should be an ndarray?
            y_train=cls.y_poisoned,
            x_benign=cls.x_benign, # FIXME: should be an ndarray?
            y_benign=cls.y_benign, # FIXME: should be an ndarray?
        )

        # Represents a non-poisoned (clean) scenario
        cls.clustering_centroid_analysis_clean = ClusteringCentroidAnalysis(
            classifier=cls.clean_classifier,
            x_train=cls.x_clean, # FIXME: should be an ndarray?
            y_train=cls.y_clean,
            x_benign=cls.x_benign, # FIXME: should be an ndarray?
            y_benign=cls.y_benign, # FIXME: should be an ndarray?
        )

    def test_extract_classifier_layer_valid(self):
        """
        Tests that the ``_extract_classifier_layer`` method extracts the desired layer correctly
        """

        extractor = self.clustering_centroid_analysis_poisoned._extract_classifier_layer("hidden_layer")
        self.assertIsInstance(extractor, Layer)

    def test_extract_classifier_layer_invalid(self):
        """
        Tests that the ``_extract_classifier_layer`` throws a ``ValueError`` if an invalid layer is attempted to be extracted
        """

        with self.assertRaises(ValueError):
            self.clustering_centroid_analysis_poisoned._extract_classifier_layer("invalid_layer")

    def test_find_centroids_basic(self):
        """
        Tests that the ``_find_centroids`` method finds the desired centroids
        :return:
        """
        dbscan_labels = np.array([0, 0, 1, 1, 2, 2])
        features_scaled = np.array([
            [1, 2],
            [1.5, 2.5],
            [5, 6],
            [5.5, 6.5],
            [9, 10],
            [9.5, 10.5]
        ])
        expected_centroids = {
            0: np.array([1.25, 2.25]),
            1: np.array([5.25, 6.25]),
            2: np.array([9.25, 10.25])
        }
        centroids = self.clustering_centroid_analysis_poisoned._find_centroids(features_scaled, dbscan_labels)
        self.assertEqual(len(centroids), len(expected_centroids))
        for label, expected_centroid in expected_centroids.items():
            np.testing.assert_allclose(centroids[label], expected_centroid)

    def test_find_centroids_with_noise(self):
        dbscan_labels = np.array([0, 0, -1, 1, 1, -1, 2, 2])
        features_scaled = np.array([
            [1, 2],
            [1.5, 2.5],
            [0, 0],  # Noise
            [5, 6],
            [5.5, 6.5],
            [10, 10], # Noise
            [9, 10],
            [9.5, 10.5]
        ])
        expected_centroids = {
            0: np.array([1.25, 2.25]),
            1: np.array([5.25, 6.25]),
            2: np.array([9.25, 10.25])
        }
        centroids = self.clustering_centroid_analysis_poisoned._find_centroids(features_scaled, dbscan_labels)
        self.assertEqual(len(centroids), len(expected_centroids))
        for label, expected_centroid in expected_centroids.items():
            np.testing.assert_allclose(centroids[label], expected_centroid)

    def test_empty_input_dict_output(self):
        dbscan_labels = np.array([])
        features_scaled = np.array([])
        expected_centroids = {}
        centroids = self.clustering_centroid_analysis_poisoned._find_centroids(features_scaled, dbscan_labels)
        self.assertEqual(centroids, expected_centroids)

    def test_data_type_and_shape_dict_output(self):
        dbscan_labels = np.array([0, 0, 1, 1])
        features_scaled = np.array([[1.0, 2.0], [1.5, 2.5], [5.0, 6.0], [5.5, 6.5]], dtype=np.float32)
        centroids = self.clustering_centroid_analysis_poisoned._find_centroids(features_scaled, dbscan_labels)
        self.assertEqual(len(centroids), 2)
        for label, centroid in centroids.items():
            self.assertEqual(centroid.dtype, np.float32)
            self.assertEqual(centroid.shape, (2,)) # Assuming 2 features


    def test_split_benign_data_basic_split(self):
        (x_val, y_val), (x_mis, y_mis) = self.clustering_centroid_analysis_poisoned._split_benign_data()
        self.assertIsInstance(x_val, np.ndarray | pd.DataFrame)
        self.assertIsInstance(y_val, np.ndarray | pd.DataFrame)
        self.assertIsInstance(x_mis, np.ndarray | pd.DataFrame)
        self.assertIsInstance(y_mis, np.ndarray | pd.DataFrame)

    def test_split_benign_data_randomness(self):
        # FIXME: should I do more robust checking for randomness? Although I set random seeds and it's another library's responsibility...
        (x_val_1, y_val_1), (x_mis_1, y_mis_1) = self.clustering_centroid_analysis_poisoned._split_benign_data(random_state=1)
        (x_val_2, y_val_2), (x_mis_2, y_mis_2) = self.clustering_centroid_analysis_poisoned._split_benign_data(random_state=2)

        with self.assertRaises(AssertionError, msg="x_validation should be different with different random states"):
            pd.testing.assert_index_equal(x_val_1.index, x_val_2.index),
        with self.assertRaises(AssertionError, msg="y_validation should be different with different random states"):
            pd.testing.assert_index_equal(y_val_1.index, y_val_2.index),
        with self.assertRaises(AssertionError, msg="x_validation should be different with different random states"):
            pd.testing.assert_index_equal(x_mis_1.index, x_mis_2.index),
        with self.assertRaises(AssertionError, msg="y_validation should be different with different random states"):
            pd.testing.assert_index_equal(y_mis_1.index, y_mis_2.index)

    def test_detect_poison_poisoned(self):
        report, poisoned_indices = self.clustering_centroid_analysis_poisoned.detect_poison()

        self.assertIsInstance(report, dict)

        # all poisoned elements are detected
        self.assertEqual(0, len(poisoned_indices))

        # all poisoned clusters are detected


    def test_detect_poison_clean(self):
        report, poisoned_indices = self.clustering_centroid_analysis_clean.detect_poison()

        self.assertIsInstance(report, dict)

        # no poisoned elements are detected
        self.assertEqual(0, len(poisoned_indices))


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


class TestReducersScalersClusterers(unittest.TestCase):
    """
    Suite of tests for the valid and invalid utils used in :class: ``ClusteringCentroidAnalysis``
    """

    def test_get_reducer_valid(self):
        reducer_cases = [
            (ReducerType.FASTICA, FastICA),
            (ReducerType.PCA, PCA),
            (ReducerType.UMAP, UMAP),
        ]
        for reducer_type, expected in reducer_cases:
            with self.subTest(reducer=reducer_type):
                reducer = get_reducer(reducer_type, nb_dims=5)
                self.assertIsInstance(reducer, expected)

    def test_get_reducer_invalid(self):
        for invalid in ["INVALID", None]:
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    get_reducer(invalid, nb_dims=5)

    def test_get_scaler_valid(self):
        scaler_cases = [
            (ScalerType.STANDARD, StandardScaler),
            (ScalerType.MINMAX, MinMaxScaler),
            (ScalerType.ROBUST, RobustScaler),
        ]
        for scaler_type, expected in scaler_cases:
            with self.subTest(scaler=scaler_type):
                scaler = get_scaler(scaler_type)
                self.assertIsInstance(scaler, expected)

    def test_get_scaler_invalid(self):
        for invalid in ["INVALID", None]:
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    get_scaler(invalid)

    def test_get_clusterer_valid(self):
        clusterer_cases = [
            (ClustererType.DBSCAN, DBSCAN),
        ]
        for clusterer_type, expected in clusterer_cases:
            with self.subTest(clusterer=clusterer_type):
                clusterer = get_clusterer(clusterer_type)
                self.assertIsInstance(clusterer, expected)

    def test_get_clusterer_invalid(self):
        for invalid in ["INVALID", None]:
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    get_clusterer(invalid)


if __name__ == "__main__":
    unittest.main()
