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

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import logging
import unittest
from typing import Union

import numpy as np
import pandas as pd

from art.attacks.poisoning import PoisoningAttackBackdoor
from art.attacks.poisoning.perturbations import create_flip_perturbation
from art.estimators.classification import KerasClassifier
from art.utils import load_mnist, load_unsw_nb15
from sklearn.cluster import DBSCAN
from sklearn.decomposition import FastICA, PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from tensorflow.python.keras import Model, Sequential
from tensorflow.python.keras.layers import Dense, Flatten
from umap import UMAP

from art.defences.detector.poison.clustering_centroid_analysis import get_reducer, get_scaler, get_clusterer, \
    ClusteringCentroidAnalysis
from art.defences.detector.poison.utils import ReducerType, ScalerType, ClustererType
from tests.estimators.classification.test_jax import classifier

logger = logging.getLogger(__name__)


def _standarization(df, col):
    arr = df[col]
    arr = np.array(arr)
    df[col] = scaler.fit_transform()


def _create_mlp_model(input_dim: int) -> Model:

    # Define a small DNN with one hidden layer
    base_model = Sequential(name="mlp_poisoned", layers=[
        Dense(64, activation="relu", name="input_layer", input_shape=(input_dim,)), # FIXME: 28, 1, 1 to input_dim,
        Dense(64, activation="relu", name="hidden_layer"),
        Dense(1, activation="sigmoid", name="output_layer")
    ])
    base_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    return base_model


def train_art_keras_classifier(x_train: Union[pd.DataFrame, np.ndarray] , y_train: Union[pd.DataFrame, np.ndarray]) -> KerasClassifier:
    """Trains a KerasClassifier using the ART wrapper."""

    # Create the Keras model
    mlp_model = _create_mlp_model(x_train.shape[1])

    # Create the ART KerasClassifier wrapper
    mlp_classifier = KerasClassifier(model=mlp_model, use_logits=False)

    # Requires ndarrays, so the dataframes are transformed
    x_values = x_train.values if type(x_train) == pd.DataFrame else x_train
    y_values = y_train.values if type(y_train) == pd.DataFrame else y_train

    # Train the model
    mlp_classifier.fit(x_values, y_values, batch_size=5000, verbose=True)

    return mlp_classifier


class TestClusteringCentroidAnalysis(unittest.TestCase):

    @classmethod
    def _preprocess(cls, x_data: pd.DataFrame, y_data: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        """
        Preprocesses x_train and y_train in order to be used in the test model
        :param x_data: information used by the model
        :param y_data: target label for the model
        :return: (x_data, y_data)
        """
        # Features filtering
        x_features = ["dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes",
                      "sttl", "dttl", "sload", "dload", "sloss", "dloss" , "sjit", "djit",
                      "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat",
                      "trans_depth", "ct_srv_src", "ct_state_ttl", "ct_dst_ltm",
                      "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd",
                      "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports",
                      "smeansz", "dmeansz", "sintpkt", "dintpkt", "res_bdy_len" # These were changed from original implementation
                      # "rate" FIXME: this one is missing? Is the dataset wrong?
        ]
        x_filtered = x_data[x_features]

        # Scaling
        x_copy = x_filtered.copy()
        y_copy = y_data.copy()
        x_numeric_cols = x_copy.select_dtypes(include='number')
        scaler = StandardScaler()

        for c in x_numeric_cols:
            arr = np.array(x_copy[c])
            x_copy[c] = scaler.fit_transform(arr.reshape(len(arr), 1))

        # One-hot encoding
        x_copy = pd.get_dummies(x_copy, columns=["proto", "service", "state"], prefix="", prefix_sep="")
        y_copy.rename(columns={"label": "intrusion"}, inplace=True)

        return x_copy, y_copy

    @classmethod
    def setUpClass(cls):
        """
        Sets up a shared model, a poisoned dataset, and a ClusteringCentroidAnalysis instance for all tests.
        """
        (x_train, y_train), (x_test, y_test) = load_unsw_nb15(frac=0.01, )

        x_train, y_train = cls._preprocess(x_train, y_train)

        cls.x_benign = x_train[:500]
        cls.y_benign = y_train[:500]
        cls.x_test = x_test[:100]
        cls.y_test = y_test[:100]

        # Define and apply a backdoor poisoning attack
        backdoor = PoisoningAttackBackdoor([
            create_flip_perturbation([1], poison_percentage=0.3)
        ])

        # Poisons the "label" column
        cls.y_poisoned, _ = backdoor.poison(y_train["intrusion"].values, np.ndarray([1]))
        cls.x_poisoned = x_train.copy()

        # Wrap the model in an ART classifier and train on poisoned data
        cls.classifier = train_art_keras_classifier(cls.x_poisoned, cls.y_poisoned)
        cls.classifier.fit(cls.x_poisoned, cls.y_poisoned, batch_size=32, nb_epochs=5)

        cls.clustering_centroid_analysis = ClusteringCentroidAnalysis(
            classifier=cls.classifier,
            x_train=cls.x_poisoned,
            y_train=cls.y_poisoned,
            x_benign=cls.x_benign,
            y_benign=cls.y_test,
        )

    def test_extract_classifier_layer_valid(self):
        """
        Tests that the ``_extract_classifier_layer`` method extracts the desired layer correctly
        """

        extractor = self.clustering_centroid_analysis._extract_classifier_layer("hidden_layer")
        self.assertIsInstance(extractor, Model)

    def test_extract_classifier_layer_invalid(self):
        """
        Tests that the ``_extract_classifier_layer`` throws a ``ValueError`` if an invalid layer is attempted to be extracted
        """

        with self.assertRaises(ValueError):
            self.clustering_centroid_analysis._extract_classifier_layer("invalid_layer")


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
