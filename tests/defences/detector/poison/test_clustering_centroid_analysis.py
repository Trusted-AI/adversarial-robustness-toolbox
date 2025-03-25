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

from sklearn.cluster import DBSCAN
from sklearn.decomposition import FastICA, PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from tensorflow.python.keras import Model, Sequential
from tensorflow.python.keras.layers import Dense
from umap import UMAP

from art.defences.detector.poison.clustering_centroid_analysis import get_reducer, get_scaler, get_clusterer, \
    ClusteringCentroidAnalysis
from art.defences.detector.poison.utils import ReducerType, ScalerType, ClustererType

logger = logging.getLogger(__name__)


class TestClusteringCentroidAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        model = Sequential([
            Dense(10, input_shape=(5,), name="input_layer"),
            Dense(5, activation="relu", name="hidden_layer"),
            Dense(1, activation="sigmoid", name="output_layer")
        ])

        cls.clustering_centroid_analysis = ClusteringCentroidAnalysis(
            classifier=model,
            x_train=None,
            y_train=None,
            x_benign=None,
            y_benign=None
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
