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
"""
TODO

"""
from __future__ import absolute_import, division, print_function, unicode_literals, annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import FastICA, PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from tensorflow.python.keras import Model, Input
from umap import UMAP

from art.defences.detector.poison.poison_filtering_defence import PoisonFilteringDefence
from art.defences.detector.poison.utils import ReducerType, ScalerType, ClustererType

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_NEURALNETWORK_TYPE

logger = logging.getLogger(__name__)


class ClusteringCentroidAnalysis(PoisonFilteringDefence):
    """
    Method from Guo et al., 2021, to perform poisoning detection using density-based clustering and centroids analysis.
    This universal detection method is intended for backdoor attacks.

    | Original paper link: https://arxiv.org/abs/2301.04554

    | Implementation and experimentation: https://hdl.handle.net/1992/75346

    """

    _DEFENCE_PARAMS = []
    _VALID_CLUSTERING = ["DBSCAN"]
    _VALID_REDUCE = ["UMAP", "PCA"]
    _VALID_ANALYSIS = []
    _MISSCLASSIFICATION_THRESHOLD = 0.95  # Umbral para considerar un cluster como envenenado

    _BENIGN_SAMPLING_SIZE = 100

    def __init__(
            self,
            classifier,
            x_train: np.ndarray,
            y_train: np.ndarray,
            x_benign: np.ndarray,
            y_benign: np.ndarray
    ):
        """
        Creates a :class: `ClusteringCentroidAnalysis` object for the given classifier

        :param classifier: model evaluated for poison
        :param x_train: dataset used to train the classifier (might be poisoned)
        :param y_train: labels used to train the classifier (might be poisoned)
        :param x_benign: subset of the training data known to be benign
        :param y_benign: subset of the labels known to be benign
        """
        super().__init__(classifier, x_train, y_train)
        self.x_benign = x_benign
        self.y_benign = y_benign
        self.reducer = get_reducer(ReducerType.UMAP, nb_dims=2)
        self.scaler = get_scaler(ScalerType.STANDARD)
        self.clusterer = get_clusterer(ClustererType.DBSCAN)

    def evaluate_defence(self, is_clean: np.ndarray, **kwargs) -> str:
        pass


    def _extract_classifier_layer(self, layer_name: str) -> Model:
        """
        Extracts the selected layer of the model. Can be used to create an intermediate model
        :param layer_name: layer to be extracted
        :return: Model that receives the base inputs and outputs up to the selected layer
        """
        return Model(inputs=self.classifier.layers[0].input, outputs=self.classifier.get_layer(layer_name).output)


    # FIXME: optimize this
    def _find_centroids(self, features_scaled, dbscan_labels):
        unique_clusters = set(dbscan_labels) - {-1}
        centroids = dict()
        for cluster_label in unique_clusters:
            cluster_indices = np.where(dbscan_labels == cluster_label)[0]
            cluster_features = features_scaled[cluster_indices]
            centroid = np.mean(cluster_features, axis=0)
            centroids[cluster_label] = centroid

        return centroids

    def _calculate_centroid_deviations(self, benign_centroid, centroids):
        return { label: centroid - benign_centroid for label, centroid in centroids.items() }

    def _get_benign_split(self, size) -> tuple[pd.DataFrame | np.ndarray, pd.DataFrame | np.ndarray]:
        """
        Retrieves a random sample of the benign dataset
        :param size: number of benign samples to retrieve
        :return: ``x_sample``, ``y_sample``
        :exception: raises ``ValueError`` if the size of the sample is bigger than the benign dataset
        """
        if size > self.x_benign.shape[0]:
            raise ValueError(f"Requested size ({size}) exceeds available benign samples ({self.x_benign.shape[0]})")

        indices = np.random.choice(self.x_benign.shape[0], size, replace=False)

        if isinstance(self.x_benign, pd.DataFrame):
            x_benign_sample = self.x_benign.iloc[indices]
        else:
            x_benign_sample = self.x_benign[indices]

        if isinstance(self.y_benign, pd.DataFrame):
            y_benign_sample = self.y_benign.iloc[indices]
        else:
            y_benign_sample = self.y_benign[indices]

        return x_benign_sample, y_benign_sample

    def _split_benign_data(
            self,
            test_size: float = 0.2,
            random_state: int = 42
    ) -> tuple[
        tuple[pd.DataFrame | np.ndarray, pd.DataFrame | np.ndarray],
        tuple[pd.DataFrame | np.ndarray, pd.DataFrame | np.ndarray]
    ]:
        """
        Splits all benign data into validation and misclassification sets.
        :param test_size: Proportion of benign data to reserve for misclassification checks (default 20%).
        :param random_state: Seed for reproducibility.
        :return: ((x_validation, y_validation), (x_misclassification, y_misclassification))
        """

        # Split the data into validation and misclassification sets
        x_validation, x_misclassification, y_validation, y_misclassification = train_test_split(
            self.x_benign,
            self.y_benign,
            test_size=test_size,
            random_state=random_state
        )

        return (x_validation, y_validation), (x_misclassification, y_misclassification)


    def detect_poison(self, **kwargs) -> tuple[dict, list[int]]:

        # 1. Dimensionality reduction and feature clustering

        # Extract features from a middle layer
        intermediate_layer = self._extract_classifier_layer("dense_2")
        features = intermediate_layer.predict(self.x_train)

        # Scale characteristics (?)
        features_scaled = self.scaler.fit_transform(features)

        # Reduce dimensionality
        features_reduced = self.reducer.fit_transform(features_scaled)

        # Apply clustering
        dbscan_labels = self.clusterer.fit_predict(features_reduced)

        # 2. Poisoned Cluster Detection

        # Compute cluster centroids
        centroids = self._find_centroids(features_scaled, dbscan_labels)

        # Find centroid deviation from benign centroid
        # TODO: this could be precalculated and cached
        (x_benign_validation, y_benign_validation), (x_benign_other, y_benign_other) = self._split_benign_data()

        benign_features = intermediate_layer.predict(x_benign_validation)
        benign_features_scaled = self.scaler.transform(benign_features) # TODO: fit_transform or transform?
        benign_centroid = np.mean(benign_features_scaled, axis=0)

        deviations = self._calculate_centroid_deviations(benign_centroid, centroids)

        # Missclassification check

        # Extract features from benign center
        other_features = intermediate_layer.predict(x_benign_other)
        other_features_scaled = self.scaler.transform(other_features)

        poisoned_clusters = []

        modified_input = Input(shape=(64,)) # FIXME: why 64?
        modified_output = self._extract_classifier_layer("dense_output")(modified_input)
        modified_model = Model(inputs=modified_input, outputs=modified_output)

        for cluster_label, deviation in deviations.items():
            # Añadir la desviación a las características de las otras muestras benignas
            modified_features = other_features_scaled + deviation
            # Obtener predicciones utilizando el modelo modificado
            predictions = modified_model.predict(modified_features)
            predicted_classes = (predictions > 0.5).astype(int).flatten()
            misclassification_ratio = np.mean(predicted_classes == 1)  # 1 es la clase objetivo (ataque)
            print(
                f"Cluster {cluster_label} tiene una tasa de misclasificación hacia la clase objetivo 1: {misclassification_ratio:.2f}")
            if misclassification_ratio >= self._MISSCLASSIFICATION_THRESHOLD:
                poisoned_clusters.append(cluster_label)

        # Paso 11: Identificar los índices de las muestras envenenadas detectadas
        detected_poisoned_indices = []
        for cluster_label in poisoned_clusters:
            cluster_indices = np.where(dbscan_labels == cluster_label)[0]
            detected_poisoned_indices.extend(cluster_indices)

        # Incluir los outliers detectados por DBSCAN
        outlier_indices = np.where(dbscan_labels == -1)[0]
        detected_poisoned_indices.extend(outlier_indices)


        #report = None
        #return report, self._is_clean_lst
        return detected_poisoned_indices


def get_reducer(reduce: ReducerType, nb_dims: int):
    """Initialize the right reducer based on the selected type."""
    if reduce == ReducerType.FASTICA:
        return FastICA(n_components=nb_dims, max_iter=1000, tol=0.005)
    if reduce == ReducerType.PCA:
        return PCA(n_components=nb_dims)
    if reduce == ReducerType.UMAP:
        return UMAP(n_components=nb_dims, random_state=42)  # TODO: should I remove the random state?

    raise ValueError(f"{reduce} dimensionality reduction method not supported.")


def get_scaler(scaler_type: ScalerType):
    """Initialize the right scaler based on the selected type."""
    if scaler_type == ScalerType.STANDARD:
        return StandardScaler()
    if scaler_type == ScalerType.MINMAX:
        return MinMaxScaler()
    if scaler_type == ScalerType.ROBUST:
        return RobustScaler()

    raise ValueError(f"{scaler_type} scaling method not supported.")


def get_clusterer(clusterer_type: ClustererType):
    """Initialize the right cluster algorithm (a.k.a., clusterer) based on the selected type. """
    if clusterer_type == ClustererType.DBSCAN:
        return DBSCAN(eps=0.5, min_samples=5)

    raise ValueError(f"{clusterer_type} cluster method not supported.")
