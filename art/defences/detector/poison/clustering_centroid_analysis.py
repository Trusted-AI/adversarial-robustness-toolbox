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
from __future__ import absolute_import, division, print_function, unicode_literals, annotations

import logging
from typing import TYPE_CHECKING

import tensorflow as tf
import numpy as np
from tensorflow.keras import Model, Sequential

from art.defences.detector.poison.ground_truth_evaluator import GroundTruthEvaluator
from sklearn.base import ClusterMixin
from sklearn.cluster import DBSCAN
from sklearn.decomposition import FastICA, PCA
from umap import UMAP

from art.defences.detector.poison.clustering_analyzer import ClusterAnalysisType
from art.defences.detector.poison.poison_filtering_defence import PoisonFilteringDefence
from art.defences.detector.poison.utils import ReducerType, ClustererType

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_NEURALNETWORK_TYPE, CLASSIFIER_TYPE

logger = logging.getLogger(__name__)

def _encode_labels(y: np.array) -> (np.array, set, np.array, dict):
    """
    Given the target column, it generates the label encoding and the reverse mapping to use in the classification process

    :param y: 1D np.array with single values that represent the different classes
    :return: (y_encoded, unique_classes, label_mapping, reverse_mapping) encoded column, set of unique classes,
        mapping from class to numeric label, and mapping from numeric label to class
    """
    label_mapping = np.unique(y)
    reverse_mapping = {k: v for v, k in enumerate(label_mapping)}
    y_encoded = np.array([reverse_mapping[v] for v in y])
    unique_classes = set(reverse_mapping.values())
    return y_encoded, unique_classes, label_mapping, reverse_mapping

@tf.function(reduce_retracing=True)
def _calculate_centroid_tf(features):
    return tf.reduce_mean(features, axis=0)

def _calculate_centroid(selected_indices: np.ndarray, features: np.array) -> np.ndarray:
    """
    Returns the centroid of all data within a specific cluster that is classified as a specific class label

    :param selected_indices: a numpy array of selected indices on which to calculate the centroid
    :param features: numpy array d-dimensional features for all given data
    :return: d-dimensional numpy array
    """
    selected_features = features[selected_indices]
    features_tf = tf.convert_to_tensor(selected_features, dtype=tf.float32)
    centroid = _calculate_centroid_tf(features_tf)
    return centroid.numpy()

def _class_clustering(y: np.array, features: np.array, label: any, clusterer: ClusterMixin) -> (np.array, np.array):
    """
    Given a class label, it clusters all the feature representations that map to that class

    :param y: array of n class labels
    :param label: class label in the classification task
    :param features: numpy array d-dimensional features for n data entries
    :return: (cluster_labels, selected_indices) ndarrays of equal size with cluster labels and corresponding original indices.
    """
    logging.info(f"Clustering class {label}...")
    selected_indices = np.where(y == label)[0]
    selected_features = features[selected_indices]
    cluster_labels = clusterer.fit_predict(selected_features)
    return cluster_labels, selected_indices

def _feature_extraction(x_train: np.array, feature_representation_model: Model) -> np.ndarray:
    """
    Extract features from the model using the feature representation sub model.

    :param x_train: numpy array d-dimensional features for n data entries. Features are extracted from here
    :return: features. numpy array of features
    """
    # Convert data to TensorFlow tensors if needed
    if not isinstance(x_train, tf.Tensor):
        data = tf.convert_to_tensor(x_train, dtype=tf.float32)

    # Apply tf.function to the feature extraction for better performance
    @tf.function
    def extract_features(x):
        return feature_representation_model(x, training=False)

    # Process in batches to avoid memory issues
    batch_size = 256
    num_batches = int(np.ceil(len(data) / batch_size))
    features = []

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(data))
        batch = data[start_idx:end_idx]
        batch_features = extract_features(batch)
        features.append(batch_features)

    # Concatenate all batches
    if len(features) > 1:
        features = tf.concat(features, axis=0)
    else:
        features = features[0]

    return features.numpy()


def _cluster_classes(y_train: np.array, unique_classes: set[int], features: np.array, clusterer: ClusterMixin) -> (np.array, dict):
    """
    Clusters all the classes in the given dataset into uniquely identifiable clusters.

    :param features: feature representations' array of n rows
    :return: (class_cluster_labels, cluster_class_mapping)
    """
    # represents the number of clusters used up until now to differentiate clusters obtained in different
    # clustering runs by classes
    logging.info("Clustering classes...")
    used_cluster_labels = 0
    cluster_class_mapping  = dict()
    class_cluster_labels = np.full(len(y_train), -1)

    logging.info(f"Unique classes are: {unique_classes}")

    for class_label in unique_classes:
        cluster_labels, selected_indices = _class_clustering(y_train, features, class_label, clusterer)
        # label values are adjusted to account for labels of previous clustering tasks
        cluster_labels[cluster_labels != -1] += used_cluster_labels
        used_cluster_labels += len(np.unique(cluster_labels[cluster_labels != -1]))

        class_cluster_labels[selected_indices] = cluster_labels

        # the class (label) corresponding to the cluster is saved for centroid deviation calculation
        for l in np.unique(cluster_labels[cluster_labels != -1]):
            cluster_class_mapping[l] = class_label

    return class_cluster_labels, cluster_class_mapping


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
    _VALID_ANALYSIS = [
        ClusterAnalysisType.SMALLER,
        ClusterAnalysisType.RELATIVE_SIZE,
        ClusterAnalysisType.DISTANCE,
        ClusterAnalysisType.SILHOUETTE_SCORES
    ]

    def _get_benign_data(self) -> (np.ndarray, np.ndarray):
        """
        Retrieves the benign data from the training data using benign indices

        :return: (x_benign, y_benign) ndarrays with the benign data.
        """
        if len(self.benign_indices) == 0:
            raise ValueError(f"Benign indices passed ({len(self.benign_indices)}) are not enough to run the algorithm")


        return self.x_train[self.benign_indices], self.y_train[self.benign_indices]

    def _extract_submodels(self, final_feature_layer_name: str) -> (Model, Model):
        """
        Extracts the feature representation and final classifier submodels from the original classifier.
        Composition of both models should result in the original model

        :param final_feature_layer_name: Name of the final layer in feature abstraction. Should be a ReLu-activated layer
            as suggested in the paper.
        :return: (feature_representation_submodel, classifying_submodel)
        """
        logging.info("Extracting submodels...")
        keras_model = self.classifier.model

        try:
            final_feature_layer = keras_model.get_layer(name=final_feature_layer_name)
        except ValueError:
            raise ValueError(f"Layer with name '{final_feature_layer_name}' not found in the model.")

        if not hasattr(final_feature_layer, 'activation') or final_feature_layer.activation != tf.keras.activations.relu:
            raise ValueError(f"Final feature layer '{final_feature_layer_name}' must have a ReLU activation.")

        # Create a feature representation submodel with weight sharing
        feature_representation_model = Model(
            inputs=keras_model.inputs,
            outputs=keras_model.get_layer(final_feature_layer_name).output,
            name="feature_representation_model"
        )

        final_feature_layer_index = keras_model.layers.index(final_feature_layer)
        classifier_submodel_layers = keras_model.layers[final_feature_layer_index + 1:]

        # Create the classifier submodel
        classifying_submodel = Sequential(classifier_submodel_layers, name="classifying_submodel")

        intermediate_shape = feature_representation_model.output_shape[1:]
        dummy_input = tf.zeros((1,) + intermediate_shape)
        classifying_submodel(dummy_input)

        return feature_representation_model, classifying_submodel

    def get_clusters(self) -> np.array:
        """
        :return: np.array with m+1 columns, where m is dimensionality of the dimensionality reducer's output.
            m columns are used for feature representations and the last column is used for cluster label.
        """
        # Ensure features have been reduced and clustering has been performed
        if not hasattr(self, "features_reduced") or self.features_reduced is None:
            raise ValueError("Features have not been reduced yet. Run detect_poison first.")

        if not hasattr(self, "class_cluster_labels") or self.class_cluster_labels is None:
            raise ValueError("Clustering has not been performed yet. Run detect_poison first.")

        # Create the output array with features and cluster labels
        # The cluster labels should be added as the last column
        result = np.column_stack((self.features_reduced, self.class_cluster_labels))

        return result

    # TODO: MAP THE ENCODINGS
    # NP ARGMAX IN THE LAST LAYER
    def __init__(
            self,
            classifier: "CLASSIFIER_TYPE",
            x_train: np.ndarray,
            y_train: np.ndarray,
            benign_indices: np.array,
            final_feature_layer_name: str,
            misclassification_threshold: float,
            reducer = UMAP(n_components=2, random_state=42),
            clusterer = DBSCAN(eps=0.8, min_samples=20)
    ):
        """
        creates a :class: `ClusteringCentroidAnalysis` object for the given classifier

        :param classifier: model evaluated for poison
        :param x_train: dataset used to train the classifier (might be poisoned)
        :param y_train: labels used to train the classifier (might be poisoned)
        :param benign_indices: array data points' indices known to be benign
        :param final_feature_layer_name: the name of the final layer that builds feature representation. Must
            be a ReLu layer
        """
        logger.info("Loading variables into CCA...")
        super().__init__(classifier, x_train, y_train)
        self.reducer = reducer
        self.clusterer = clusterer
        self.benign_indices = benign_indices
        self.y_train, self.unique_classes, self.class_mapping, self.reverse_class_mapping = _encode_labels(y_train)

        self.x_benign, self.y_benign = self._get_benign_data()

        self.feature_representation_model, self.classifying_submodel = self._extract_submodels(final_feature_layer_name)

        self.misclassification_threshold = np.float64(misclassification_threshold)
        logger.info("CCA object created successfully.")

    def evaluate_defence(self, is_clean: np.ndarray, **kwargs) -> str:
        evaluator = GroundTruthEvaluator()

        # Segment predicted values by class
        assigned_clean_by_class = []
        for class_label in self.unique_classes:
            # Get indices for this class
            class_indices = np.where(self.y_train == class_label)[0]
            # Get assigned_clean values for those indices
            assigned_clean_by_class.append(self.is_clean[class_indices])

        # Segment ground truth by class
        is_clean_by_class = []
        for class_label in self.unique_classes:
            class_indices = np.where(self.y_train == class_label)[0]
            is_clean_by_class.append(is_clean[class_indices])

        # Create evaluator and analyze results
        errors_by_class, confusion_matrix_json = evaluator.analyze_correctness(
            assigned_clean_by_class=assigned_clean_by_class,
            is_clean_by_class=is_clean_by_class
        )

        return confusion_matrix_json


    def _calculate_misclassification_rate(self, class_label: int, deviation: np.array) -> np.float64:
        total_elements = 0
        misclassified_elements = 0

        for other_class_label in self.unique_classes - {class_label}:
            other_class_data = self.x_benign[self.y_benign == other_class_label]
            total_elements += len(other_class_data)

            deviated_features = self.feature_representation_model.predict(other_class_data) + deviation
            deviated_predictions = np.argmax(self.classifying_submodel.predict(deviated_features), axis=1)

            # how many elements of other_class_label are misclassified towards class label if its deviation is applied?
            misclassified_elements += len(deviated_predictions[deviated_predictions == class_label])

        return np.float64(misclassified_elements) / np.float64(total_elements)

    def detect_poison(self, **kwargs) -> (dict, list[int]):

        self.is_clean = np.ones(len(self.y_train))

        self.features = _feature_extraction(self.x_train, self.feature_representation_model)
        self.features_reduced = self.reducer.fit_transform(self.features)

        self.class_cluster_labels, self.cluster_class_mapping = _cluster_classes(self.y_train,
                                                                       self.unique_classes,
                                                                       self.features_reduced,
                                                                       self.clusterer)

        # outliers are poisoned
        outlier_indices = np.where(self.class_cluster_labels == -1)[0]
        self.is_clean[outlier_indices] = 0

        logging.info("Calculating real centroids...")
        real_centroids = dict()

        # for each cluster found for each target class
        for label in np.unique(self.class_cluster_labels[self.class_cluster_labels != -1]):
            real_centroids[label] = _calculate_centroid(np.where(self.class_cluster_labels == label)[0],
                                                        self.features)

        logging.info("Calculating benign centroids...")
        benign_centroids = dict()

        logger.info(f"Target classes are: {self.unique_classes}")

        # for each target class
        for class_label in self.unique_classes:
            benign_class_indices = np.intersect1d(self.benign_indices, np.where(self.y_train == class_label)[0])
            benign_centroids[class_label] = _calculate_centroid(benign_class_indices, #FIXME: this is wrong. y_benign has different dimensions and features extracted wont be the same
                                                                self.features)

        logging.info("Calculating misclassification rates...")
        misclassification_rates = dict()

        for cluster_label, centroid in real_centroids.items():
            class_label = self.cluster_class_mapping[cluster_label]
            # B^k_i
            deviation = centroid - benign_centroids[class_label]

            # MR^k_i
            # with unique cluster labels for each cluster in each clustering run, the label already maps to a target class
            misclassification_rates[cluster_label] = self._calculate_misclassification_rate(class_label, deviation)


        logging.info("Evaluating cluster misclassification...")
        for cluster_label, mr in misclassification_rates.items():
            if mr >= 1 - self.misclassification_threshold:
                cluster_indices = np.where(self.class_cluster_labels == cluster_label)[0]
                self.is_clean[cluster_indices] = 0

        return dict(), self.is_clean.copy()


def get_reducer(reduce: ReducerType, nb_dims: int):
    """Initialize the right reducer based on the selected type."""
    if reduce == ReducerType.FASTICA:
        return FastICA(n_components=nb_dims, max_iter=1000, tol=0.005)
    if reduce == ReducerType.PCA:
        return PCA(n_components=nb_dims)
    if reduce == ReducerType.UMAP:
        return UMAP(n_components=nb_dims, random_state=42)  # TODO: should I remove the random state?

    raise ValueError(f"{reduce} dimensionality reduction method not supported.")


def get_clusterer(clusterer_type: ClustererType) -> ClusterMixin:
    """Initialize the right cluster algorithm (a.k.a., clusterer) based on the selected type. """
    if clusterer_type == ClustererType.DBSCAN:
        return DBSCAN(eps=0.8, min_samples=20)

    raise ValueError(f"{clusterer_type} cluster method not supported.")