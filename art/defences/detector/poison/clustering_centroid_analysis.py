# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2025
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
from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
    annotations,
)

import logging
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from art.defences.detector.poison.ground_truth_evaluator import GroundTruthEvaluator
from art.defences.detector.poison.poison_filtering_defence import PoisonFilteringDefence

if TYPE_CHECKING:
    from tensorflow.keras import Model
    from umap import UMAP
    from sklearn.base import ClusterMixin
    from art.utils import CLASSIFIER_TYPE
    import tensorflow as tf_types
else:
    UMAP = Any
    ClusterMixin = Any
    Model = Any
    Sequential = Any
    tf_types = Any
    ClassifierType = Any


def _encode_labels(y: np.ndarray) -> tuple[np.ndarray, set, np.ndarray, dict]:
    """
    Given the target column, it generates the label encoding and the reverse mapping to use in the
    classification process

    :param y: 1D np.ndarray with single values that represent the different classes
    :return: (y_encoded, unique_classes, label_mapping, reverse_mapping) encoded column, set of unique classes,
        mapping from class to numeric label, and mapping from numeric label to class
    """
    label_mapping = np.unique(y)
    reverse_mapping = {k: v for v, k in enumerate(label_mapping)}
    y_encoded = np.array([reverse_mapping[v] for v in y])
    unique_classes = set(reverse_mapping.values())
    return y_encoded, unique_classes, label_mapping, reverse_mapping


class ClusteringCentroidAnalysisTensorFlowV2(PoisonFilteringDefence):
    """
    Method from Guo et al., 2021, to perform poisoning detection using density-based clustering and centroids analysis.
    This universal detection method is intended for backdoor attacks.

    | Original paper link: https://arxiv.org/abs/2301.04554

    | Implementation and experimentation: https://hdl.handle.net/1992/75346

    """

    defence_params = [
        "classifier",
        "x_train",
        "y_train",
        "benign_indices",
        "final_feature_layer_name",
        "misclassification_threshold",
        "reducer",
        "clsuterer",
    ]
    valid_clustering = ["DBSCAN"]
    valid_reduce = ["UMAP"]

    def __init__(
        self,
        classifier: CLASSIFIER_TYPE,
        x_train: np.ndarray,
        y_train: np.ndarray,
        benign_indices: np.ndarray,
        final_feature_layer_name: str,
        misclassification_threshold: float,
        reducer: UMAP | None = None,
        clusterer: ClusterMixin | None = None,
        feature_extraction_batch_size: int = 32,
        misclassification_batch_size: int = 32,
    ):
        """
        Creates a :class: `ClusteringCentroidAnalysis` object for the given classifier

        :param classifier: model evaluated for poison
        :param x_train: dataset used to train the classifier (might be poisoned)
        :param y_train: labels used to train the classifier (might be poisoned)
        :param benign_indices: array data points' indices known to be benign
        :param final_feature_layer_name: the name of the final layer that builds feature representation. Must
            be a ReLu layer
        :param misclassification_threshold: maximum misclassification threshold to consider a cluster as clean
        :param reducer: dimensionality reducer used to reduce feature space. UMAP is used as default
        :param clusterer: clustering algorithm used to cluster the features. DBSCAN is used as default
        :param feature_extraction_batch_size: batch size for feature extraction.
            Use lower values in case of low GPU memory availability
        :param misclassification_batch_size: batch size for misclassification.
            Use lower values in case of low GPU memory
        """
        # Tensorflow is imported dynamically to keep ART framework-agnostic
        try:
            import tensorflow as tf_runtime
            from tensorflow.keras import Model, Sequential

            # Keep a runtime and function attributes to prevent garbage collection on these during the
            # algorithm lifespan
            self._tf_runtime = tf_runtime
            self._KerasModel = Model
            self._KerasSequential = Sequential
            self._tf_runtime.get_logger().setLevel(logging.WARN)
        except ImportError as e:
            raise ImportError("TensorFlow is required for ClusteringCentroidAnalysis. ") from e

        if clusterer is None:
            try:
                from sklearn.base import ClusterMixin
                from sklearn.cluster import DBSCAN

                # Default clusterer, recommended by original authors
                self.clusterer = DBSCAN(eps=0.8, min_samples=20)
            except ImportError as e:
                raise ImportError(
                    "Scikit-learn is required for default clusterer in ClusteringCentroidAnalysis. "
                ) from e

        if reducer is None:
            try:
                from umap import UMAP

                # Default reducer, recommended by original authors
                self.reducer = UMAP(n_neighbors=5, min_dist=0)
            except ImportError as e:
                raise ImportError("UMAP is required for default reducer in ClusteringCentroidAnalysis. ") from e

        logging.info("Loading variables into CCA...")
        super().__init__(classifier, x_train, y_train)
        self.benign_indices = benign_indices
        (
            self.y_train_np,
            self.unique_classes,
            self.class_mapping,
            self.reverse_class_mapping,
        ) = _encode_labels(y_train)

        # Data is loaded as NumPy arrays first
        self.x_benign_np, self.y_benign_np = self._get_benign_data()

        self.feature_extraction_batch_size = feature_extraction_batch_size
        self.misclassification_batch_size = misclassification_batch_size

        # tf.data.Datasets are used as these are more efficient for TF-specific operations
        # Create x_train_dataset for feature extraction
        # Ensure data type is float32 for TensorFlow
        self.x_train_dataset = (
            self._tf_runtime.data.Dataset.from_tensor_slices(
                self.x_train.astype(self._tf_runtime.float32.as_numpy_dtype)
            )
            .batch(self.feature_extraction_batch_size)
            .prefetch(self._tf_runtime.data.AUTOTUNE)
        )

        # Create x_benign_dataset for misclassification rate calculation
        # Ensure data type is float32 for TensorFlow
        self.x_benign_dataset = (
            self._tf_runtime.data.Dataset.from_tensor_slices(
                self.x_benign_np.astype(self._tf_runtime.float32.as_numpy_dtype)
            )
            .batch(self.misclassification_batch_size)
            .prefetch(self._tf_runtime.data.AUTOTUNE)
        )

        self.feature_representation_model, self.classifying_submodel = self._extract_submodels(final_feature_layer_name)

        self.misclassification_threshold = np.float64(misclassification_threshold)

        # Get feature_shape and deviation_shape before wrapping functions
        # It's fundamental to wrap functions with input signatures
        if (
            hasattr(self.feature_representation_model, "output_shape")
            and self.feature_representation_model.output_shape[1:] is not None
        ):
            # Assumes output_shape is something like (None, 5) or (None, 28, 28, 3)
            self.feature_output_shape = self.feature_representation_model.output_shape[1:]
            # Handle case where output_shape[1:] might be an empty tuple if model outputs a scalar.
            # Unlikely for feature extractor, but good to be robust.
            if not self.feature_output_shape:
                logging.warning("feature_representation_model.output_shape[1:] is empty. Attempting dummy inference.")
                # Fallback to dummy inference if output_shape is unexpectedly empty
                dummy_input_shape = (
                    1,
                    *self.x_train.shape[1:],
                )  # Use actual classifier input shape
                dummy_input = self._tf_runtime.random.uniform(shape=dummy_input_shape, dtype=self._tf_runtime.float32)
                dummy_features = self.feature_representation_model(dummy_input)
                self.feature_output_shape = dummy_features.shape[1:]
        else:
            # Fallback if output_shape attribute is not present or usable
            logging.warning(
                "feature_representation_model.output_shape not directly available. "
                "Performing dummy inference for shape."
            )
            dummy_input_shape = (1, *self.x_train.shape[1:])  # Use actual classifier input shape
            dummy_input = self._tf_runtime.random.uniform(shape=dummy_input_shape, dtype=self._tf_runtime.float32)
            dummy_features = self.feature_representation_model(dummy_input)
            self.feature_output_shape = dummy_features.shape[1:]

        # If after all attempts, feature_output_shape is still empty/invalid, raise an error
        if not self.feature_output_shape or any(d is None for d in self.feature_output_shape):
            raise RuntimeError(
                f"Could not determine a valid feature_output_shape ({self.feature_output_shape}) for "
                f"tf.function input_signature. "
                "Ensure feature_representation_model is built and outputs a known shape."
            )

        # Dynamic @tf.function wrapping
        self._calculate_centroid_tf = self._tf_runtime.function(
            self._calculate_centroid_tf_unwrapped, reduce_retracing=True
        )
        self._calculate_features = self._tf_runtime.function(self._calculate_features_unwrapped)

        self._predict_with_deviation = self._tf_runtime.function(
            self._predict_with_deviation_unwrapped,
            input_signature=[
                self._tf_runtime.TensorSpec(shape=[None, *self.feature_output_shape], dtype=self._tf_runtime.float32),
                self._tf_runtime.TensorSpec(shape=self.feature_output_shape, dtype=self._tf_runtime.float32),
            ],
            reduce_retracing=True,
        )

        logging.info("CCA object created successfully.")

    def _calculate_centroid_tf_unwrapped(self, features):
        return self._tf_runtime.reduce_mean(features, axis=0)

    def _calculate_centroid(self, selected_indices: np.ndarray, features: np.ndarray) -> np.ndarray:
        """
        Returns the centroid of all data within a specific cluster that is classified as a specific class label

        :param selected_indices: a numpy array of selected indices on which to calculate the centroid
        :param features: numpy array d-dimensional features for all given data
        :return: d-dimensional numpy array
        """
        selected_features = features[selected_indices]
        features_tf = self._tf_runtime.convert_to_tensor(selected_features, dtype=self._tf_runtime.float32)
        centroid = self._calculate_centroid_tf(features_tf)
        return centroid.numpy()

    def _class_clustering(
        self, y: np.ndarray, features: np.ndarray, label: int | str, clusterer: ClusterMixin
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Given a class label, it clusters all the feature representations that map to that class

        :param y: array of n class labels
        :param label: class label in the classification task
        :param features: numpy array d-dimensional features for n data entries
        :param clusterer: clustering algorithm used
        :return: (cluster_labels, selected_indices) ndarrays of equal size with cluster labels and corresponding
            original indices.
        """
        logging.info("Clustering class %s...", label)
        selected_indices = np.where(y == label)[0]
        selected_features = features[selected_indices]
        cluster_labels = clusterer.fit_predict(selected_features)
        return cluster_labels, selected_indices

    def _calculate_features_unwrapped(self, feature_representation_model: Model, x: np.ndarray) -> np.ndarray:
        """
        Calculates the features using the first DNN slice

        :param feature_representation_model: DNN submodel from input up to feature abstraction
        :param x: input data
        :return: features array
        """
        return feature_representation_model(x, training=False)

    def _feature_extraction(self, x_train: tf_types.data.Dataset, feature_representation_model: Model) -> np.ndarray:
        """
        Extract features from the model using the feature representation sub model.

        :param x_train: Tensorflow dataset with features for n data entries. Features are extracted from here
        :param feature_representation_model: DNN submodel from input up to feature abstraction
        :return: features. numpy array of features
        """
        # Convert data to TensorFlow tensors if needed
        features_np_list: list[np.ndarray] = []

        # Iterate directly over the provided input_dataset
        for batch_data_tf in x_train:
            batch_features = self._calculate_features(feature_representation_model, batch_data_tf)
            features_np_list.append(batch_features.numpy())  # Convert to NumPy immediately

        # Concatenate all batches of numpy arrays on CPU (system RAM)
        final_features_np: np.ndarray = np.concatenate(features_np_list, axis=0)

        return final_features_np

    def _cluster_classes(
        self,
        y_train: np.ndarray,
        unique_classes: set[int],
        features: np.ndarray,
        clusterer: ClusterMixin,
    ) -> tuple[np.ndarray, dict]:
        """
        Clusters all the classes in the given dataset into uniquely identifiable clusters.

        :param y_train: numpy array of labels for n data entries
        :param unique_classes: set of unique classes
        :param features: feature representations' array of n rows
        :param clusterer: clustering algorithm used
        :return: (class_cluster_labels, cluster_class_mapping)
        """
        # represents the number of clusters used up until now to differentiate clusters obtained in different
        # clustering runs by classes
        logging.info("Clustering classes...")
        used_cluster_labels = 0
        cluster_class_mapping = {}
        class_cluster_labels = np.full(len(y_train), -1)

        logging.debug("Unique classes are: %s", unique_classes)

        for class_label in unique_classes:
            cluster_labels, selected_indices = self._class_clustering(y_train, features, class_label, clusterer)
            # label values are adjusted to account for labels of previous clustering tasks
            cluster_labels[cluster_labels != -1] += used_cluster_labels
            used_cluster_labels += len(np.unique(cluster_labels[cluster_labels != -1]))

            class_cluster_labels[selected_indices] = cluster_labels

            # the class (label) corresponding to the cluster is saved for centroid deviation calculation
            for label in np.unique(cluster_labels[cluster_labels != -1]):
                cluster_class_mapping[label] = class_label

        return class_cluster_labels, cluster_class_mapping

    def _get_benign_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Retrieves the benign data from the training data using benign indices

        :return: (x_benign, y_benign) ndarrays with the benign data.
        """
        if len(self.benign_indices) == 0:
            raise ValueError(f"Benign indices passed ({len(self.benign_indices)}) are not enough to run the algorithm")

        return self.x_train[self.benign_indices], self.y_train[self.benign_indices]

    def _extract_submodels(self, final_feature_layer_name: str) -> tuple[Model, Model]:
        """
        Extracts the feature representation and final classifier submodels from the original classifier.
        Composition of both models should result in the original model

        :param final_feature_layer_name: Name of the final layer in feature abstraction.
            Should be a ReLu-activated layer as suggested in the paper.
        :return: (feature_representation_submodel, classifying_submodel)
        """
        logging.info("Extracting submodels...")
        keras_model: Model = self.classifier.model

        try:
            final_feature_layer = keras_model.get_layer(name=final_feature_layer_name)
        except ValueError as exc:
            raise ValueError(f"Layer with name '{final_feature_layer_name}' not found in the model.") from exc

        if (
            not hasattr(final_feature_layer, "activation")
            or final_feature_layer.activation != self._tf_runtime.keras.activations.relu
        ):
            warnings.warn(
                f"Final feature layer '{final_feature_layer_name}' must have a ReLU activation.",
                UserWarning,
            )

        # Create a feature representation submodel with weight sharing
        feature_representation_model = self._KerasModel(
            inputs=keras_model.inputs,
            outputs=keras_model.get_layer(final_feature_layer_name).output,
            name="feature_representation_model",
        )

        final_feature_layer_index = keras_model.layers.index(final_feature_layer)
        classifier_submodel_layers = keras_model.layers[final_feature_layer_index + 1 :]

        # Create the classifier submodel
        classifying_submodel = self._KerasSequential(classifier_submodel_layers, name="classifying_submodel")

        intermediate_shape = feature_representation_model.output_shape[1:]
        dummy_input = self._tf_runtime.zeros((1,) + intermediate_shape)
        classifying_submodel(dummy_input, training=False)

        return feature_representation_model, classifying_submodel

    def get_clusters(self) -> np.ndarray:
        """
        :return: np.ndarray with m+1 columns, where m is dimensionality of the dimensionality reducer's output.
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
        _, confusion_matrix_json = evaluator.analyze_correctness(
            assigned_clean_by_class=assigned_clean_by_class,
            is_clean_by_class=is_clean_by_class,
        )

        return confusion_matrix_json

    def _predict_with_deviation_unwrapped(
        self, features: tf_types.Tensor, deviation: tf_types.Tensor
    ) -> tf_types.Tensor:
        deviated_features = self._tf_runtime.nn.relu(features + deviation)
        return self.classifying_submodel(deviated_features, training=False)

    def _calculate_misclassification_rate(self, class_label: int, deviation: np.ndarray) -> np.float64:
        """
        Calculate the misclassification rate when applying a deviation to other classes.

        :param class_label: The class label for which the deviation is calculated
        :param deviation: The deviation vector to apply
        :return: The misclassification rate (0.0 to 1.0)
        """
        # Convert deviation to a tensor once
        deviation_tf = self._tf_runtime.convert_to_tensor(deviation, dtype=self._tf_runtime.float32)

        total_elements = 0
        misclassified_elements = 0

        # Get all classes except the current one
        other_classes = self.unique_classes - {class_label}

        # Process each class
        for other_class_label in other_classes:
            # Get data for this class
            other_class_mask = self.y_benign_np == other_class_label
            other_class_data = self.x_benign_np[other_class_mask]

            if len(other_class_data) == 0:
                continue

            class_x_dataset = (
                self._tf_runtime.data.Dataset.from_tensor_slices(
                    other_class_data.astype(self._tf_runtime.float32.as_numpy_dtype)
                )
                .batch(self.misclassification_batch_size)
                .prefetch(self._tf_runtime.data.AUTOTUNE)
            )

            total_elements += len(other_class_data)
            class_misclassified = 0

            # Batches of the class are processed with deviation to determine misclassification
            for batch_data_tf in class_x_dataset:
                features_tf = self._calculate_features(self.feature_representation_model, batch_data_tf)

                predictions = self._predict_with_deviation(features_tf, deviation_tf)

                pred_classes_np = self._tf_runtime.argmax(predictions, axis=1).numpy()

                batch_misclassified = np.sum(pred_classes_np == class_label)
                class_misclassified += batch_misclassified

            misclassified_elements += class_misclassified

        # Avoid division by zero
        if total_elements == 0:
            return np.float64(0.0)

        logging.debug(
            "MR --> %s : %s / %s = %s",
            class_label,
            misclassified_elements,
            total_elements,
            np.float64(misclassified_elements) / np.float64(total_elements),
        )

        return np.float64(misclassified_elements) / np.float64(total_elements)

    def detect_poison(self, **kwargs) -> tuple[dict, list[int]]:
        # Saves important information about the algorithm execution for further analysis
        report: dict[str, Any] = {}

        self.is_clean_np = np.ones(len(self.y_train))

        self.features = self._feature_extraction(self.x_train_dataset, self.feature_representation_model)

        # Small fix to add flexibility to use CCAUD in non-flattened scenarios. Not recommended, but available
        if len(self.features.shape) > 2:
            num_samples = self.features.shape[0]
            self.features = self.features.reshape(num_samples, -1)  # Flattening

        self.features_reduced = self.reducer.fit_transform(self.features)

        self.class_cluster_labels, self.cluster_class_mapping = self._cluster_classes(
            self.y_train, self.unique_classes, self.features_reduced, self.clusterer
        )

        # outliers are poisoned
        outlier_indices = np.where(self.class_cluster_labels == -1)[0]
        self.is_clean_np[outlier_indices] = 0

        # cluster labels are saved in the report
        report["cluster_labels"] = self.get_clusters()
        report["cluster_data"] = {}

        logging.info("Calculating real centroids...")
        real_centroids = {}

        # for each cluster found for each target class
        for label in np.unique(self.class_cluster_labels[self.class_cluster_labels != -1]):
            selected_elements = np.where(self.class_cluster_labels == label)[0]
            real_centroids[label] = self._calculate_centroid(selected_elements, self.features)

            report["cluster_data"][label] = {}
            report["cluster_data"][label]["size"] = len(selected_elements)

        logging.info("Calculating benign centroids...")
        benign_centroids = {}

        logging.info("Target classes are: %s", self.unique_classes)

        # for each target class
        for class_label in self.unique_classes:
            benign_class_indices = np.intersect1d(self.benign_indices, np.where(self.y_train == class_label)[0])
            benign_centroids[class_label] = self._calculate_centroid(benign_class_indices, self.features)

        logging.info("Calculating misclassification rates...")
        misclassification_rates = {}

        for cluster_label, centroid in real_centroids.items():
            class_label = self.cluster_class_mapping[cluster_label]
            # B^k_i
            deviation = centroid - benign_centroids[class_label]

            # MR^k_i
            # with unique cluster labels for each cluster in each clustering run, the label
            # already maps to a target class
            misclassification_rates[cluster_label] = self._calculate_misclassification_rate(class_label, deviation)
            logging.info(
                "MR (k=%s, i=%s, |d|=%s) = %s",
                cluster_label,
                class_label,
                np.linalg.norm(deviation),  # This will be evaluated, but only if the log is emitted
                misclassification_rates[cluster_label],
            )

            report["cluster_data"][cluster_label]["centroid_l2"] = np.linalg.norm(real_centroids[cluster_label])
            report["cluster_data"][cluster_label]["deviation_l2"] = np.linalg.norm(deviation)
            report["cluster_data"][cluster_label]["class"] = class_label
            report["cluster_data"][cluster_label]["misclassification_rate"] = misclassification_rates[cluster_label]

        logging.info("Evaluating cluster misclassification...")
        for cluster_label, mr in misclassification_rates.items():
            if mr >= 1 - self.misclassification_threshold:
                cluster_indices = np.where(self.class_cluster_labels == cluster_label)[0]
                self.is_clean_np[cluster_indices] = 0
                logging.info(
                    "Cluster k=%s i=%s considered poison (%s >= %s)",
                    cluster_label,
                    self.cluster_class_mapping[cluster_label],
                    misclassification_rates[cluster_label],
                    1 - self.misclassification_threshold,
                )

        # Forced conversion for interface consistency
        self.is_clean: list[int] = self.is_clean_np.tolist()
        return report, self.is_clean.copy()
