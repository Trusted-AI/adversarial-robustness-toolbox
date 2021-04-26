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
This module implements methods performing poisoning detection based on activations clustering.

| Paper link: https://arxiv.org/abs/1811.03728

| Please keep in mind the limitations of defences. For more information on the limitations of this
    defence, see https://arxiv.org/abs/1905.13409 . For details on how to evaluate classifier security
    in general, see https://arxiv.org/abs/1902.06705
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import os
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from sklearn.cluster import KMeans, MiniBatchKMeans

from art import config
from art.data_generators import DataGenerator
from art.defences.detector.poison.clustering_analyzer import ClusteringAnalyzer
from art.defences.detector.poison.ground_truth_evaluator import GroundTruthEvaluator
from art.defences.detector.poison.poison_filtering_defence import PoisonFilteringDefence
from art.utils import segment_by_class
from art.visualization import create_sprite, save_image, plot_3d

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_NEURALNETWORK_TYPE

logger = logging.getLogger(__name__)


class ActivationDefence(PoisonFilteringDefence):
    """
    Method from Chen et al., 2018 performing poisoning detection based on activations clustering.

    | Paper link: https://arxiv.org/abs/1811.03728

    | Please keep in mind the limitations of defences. For more information on the limitations of this
        defence, see https://arxiv.org/abs/1905.13409 . For details on how to evaluate classifier security
        in general, see https://arxiv.org/abs/1902.06705
    """

    defence_params = ["nb_clusters", "clustering_method", "nb_dims", "reduce", "cluster_analysis", "generator"]
    valid_clustering = ["KMeans"]
    valid_reduce = ["PCA", "FastICA", "TSNE"]
    valid_analysis = ["smaller", "distance", "relative-size", "silhouette-scores"]

    TOO_SMALL_ACTIVATIONS = 32  # Threshold used to print a warning when activations are not enough

    def __init__(
        self,
        classifier: "CLASSIFIER_NEURALNETWORK_TYPE",
        x_train: Optional[np.ndarray],
        y_train: Optional[np.ndarray],
        generator: Optional[DataGenerator] = None,
    ) -> None:
        """
        Create an :class:`.ActivationDefence` object with the provided classifier.

        :param classifier: Model evaluated for poison.
        :param x_train: A dataset used to train the classifier.
        :param y_train: Labels used to train the classifier.
        :param generator: A data generator to be used instead of `x_train` and `y_train`.
        """
        super().__init__(classifier, x_train, y_train)
        self.classifier: "CLASSIFIER_NEURALNETWORK_TYPE" = classifier
        self.nb_clusters = 2
        self.clustering_method = "KMeans"
        self.nb_dims = 10
        self.reduce = "PCA"
        self.cluster_analysis = "smaller"
        self.generator = generator
        self.activations_by_class: List[np.ndarray] = []
        self.clusters_by_class: List[np.ndarray] = []
        self.assigned_clean_by_class: List[np.ndarray] = []
        self.is_clean_by_class: List[np.ndarray] = []
        self.errors_by_class: List[np.ndarray] = []
        self.red_activations_by_class: List[np.ndarray] = []  # Activations reduced by class
        self.evaluator = GroundTruthEvaluator()
        self.is_clean_lst: List[int] = []
        self.confidence_level: List[float] = []
        self.poisonous_clusters: List[List[np.ndarray]] = []
        self.clusterer = MiniBatchKMeans(n_clusters=self.nb_clusters)
        self._check_params()

    def evaluate_defence(self, is_clean: np.ndarray, **kwargs) -> str:
        """
        If ground truth is known, this function returns a confusion matrix in the form of a JSON object.

        :param is_clean: Ground truth, where is_clean[i]=1 means that x_train[i] is clean and is_clean[i]=0 means
                         x_train[i] is poisonous.
        :param kwargs: A dictionary of defence-specific parameters.
        :return: JSON object with confusion matrix.
        """
        if is_clean is None or is_clean.size == 0:
            raise ValueError("is_clean was not provided while invoking evaluate_defence.")

        self.set_params(**kwargs)

        if not self.activations_by_class and self.generator is None:
            activations = self._get_activations()
            self.activations_by_class = self._segment_by_class(activations, self.y_train)

        (
            self.clusters_by_class,
            self.red_activations_by_class,
        ) = self.cluster_activations()
        _, self.assigned_clean_by_class = self.analyze_clusters()

        # Now check ground truth:
        if self.generator is not None:
            batch_size = self.generator.batch_size
            num_samples = self.generator.size
            num_classes = self.classifier.nb_classes
            self.is_clean_by_class = [np.empty(0, dtype=int) for _ in range(num_classes)]

            # calculate is_clean_by_class for each batch
            for batch_idx in range(num_samples // batch_size):  # type: ignore
                _, y_batch = self.generator.get_batch()
                is_clean_batch = is_clean[batch_idx * batch_size : batch_idx * batch_size + batch_size]
                clean_by_class_batch = self._segment_by_class(is_clean_batch, y_batch)
                self.is_clean_by_class = [
                    np.append(self.is_clean_by_class[class_idx], clean_by_class_batch[class_idx])
                    for class_idx in range(num_classes)
                ]

        else:
            self.is_clean_by_class = self._segment_by_class(is_clean, self.y_train)
        self.errors_by_class, conf_matrix_json = self.evaluator.analyze_correctness(
            self.assigned_clean_by_class, self.is_clean_by_class
        )
        return conf_matrix_json

    # pylint: disable=W0221
    def detect_poison(self, **kwargs) -> Tuple[Dict[str, Any], List[int]]:
        """
        Returns poison detected and a report.

        :param clustering_method: clustering algorithm to be used. Currently `KMeans` is the only method supported
        :type clustering_method: `str`
        :param nb_clusters: number of clusters to find. This value needs to be greater or equal to one
        :type nb_clusters: `int`
        :param reduce: method used to reduce dimensionality of the activations. Supported methods include  `PCA`,
                       `FastICA` and `TSNE`
        :type reduce: `str`
        :param nb_dims: number of dimensions to be reduced
        :type nb_dims: `int`
        :param cluster_analysis: heuristic to automatically determine if a cluster contains poisonous data. Supported
                                 methods include `smaller` and `distance`. The `smaller` method defines as poisonous the
                                 cluster with less number of data points, while the `distance` heuristic uses the
                                 distance between the clusters.
        :type cluster_analysis: `str`
        :return: (report, is_clean_lst):
                where a report is a dict object that contains information specified by the clustering analysis technique
                where is_clean is a list, where is_clean_lst[i]=1 means that x_train[i]
                there is clean and is_clean_lst[i]=0, means that x_train[i] was classified as poison.
        """
        old_nb_clusters = self.nb_clusters
        self.set_params(**kwargs)
        if self.nb_clusters != old_nb_clusters:
            self.clusterer = MiniBatchKMeans(n_clusters=self.nb_clusters)

        if self.generator is not None:
            self.clusters_by_class, self.red_activations_by_class = self.cluster_activations()
            report, self.assigned_clean_by_class = self.analyze_clusters()

            batch_size = self.generator.batch_size
            num_samples = self.generator.size
            self.is_clean_lst = []

            # loop though the generator to generator a report
            for _ in range(num_samples // batch_size):  # type: ignore
                _, y_batch = self.generator.get_batch()
                indices_by_class = self._segment_by_class(np.arange(batch_size), y_batch)
                is_clean_lst = [0] * batch_size
                for class_idx, idxs in enumerate(indices_by_class):
                    for idx_in_class, idx in enumerate(idxs):
                        is_clean_lst[idx] = self.assigned_clean_by_class[class_idx][idx_in_class]
                self.is_clean_lst += is_clean_lst
            return report, self.is_clean_lst

        if not self.activations_by_class:
            activations = self._get_activations()
            self.activations_by_class = self._segment_by_class(activations, self.y_train)
        (
            self.clusters_by_class,
            self.red_activations_by_class,
        ) = self.cluster_activations()
        report, self.assigned_clean_by_class = self.analyze_clusters()
        # Here, assigned_clean_by_class[i][j] is 1 if the jth data point in the ith class was
        # determined to be clean by activation cluster

        # Build an array that matches the original indexes of x_train
        n_train = len(self.x_train)
        indices_by_class = self._segment_by_class(np.arange(n_train), self.y_train)
        self.is_clean_lst = [0] * n_train

        for assigned_clean, indices_dp in zip(self.assigned_clean_by_class, indices_by_class):
            for assignment, index_dp in zip(assigned_clean, indices_dp):
                if assignment == 1:
                    self.is_clean_lst[index_dp] = 1

        return report, self.is_clean_lst

    def cluster_activations(self, **kwargs) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Clusters activations and returns cluster_by_class and red_activations_by_class, where cluster_by_class[i][j] is
        the cluster to which the j-th data point in the ith class belongs and the correspondent activations reduced by
        class red_activations_by_class[i][j].

        :param kwargs: A dictionary of cluster-specific parameters.
        :return: Clusters per class and activations by class.
        """
        self.set_params(**kwargs)

        if self.generator is not None:
            batch_size = self.generator.batch_size
            num_samples = self.generator.size
            num_classes = self.classifier.nb_classes
            for batch_idx in range(num_samples // batch_size):  # type: ignore
                x_batch, y_batch = self.generator.get_batch()

                batch_activations = self._get_activations(x_batch)
                activation_dim = batch_activations.shape[-1]

                # initialize values list of lists on first run
                if batch_idx == 0:
                    self.activations_by_class = [np.empty((0, activation_dim)) for _ in range(num_classes)]
                    self.clusters_by_class = [np.empty(0, dtype=int) for _ in range(num_classes)]
                    self.red_activations_by_class = [np.empty((0, self.nb_dims)) for _ in range(num_classes)]

                activations_by_class = self._segment_by_class(batch_activations, y_batch)
                clusters_by_class, red_activations_by_class = cluster_activations(
                    activations_by_class,
                    nb_clusters=self.nb_clusters,
                    nb_dims=self.nb_dims,
                    reduce=self.reduce,
                    clustering_method=self.clustering_method,
                    generator=self.generator,
                    clusterer_new=self.clusterer,
                )

                for class_idx in range(num_classes):
                    self.activations_by_class[class_idx] = np.vstack(
                        [self.activations_by_class[class_idx], activations_by_class[class_idx]]
                    )
                    self.clusters_by_class[class_idx] = np.append(
                        self.clusters_by_class[class_idx], clusters_by_class[class_idx]
                    )
                    self.red_activations_by_class[class_idx] = np.vstack(
                        [self.red_activations_by_class[class_idx], red_activations_by_class[class_idx]]
                    )
            return self.clusters_by_class, self.red_activations_by_class

        if not self.activations_by_class:
            activations = self._get_activations()
            self.activations_by_class = self._segment_by_class(activations, self.y_train)

        [self.clusters_by_class, self.red_activations_by_class] = cluster_activations(
            self.activations_by_class,
            nb_clusters=self.nb_clusters,
            nb_dims=self.nb_dims,
            reduce=self.reduce,
            clustering_method=self.clustering_method,
        )

        return self.clusters_by_class, self.red_activations_by_class

    def analyze_clusters(self, **kwargs) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        This function analyzes the clusters according to the provided method.

        :param kwargs: A dictionary of cluster-analysis-specific parameters.
        :return: (report, assigned_clean_by_class), where the report is a dict object and assigned_clean_by_class
                 is an array of arrays that contains what data points where classified as clean.
        """
        self.set_params(**kwargs)

        if not self.clusters_by_class:
            self.cluster_activations()

        analyzer = ClusteringAnalyzer()
        if self.cluster_analysis == "smaller":
            (
                self.assigned_clean_by_class,
                self.poisonous_clusters,
                report,
            ) = analyzer.analyze_by_size(self.clusters_by_class)
        elif self.cluster_analysis == "relative-size":
            (
                self.assigned_clean_by_class,
                self.poisonous_clusters,
                report,
            ) = analyzer.analyze_by_relative_size(self.clusters_by_class)
        elif self.cluster_analysis == "distance":
            (self.assigned_clean_by_class, self.poisonous_clusters, report,) = analyzer.analyze_by_distance(
                self.clusters_by_class,
                separated_activations=self.red_activations_by_class,
            )
        elif self.cluster_analysis == "silhouette-scores":
            (self.assigned_clean_by_class, self.poisonous_clusters, report,) = analyzer.analyze_by_silhouette_score(
                self.clusters_by_class,
                reduced_activations_by_class=self.red_activations_by_class,
            )
        else:
            raise ValueError("Unsupported cluster analysis technique " + self.cluster_analysis)

        # Add to the report current parameters used to run the defence and the analysis summary
        report = dict(list(report.items()) + list(self.get_params().items()))

        return report, self.assigned_clean_by_class

    @staticmethod
    def relabel_poison_ground_truth(
        classifier: "CLASSIFIER_NEURALNETWORK_TYPE",
        x: np.ndarray,
        y_fix: np.ndarray,
        test_set_split: float = 0.7,
        tolerable_backdoor: float = 0.01,
        max_epochs: int = 50,
        batch_epochs: int = 10,
    ) -> Tuple[float, "CLASSIFIER_NEURALNETWORK_TYPE"]:
        """
        Revert poison attack by continue training the current classifier with `x`, `y_fix`. `test_set_split` determines
        the percentage in x that will be used as training set, while `1-test_set_split` determines how many data points
        to use for test set.

        :param classifier: Classifier to be fixed.
        :param x: Samples.
        :param y_fix: True label of `x_poison`.
        :param test_set_split: this parameter determine how much data goes to the training set.
               Here `test_set_split*len(y_fix)` determines the number of data points in `x_train`
               and `(1-test_set_split) * len(y_fix)` the number of data points in `x_test`.
        :param tolerable_backdoor: Threshold that determines what is the maximum tolerable backdoor success rate.
        :param max_epochs: Maximum number of epochs that the model will be trained.
        :param batch_epochs: Number of epochs to be trained before checking current state of model.
        :return: (improve_factor, classifier).
        """
        # Split data into testing and training:
        n_train = int(len(x) * test_set_split)
        x_train, x_test = x[:n_train], x[n_train:]
        y_train, y_test = y_fix[:n_train], y_fix[n_train:]

        filename = "original_classifier" + str(time.time()) + ".p"
        ActivationDefence._pickle_classifier(classifier, filename)

        # Now train using y_fix:
        improve_factor, _ = train_remove_backdoor(
            classifier,
            x_train,
            y_train,
            x_test,
            y_test,
            tolerable_backdoor=tolerable_backdoor,
            max_epochs=max_epochs,
            batch_epochs=batch_epochs,
        )

        # Only update classifier if there was an improvement:
        if improve_factor < 0:
            classifier = ActivationDefence._unpickle_classifier(filename)
            return 0, classifier

        ActivationDefence._remove_pickle(filename)
        return improve_factor, classifier

    @staticmethod
    def relabel_poison_cross_validation(
        classifier: "CLASSIFIER_NEURALNETWORK_TYPE",
        x: np.ndarray,
        y_fix: np.ndarray,
        n_splits: int = 10,
        tolerable_backdoor: float = 0.01,
        max_epochs: int = 50,
        batch_epochs: int = 10,
    ) -> Tuple[float, "CLASSIFIER_NEURALNETWORK_TYPE"]:
        """
        Revert poison attack by continue training the current classifier with `x`, `y_fix`. `n_splits` determines the
        number of cross validation splits.

        :param classifier: Classifier to be fixed.
        :param x: Samples that were miss-labeled.
        :param y_fix: True label of `x`.
        :param n_splits: Determines how many splits to use in cross validation (only used if `cross_validation=True`).
        :param tolerable_backdoor: Threshold that determines what is the maximum tolerable backdoor success rate.
        :param max_epochs: Maximum number of epochs that the model will be trained.
        :param batch_epochs: Number of epochs to be trained before checking current state of model.
        :return: (improve_factor, classifier)
        """
        # pylint: disable=E0001
        from sklearn.model_selection import KFold

        # Train using cross validation
        k_fold = KFold(n_splits=n_splits)
        KFold(n_splits=n_splits, random_state=None, shuffle=True)

        filename = "original_classifier" + str(time.time()) + ".p"
        ActivationDefence._pickle_classifier(classifier, filename)
        curr_improvement = 0

        for train_index, test_index in k_fold.split(x):
            # Obtain partition:
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y_fix[train_index], y_fix[test_index]
            # Unpickle original model:
            curr_classifier = ActivationDefence._unpickle_classifier(filename)

            new_improvement, fixed_classifier = train_remove_backdoor(
                curr_classifier,
                x_train,
                y_train,
                x_test,
                y_test,
                tolerable_backdoor=tolerable_backdoor,
                max_epochs=max_epochs,
                batch_epochs=batch_epochs,
            )
            if curr_improvement < new_improvement and new_improvement > 0:
                curr_improvement = new_improvement
                classifier = fixed_classifier
                logger.info("Selected as best model so far: %s", curr_improvement)

        ActivationDefence._remove_pickle(filename)
        return curr_improvement, classifier

    @staticmethod
    def _pickle_classifier(classifier: "CLASSIFIER_NEURALNETWORK_TYPE", file_name: str) -> None:
        """
        Pickles the self.classifier and stores it using the provided file_name in folder `art.config.ART_DATA_PATH`.

        :param classifier: Classifier to be pickled.
        :param file_name: Name of the file where the classifier will be pickled.
        """
        full_path = os.path.join(config.ART_DATA_PATH, file_name)
        folder = os.path.split(full_path)[0]
        if not os.path.exists(folder):
            os.makedirs(folder)

        with open(full_path, "wb") as f_classifier:
            pickle.dump(classifier, f_classifier)

    @staticmethod
    def _unpickle_classifier(file_name: str) -> "CLASSIFIER_NEURALNETWORK_TYPE":
        """
        Unpickles classifier using the filename provided. Function assumes that the pickle is in
        `art.config.ART_DATA_PATH`.

        :param file_name: Path of the pickled classifier relative to `ART_DATA_PATH`.
        :return: The loaded classifier.
        """
        full_path = os.path.join(config.ART_DATA_PATH, file_name)
        logger.info("Loading classifier from %s", full_path)
        with open(full_path, "rb") as f_classifier:
            loaded_classifier = pickle.load(f_classifier)
            return loaded_classifier

    @staticmethod
    def _remove_pickle(file_name: str) -> None:
        """
        Erases the pickle with the provided file name.

        :param file_name: File name without directory.
        """
        full_path = os.path.join(config.ART_DATA_PATH, file_name)
        os.remove(full_path)

    def visualize_clusters(
        self, x_raw: np.ndarray, save: bool = True, folder: str = ".", **kwargs
    ) -> List[List[List[np.ndarray]]]:
        """
        This function creates the sprite/mosaic visualization for clusters. When save=True,
        it also stores a sprite (mosaic) per cluster in art.config.ART_DATA_PATH.

        :param x_raw: Images used to train the classifier (before pre-processing).
        :param save: Boolean specifying if image should be saved.
        :param folder: Directory where the sprites will be saved inside art.config.ART_DATA_PATH folder.
        :param kwargs: a dictionary of cluster-analysis-specific parameters.
        :return: Array with sprite images sprites_by_class, where sprites_by_class[i][j] contains the
                                  sprite of class i cluster j.
        """
        self.set_params(**kwargs)

        if not self.clusters_by_class:
            self.cluster_activations()

        x_raw_by_class = self._segment_by_class(x_raw, self.y_train)
        x_raw_by_cluster: List[List[List[np.ndarray]]] = [
            [[] for _ in range(self.nb_clusters)] for _ in range(self.classifier.nb_classes)
        ]

        # Get all data in x_raw in the right cluster
        for n_class, cluster in enumerate(self.clusters_by_class):
            for j, assigned_cluster in enumerate(cluster):
                x_raw_by_cluster[n_class][assigned_cluster].append(x_raw_by_class[n_class][j])

        # Now create sprites:
        sprites_by_class: List[List[List[np.ndarray]]] = [
            [[] for _ in range(self.nb_clusters)] for _ in range(self.classifier.nb_classes)
        ]
        for i, class_i in enumerate(x_raw_by_cluster):
            for j, images_cluster in enumerate(class_i):
                title = "Class_" + str(i) + "_cluster_" + str(j) + "_clusterSize_" + str(len(images_cluster))
                f_name = title + ".png"
                f_name = os.path.join(folder, f_name)
                sprite = create_sprite(np.array(images_cluster))
                if save:
                    save_image(sprite, f_name)
                sprites_by_class[i][j] = sprite

        return sprites_by_class

    def plot_clusters(self, save: bool = True, folder: str = ".", **kwargs) -> None:
        """
        Creates a 3D-plot to visualize each cluster each cluster is assigned a different color in the plot. When
        save=True, it also stores the 3D-plot per cluster in art.config.ART_DATA_PATH.

        :param save: Boolean specifying if image should be saved.
        :param folder: Directory where the sprites will be saved inside art.config.ART_DATA_PATH folder.
        :param kwargs: a dictionary of cluster-analysis-specific parameters.
        """
        self.set_params(**kwargs)

        if not self.clusters_by_class:
            self.cluster_activations()

        # Get activations reduced to 3-components:
        separated_reduced_activations = []
        for activation in self.activations_by_class:
            reduced_activations = reduce_dimensionality(activation, nb_dims=3)
            separated_reduced_activations.append(reduced_activations)

        # For each class generate a plot:
        for class_id, (labels, coordinates) in enumerate(zip(self.clusters_by_class, separated_reduced_activations)):
            f_name = ""
            if save:
                f_name = os.path.join(folder, "plot_class_" + str(class_id) + ".png")
            plot_3d(coordinates, labels, save=save, f_name=f_name)

    def _check_params(self):
        if self.nb_clusters <= 1:
            raise ValueError(
                "Wrong number of clusters, should be greater or equal to 2. Provided: " + str(self.nb_clusters)
            )
        if self.nb_dims <= 0:
            raise ValueError("Wrong number of dimensions.")
        if self.clustering_method not in self.valid_clustering:
            raise ValueError("Unsupported clustering method: " + self.clustering_method)
        if self.reduce not in self.valid_reduce:
            raise ValueError("Unsupported reduction method: " + self.reduce)
        if self.cluster_analysis not in self.valid_analysis:
            raise ValueError("Unsupported method for cluster analysis method: " + self.cluster_analysis)
        if self.generator and not isinstance(self.generator, DataGenerator):
            raise TypeError("Generator must a an instance of DataGenerator")

    def _get_activations(self, x_train: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Find activations from :class:`.Classifier`.
        """
        logger.info("Getting activations")

        if self.classifier.layer_names is not None:
            nb_layers = len(self.classifier.layer_names)
        else:
            raise ValueError("No layer names identified.")
        protected_layer = nb_layers - 1

        if self.generator is not None:
            activations = self.classifier.get_activations(
                x_train, layer=protected_layer, batch_size=self.generator.batch_size
            )
        else:
            activations = self.classifier.get_activations(self.x_train, layer=protected_layer, batch_size=128)

        # wrong way to get activations activations = self.classifier.predict(self.x_train)
        nodes_last_layer = np.shape(activations)[1]

        if nodes_last_layer <= self.TOO_SMALL_ACTIVATIONS:
            logger.warning(
                "Number of activations in last hidden layer is too small. Method may not work properly. " "Size: %s",
                str(nodes_last_layer),
            )
        return activations

    def _segment_by_class(self, data: np.ndarray, features: np.ndarray) -> List[np.ndarray]:
        """
        Returns segmented data according to specified features.

        :param data: Data to be segmented.
        :param features: Features used to segment data, e.g., segment according to predicted label or to `y_train`.
        :return: Segmented data according to specified features.
        """
        n_classes = self.classifier.nb_classes
        return segment_by_class(data, features, n_classes)


def measure_misclassification(
    classifier: "CLASSIFIER_NEURALNETWORK_TYPE", x_test: np.ndarray, y_test: np.ndarray
) -> float:
    """
    Computes 1-accuracy given x_test and y_test

    :param classifier: Classifier to be used for predictions.
    :param x_test: Test set.
    :param y_test: Labels for test set.
    :return: 1-accuracy.
    """
    predictions = np.argmax(classifier.predict(x_test), axis=1)
    return 1.0 - np.sum(predictions == np.argmax(y_test, axis=1)) / y_test.shape[0]


def train_remove_backdoor(
    classifier: "CLASSIFIER_NEURALNETWORK_TYPE",
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    tolerable_backdoor: float,
    max_epochs: int,
    batch_epochs: int,
) -> tuple:
    """
    Trains the provider classifier until the tolerance or number of maximum epochs are reached.

    :param classifier: Classifier to be used for predictions.
    :param x_train: Training set.
    :param y_train: Labels used for training.
    :param x_test: Samples in test set.
    :param y_test: Labels in test set.
    :param tolerable_backdoor: Parameter that determines how many misclassifications are acceptable.
    :param max_epochs: maximum number of epochs to be run.
    :param batch_epochs: groups of epochs that will be run together before checking for termination.
    :return: (improve_factor, classifier).
    """
    # Measure poison success in current model:
    initial_missed = measure_misclassification(classifier, x_test, y_test)

    curr_epochs = 0
    curr_missed = 1.0
    while curr_epochs < max_epochs and curr_missed > tolerable_backdoor:
        classifier.fit(x_train, y_train, nb_epochs=batch_epochs)
        curr_epochs += batch_epochs
        curr_missed = measure_misclassification(classifier, x_test, y_test)
        logger.info("Current epoch: %s", curr_epochs)
        logger.info("Misclassifications: %s", curr_missed)

    improve_factor = initial_missed - curr_missed
    return improve_factor, classifier


def cluster_activations(
    separated_activations: List[np.ndarray],
    nb_clusters: int = 2,
    nb_dims: int = 10,
    reduce: str = "FastICA",
    clustering_method: str = "KMeans",
    generator: Optional[DataGenerator] = None,
    clusterer_new: Optional[MiniBatchKMeans] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Clusters activations and returns two arrays.
    1) separated_clusters: where separated_clusters[i] is a 1D array indicating which cluster each data point
    in the class has been assigned.
    2) separated_reduced_activations: activations with dimensionality reduced using the specified reduce method.

    :param separated_activations: List where separated_activations[i] is a np matrix for the ith class where
           each row corresponds to activations for a given data point.
    :param nb_clusters: number of clusters (defaults to 2 for poison/clean).
    :param nb_dims: number of dimensions to reduce activation to via PCA.
    :param reduce: Method to perform dimensionality reduction, default is FastICA.
    :param clustering_method: Clustering method to use, default is KMeans.
    :param generator: whether or not a the activations are a batch or full activations
    :return: (separated_clusters, separated_reduced_activations).
    :param clusterer_new: whether or not a the activations are a batch or full activations
    :return: (separated_clusters, separated_reduced_activations)
    """
    separated_clusters = []
    separated_reduced_activations = []

    if clustering_method == "KMeans":
        clusterer = KMeans(n_clusters=nb_clusters)
    else:
        raise ValueError(clustering_method + " clustering method not supported.")

    for activation in separated_activations:
        # Apply dimensionality reduction
        nb_activations = np.shape(activation)[1]
        if nb_activations > nb_dims:
            # TODO: address issue where if fewer samples than nb_dims this fails
            reduced_activations = reduce_dimensionality(activation, nb_dims=nb_dims, reduce=reduce)
        else:
            logger.info(
                "Dimensionality of activations = %i less than nb_dims = %i. Not applying dimensionality " "reduction.",
                nb_activations,
                nb_dims,
            )
            reduced_activations = activation
        separated_reduced_activations.append(reduced_activations)

        # Get cluster assignments
        if generator is not None and clusterer_new is not None:
            clusterer_new = clusterer_new.partial_fit(reduced_activations)
            # NOTE: this may cause earlier predictions to be less accurate
            clusters = clusterer_new.predict(reduced_activations)
        else:
            clusters = clusterer.fit_predict(reduced_activations)
        separated_clusters.append(clusters)

    return separated_clusters, separated_reduced_activations


def reduce_dimensionality(activations: np.ndarray, nb_dims: int = 10, reduce: str = "FastICA") -> np.ndarray:
    """
    Reduces dimensionality of the activations provided using the specified number of dimensions and reduction technique.

    :param activations: Activations to be reduced.
    :param nb_dims: number of dimensions to reduce activation to via PCA.
    :param reduce: Method to perform dimensionality reduction, default is FastICA.
    :return: Array with the reduced activations.
    """
    # pylint: disable=E0001
    from sklearn.decomposition import FastICA, PCA

    if reduce == "FastICA":
        projector = FastICA(n_components=nb_dims, max_iter=1000, tol=0.005)
    elif reduce == "PCA":
        projector = PCA(n_components=nb_dims)
    else:
        raise ValueError(reduce + " dimensionality reduction method not supported.")

    reduced_activations = projector.fit_transform(activations)
    return reduced_activations
