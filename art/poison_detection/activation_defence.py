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
"""
This module implements methods performing poisoning detection based on activations clustering.

| Paper link: https://arxiv.org/abs/1811.03728
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import os

import numpy as np

from art.poison_detection.clustering_analyzer import ClusteringAnalyzer
from art.poison_detection.ground_truth_evaluator import GroundTruthEvaluator
from art.poison_detection.poison_filtering_defence import PoisonFilteringDefence
from art.visualization import create_sprite, save_image, plot_3d

logger = logging.getLogger(__name__)


class ActivationDefence(PoisonFilteringDefence):
    """
    Method from Chen et al., 2018 performing poisoning detection based on activations clustering.

    | Paper link: https://arxiv.org/abs/1811.03728
    """
    defence_params = ['nb_clusters', 'clustering_method', 'nb_dims', 'reduce', 'cluster_analysis']
    valid_clustering = ['KMeans']
    valid_reduce = ['PCA', 'FastICA', 'TSNE']
    valid_analysis = ['smaller', 'distance', 'relative-size', 'silhouette-scores']

    TOO_SMALL_ACTIVATIONS = 32  # Threshold used to print a warning when activations are not enough

    def __init__(self, classifier, x_train, y_train):
        """
        Create an :class:`.ActivationDefence` object with the provided classifier.

        :param classifier: Model evaluated for poison.
        :type classifier: :class:`.Classifier`
        :param x_train: dataset used to train the classifier.
        :type x_train: `np.ndarray`
        :param y_train: labels used to train the classifier.
        :type y_train: `np.ndarray`
        """
        super(ActivationDefence, self).__init__(classifier, x_train, y_train)
        kwargs = {'nb_clusters': 2, 'clustering_method': "KMeans", 'nb_dims': 10, 'reduce': 'PCA',
                  'cluster_analysis': "smaller"}
        self.set_params(**kwargs)
        self.activations_by_class = []
        self.clusters_by_class = []
        self.assigned_clean_by_class = []
        self.is_clean_by_class = []
        self.errors_by_class = []
        self.red_activations_by_class = []  # Activations reduced by class
        self.evaluator = GroundTruthEvaluator()
        self.is_clean_lst = []
        self.confidence_level = []
        self.poisonous_clusters = []

    def evaluate_defence(self, is_clean, **kwargs):
        """
        If ground truth is known, this function returns a confusion matrix in the form of a JSON object.

        :param is_clean: Ground truth, where is_clean[i]=1 means that x_train[i] is clean and is_clean[i]=0 means
                         x_train[i] is poisonous.
        :type is_clean: :class `np.ndarray`
        :param kwargs: A dictionary of defence-specific parameters.
        :type kwargs: `dict`
        :return: JSON object with confusion matrix.
        :rtype: `jsonObject`
        """
        if is_clean is None or is_clean.size == 0:
            raise ValueError("is_clean was not provided while invoking evaluate_defence.")

        self.set_params(**kwargs)

        if not self.activations_by_class:
            activations = self._get_activations()
            self.activations_by_class = self._segment_by_class(activations, self.y_train)

        self.clusters_by_class, self.red_activations_by_class = self.cluster_activations()
        _, self.assigned_clean_by_class = self.analyze_clusters()

        # Now check ground truth:
        self.is_clean_by_class = self._segment_by_class(is_clean, self.y_train)
        self.errors_by_class, conf_matrix_json = self.evaluator.analyze_correctness(self.assigned_clean_by_class,
                                                                                    self.is_clean_by_class)
        return conf_matrix_json

    # pylint: disable=W0221
    def detect_poison(self, clustering_method='KMeans', nb_clusters=2, reduce='PCA', nb_dims=2,
                      cluster_analysis='smaller'):
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
        :rtype: `tuple`
        """

        args = {'clustering_method': clustering_method,
                'nb_clusters': nb_clusters,
                'reduce': reduce,
                'nb_dims': nb_dims,
                'cluster_analysis': cluster_analysis}

        self.set_params(**args)

        if not self.activations_by_class:
            activations = self._get_activations()
            self.activations_by_class = self._segment_by_class(activations, self.y_train)
        self.clusters_by_class, self.red_activations_by_class = self.cluster_activations()
        report, self.assigned_clean_by_class = self.analyze_clusters()
        # Here, assigned_clean_by_class[i][j] is 1 if the jth datapoint in the ith class was
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

    def cluster_activations(self, **kwargs):
        """
        Clusters activations and returns cluster_by_class and red_activations_by_class, where cluster_by_class[i][j] is
        the cluster to which the j-th datapoint in the ith class belongs and the correspondent activations reduced by
        class red_activations_by_class[i][j].

        :param kwargs: A dictionary of cluster-specific parameters.
        :type kwargs: `dict`
        :return: Clusters per class and activations by class.
        :rtype: `tuple`
        """
        self.set_params(**kwargs)
        if not self.activations_by_class:
            activations = self._get_activations()
            self.activations_by_class = self._segment_by_class(activations, self.y_train)

        [self.clusters_by_class, self.red_activations_by_class] = cluster_activations(
            self.activations_by_class,
            nb_clusters=self.nb_clusters,
            nb_dims=self.nb_dims,
            reduce=self.reduce,
            clustering_method=self.clustering_method)

        return self.clusters_by_class, self.red_activations_by_class

    def analyze_clusters(self, **kwargs):
        """
        This function analyzes the clusters according to the provided method.

        :param kwargs: A dictionary of cluster-analysis-specific parameters.
        :type kwargs: `dict`
        :return: (report, assigned_clean_by_class), where the report is a dict object and assigned_clean_by_class
                 is an array of arrays that contains what data points where classified as clean.
        :rtype: `tuple(dict, np.ndarray)`
        """
        self.set_params(**kwargs)

        if not self.clusters_by_class:
            self.cluster_activations()

        analyzer = ClusteringAnalyzer()

        if self.cluster_analysis == 'smaller':
            self.assigned_clean_by_class, self.poisonous_clusters, report \
                = analyzer.analyze_by_size(self.clusters_by_class)
        elif self.cluster_analysis == 'relative-size':
            self.assigned_clean_by_class, self.poisonous_clusters, report \
                = analyzer.analyze_by_relative_size(self.clusters_by_class)
        elif self.cluster_analysis == 'distance':
            self.assigned_clean_by_class, self.poisonous_clusters, report \
                = analyzer.analyze_by_distance(self.clusters_by_class,
                                               separated_activations=self.red_activations_by_class)
        elif self.cluster_analysis == 'silhouette-scores':
            self.assigned_clean_by_class, self.poisonous_clusters, report \
                = analyzer.analyze_by_silhouette_score(self.clusters_by_class,
                                                       reduced_activations_by_class=self.red_activations_by_class)
        else:
            raise ValueError(
                "Unsupported cluster analysis technique " + self.cluster_analysis)

        # Add to the report current parameters used to run the defence and the analysis summary
        report = dict(list(report.items()) + list(self.get_params().items()))

        return report, self.assigned_clean_by_class

    @staticmethod
    def relabel_poison_ground_truth(classifier, x, y_fix, test_set_split=0.7, tolerable_backdoor=0.01,
                                    max_epochs=50, batch_epochs=10):
        """
        Revert poison attack by continue training the current classifier with `x`, `y_fix`. `test_set_split` determines
        the percentage in x that will be used as training set, while `1-test_set_split` determines how many data points
        to use for test set.

        :param classifier: Classifier to be fixed
        :type classifier: :class:`.Classifier`
        :param x: samples
        :type x: `np.ndarray`
        :param y_fix: true label of x_poison
        :type y_fix: `np.ndarray`
        :param test_set_split: this parameter determine how much data goes to the training set.
               Here `test_set_split*len(y_fix)` determines the number of data points in `x_train`
               and `(1-test_set_split) * len(y_fix)` the number of data points in `x_test`.
        :param tolerable_backdoor: Threshold that determines what is the maximum tolerable backdoor success rate.
        :type tolerable_backdoor: `float`
        :param max_epochs: Maximum number of epochs that the model will be trained
        :type max_epochs: `int`
        :param batch_epochs: Number of epochs to be trained before checking current state of model
        :type batch_epochs: `int`
        :return: (improve_factor, classifier)
        :rtype: `float`, `.Classifier`
        """
        # Split data into testing and training:
        n_train = int(len(x) * test_set_split)
        x_train, x_test = x[:n_train], x[n_train:]
        y_train, y_test = y_fix[:n_train], y_fix[n_train:]

        import time
        filename = 'original_classifier' + str(time.time()) + '.p'
        ActivationDefence._pickle_classifier(classifier, filename)

        # Now train using y_fix:
        improve_factor, _ = train_remove_backdoor(classifier, x_train, y_train, x_test, y_test,
                                                  tolerable_backdoor=tolerable_backdoor, max_epochs=max_epochs,
                                                  batch_epochs=batch_epochs)

        # Only update classifier if there was an improvement:
        if improve_factor < 0:
            classifier = ActivationDefence._unpickle_classifier(filename)
            return 0, classifier

        ActivationDefence._remove_pickle(filename)
        return improve_factor, classifier

    @staticmethod
    def relabel_poison_cross_validation(classifier, x, y_fix, n_splits=10, tolerable_backdoor=0.01,
                                        max_epochs=50, batch_epochs=10):
        """
        Revert poison attack by continue training the current classifier with `x`, `y_fix`. `n_splits` determines the
        number of cross validation splits.

        :param classifier: Classifier to be fixed
        :type classifier: :class:`.Classifier`
        :param x: Samples that were miss-labeled.
        :type x: `np.ndarray`
        :param y_fix: True label of `x`.
        :type y_fix: `np.ndarray`
        :param n_splits: Determines how many splits to use in cross validation (only used if `cross_validation=True`).
        :type n_splits: `int`
        :param tolerable_backdoor: Threshold that determines what is the maximum tolerable backdoor success rate.
        :type tolerable_backdoor: `float`
        :param max_epochs: Maximum number of epochs that the model will be trained.
        :type max_epochs: `int`
        :param batch_epochs: Number of epochs to be trained before checking current state of model.
        :type batch_epochs: `int`
        :return: (improve_factor, classifier)
        :rtype: `float`, `.Classifier`
        """
        # pylint: disable=E0001
        # Train using cross validation
        from sklearn.model_selection import KFold
        k_fold = KFold(n_splits=n_splits)
        KFold(n_splits=n_splits, random_state=None, shuffle=True)

        import time
        filename = 'original_classifier' + str(time.time()) + '.p'
        ActivationDefence._pickle_classifier(classifier, filename)
        curr_improvement = 0

        for _, (train_index, test_index) in enumerate(k_fold.split(x)):
            # Obtain partition:
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y_fix[train_index], y_fix[test_index]
            # Unpickle original model:
            curr_classifier = ActivationDefence._unpickle_classifier(filename)

            new_improvement, fixed_classifier = train_remove_backdoor(curr_classifier, x_train, y_train, x_test,
                                                                      y_test,
                                                                      tolerable_backdoor=tolerable_backdoor,
                                                                      max_epochs=max_epochs,
                                                                      batch_epochs=batch_epochs)
            if curr_improvement < new_improvement and new_improvement > 0:
                curr_improvement = new_improvement
                classifier = fixed_classifier
                logger.info('Selected as best model so far: %s', curr_improvement)

        ActivationDefence._remove_pickle(filename)
        return curr_improvement, classifier

    @staticmethod
    def _pickle_classifier(classifier, file_name):
        """
        Pickles the self.classifier and stores it using the provided file_name in folder `art.DATA_PATH`.

        :param classifier: Classifier to be pickled.
        :type classifier: :class:`.Classifier`
        :param file_name: Name of the file where the classifier will be pickled
        :return: None
        """
        import pickle
        from art import DATA_PATH
        full_path = os.path.join(DATA_PATH, file_name)
        folder = os.path.split(full_path)[0]
        if not os.path.exists(folder):
            os.makedirs(folder)

        with open(full_path, 'wb') as f_classifier:
            pickle.dump(classifier, f_classifier)

    @staticmethod
    def _unpickle_classifier(file_name):
        """
        Unpickles classifier using the filename provided. Function assumes that the pickle is in `art.DATA_PATH`.

        :param file_name:
        :return:
        """
        from art import DATA_PATH
        import pickle

        full_path = os.path.join(DATA_PATH, file_name)
        logger.info('Loading classifier from %s', full_path)
        with open(full_path, 'rb') as f_classifier:
            loaded_classifier = pickle.load(f_classifier)
            return loaded_classifier

    @staticmethod
    def _remove_pickle(file_name):
        """
        Erases the pickle with the provided file name

        :param file_name: File name without directory
        :return: None
        """
        from art import DATA_PATH
        full_path = os.path.join(DATA_PATH, file_name)
        os.remove(full_path)

    def visualize_clusters(self, x_raw, save=True, folder='.', **kwargs):
        """
        This function creates the sprite/mosaic visualization for clusters. When save=True,
        it also stores a sprite (mosaic) per cluster in DATA_PATH.

        :param x_raw: Images used to train the classifier (before pre-processing)
        :type x_raw: `np.darray`
        :param save: Boolean specifying if image should be saved
        :type  save: `bool`
        :param folder: Directory where the sprites will be saved inside DATA_PATH folder
        :type folder: `str`
        :param kwargs: a dictionary of cluster-analysis-specific parameters
        :type kwargs: `dict`
        :return: Array with sprite images sprites_by_class, where sprites_by_class[i][j] contains the
                                  sprite of class i cluster j.
        :rtype: `np.ndarray`
        """
        self.set_params(**kwargs)

        if not self.clusters_by_class:
            self.cluster_activations()

        x_raw_by_class = self._segment_by_class(x_raw, self.y_train)
        x_raw_by_cluster = [[[] for _ in range(self.nb_clusters)] for y in range(self.classifier.nb_classes())]

        # Get all data in x_raw in the right cluster
        for n_class, cluster in enumerate(self.clusters_by_class):
            for j, assigned_cluster in enumerate(cluster):
                x_raw_by_cluster[n_class][assigned_cluster].append(x_raw_by_class[n_class][j])

        # Now create sprites:
        sprites_by_class = [[[] for _ in range(self.nb_clusters)] for y in range(self.classifier.nb_classes())]
        for i, class_i in enumerate(x_raw_by_cluster):
            for j, images_cluster in enumerate(class_i):
                title = 'Class_' + str(i) + '_cluster_' + str(j) + '_clusterSize_' + str(len(images_cluster))
                f_name = title + '.png'
                f_name = os.path.join(folder, f_name)
                sprite = create_sprite(images_cluster)
                if save:
                    save_image(sprite, f_name)
                sprites_by_class[i][j] = sprite

        return sprites_by_class

    def plot_clusters(self, save=True, folder='.', **kwargs):
        """
        Creates a 3D-plot to visualize each cluster each cluster is assigned a different color in the plot. When
        save=True, it also stores the 3D-plot per cluster in DATA_PATH.

        :param save: Boolean specifying if image should be saved
        :type  save: `bool`
        :param folder: Directory where the sprites will be saved inside DATA_PATH folder
        :type folder: `str`
        :param kwargs: a dictionary of cluster-analysis-specific parameters
        :type kwargs: `dict`
        :return: None
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
            f_name = ''
            if save:
                f_name = os.path.join(folder, 'plot_class_' + str(class_id) + '.png')
            plot_3d(coordinates, labels, save=save, f_name=f_name)

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and applies defence-specific checks before saving them as attributes.
        If a parameter is not provided, it takes its default value.

        :param nb_clusters: Number of clusters to be produced. Should be greater than 2.
        :type nb_clusters: `int`
        :param clustering_method: Clustering method to use
        :type clustering_method: `str`
        :param nb_dims: Number of dimensions to project on
        :type nb_dims: `int`
        :param reduce: Reduction technique
        :type reduce: `str`
        :param cluster_analysis: Method to analyze the clusters
        :type cluster_analysis: `str`
        """
        # Save defence-specific parameters
        super(ActivationDefence, self).set_params(**kwargs)

        if self.nb_clusters <= 1:
            raise ValueError(
                "Wrong number of clusters, should be greater or equal to 2. Provided: " + str(self.nb_clusters))
        if self.nb_dims <= 0:
            raise ValueError("Wrong number of dimensions ")
        if self.clustering_method not in self.valid_clustering:
            raise ValueError("Unsupported clustering method: " + self.clustering_method)
        if self.reduce not in self.valid_reduce:
            raise ValueError("Unsupported reduction method: " + self.reduce)
        if self.cluster_analysis not in self.valid_analysis:
            raise ValueError("Unsupported method for cluster analysis method: " + self.cluster_analysis)

        return True

    def _get_activations(self):
        """
        Find activations from :class:`.Classifier`.
        """
        logger.info('Getting activations')

        nb_layers = len(self.classifier.layer_names)
        activations = self.classifier.get_activations(self.x_train, layer=nb_layers - 1)

        # wrong way to get activations activations = self.classifier.predict(self.x_train)
        nodes_last_layer = np.shape(activations)[1]

        if nodes_last_layer <= self.TOO_SMALL_ACTIVATIONS:
            logger.warning("Number of activations in last hidden layer is too small. Method may not work properly. "
                           "Size: %s", str(nodes_last_layer))
        return activations

    def _segment_by_class(self, data, features):
        """
        Returns segmented data according to specified features.

        :param data: to be segmented
        :type data: `np.ndarray`
        :param features: features used to segment data, e.g., segment according to predicted label or to `y_train`
        :type features: `np.ndarray`
        :return: segmented data according to specified features.
        :rtype: `list`
        """
        n_classes = self.classifier.nb_classes()
        by_class = [[] for _ in range(n_classes)]
        for indx, feature in enumerate(features):
            if n_classes > 2:
                assigned = np.argmax(feature)
            else:
                assigned = int(feature)
            by_class[assigned].append(data[indx])

        return [np.asarray(i) for i in by_class]


def measure_misclassification(classifier, x_test, y_test):
    """
    Computes 1-accuracy given x_test and y_test

    :param classifier: art.classifier to be used for predictions
    :param x_test: test set
    :type x_test: `np.darray`
    :param y_test: labels test set
    :type y_test: `np.darray`
    :return: 1-accuracy
    :rtype `float`
    """
    predictions = np.argmax(classifier.predict(x_test), axis=1)
    return 1 - np.sum(predictions == np.argmax(y_test, axis=1)) / y_test.shape[0]


def train_remove_backdoor(classifier, x_train, y_train, x_test, y_test, tolerable_backdoor,
                          max_epochs, batch_epochs):
    """
    Trains the provider classifier until the tolerance or number of maximum epochs are reached.

    :param classifier: art.classifier to be used for predictions
    :type classifier: `art.classifier`
    :param x_train: training set
    :type x_train: `np.darray`
    :param y_train: labels used for training
    :type y_train: `np.darray`
    :param x_test: samples in test set
    :type x_test: `np.darray`
    :param y_test: labels in test set
    :type y_train: `np.darray`
    :param tolerable_backdoor: Parameter that determines how many missclassifications are acceptable.
    :type tolerable_backdoor: `float`
    :param max_epochs: maximum number of epochs to be run
    :type max_epochs: `int`
    :param batch_epochs: groups of epochs that will be run together before checking for termination
    :type batch_epochs: `int`
    :return: (improve_factor, classifier)
    :rtype `tuple`
    """
    # Measure poison success in current model:
    initial_missed = measure_misclassification(classifier, x_test, y_test)

    curr_epochs = 0
    curr_missed = 1
    while curr_epochs < max_epochs and curr_missed > tolerable_backdoor:
        classifier.fit(x_train, y_train, nb_epochs=batch_epochs)
        curr_epochs += batch_epochs
        curr_missed = measure_misclassification(classifier, x_test, y_test)
        logger.info('Current epoch: %s', curr_epochs)
        logger.info('Misclassifications: %s', curr_missed)

    improve_factor = initial_missed - curr_missed
    return improve_factor, classifier


def cluster_activations(separated_activations, nb_clusters=2, nb_dims=10, reduce='FastICA', clustering_method='KMeans'):
    """
    Clusters activations and returns two arrays.
    1) separated_clusters: where separated_clusters[i] is a 1D array indicating which cluster each datapoint
    in the class has been assigned
    2) separated_reduced_activations: activations with dimensionality reduced using the specified reduce method

    :param separated_activations: list where separated_activations[i] is a np matrix for the ith class where
    each row corresponds to activations for a given data point
    :type separated_activations: `list`
    :param nb_clusters: number of clusters (defaults to 2 for poison/clean)
    :type nb_clusters: `int`
    :param nb_dims: number of dimensions to reduce activation to via PCA
    :type nb_dims: `int`
    :param reduce: Method to perform dimensionality reduction, default is FastICA
    :type reduce: `str`
    :param clustering_method: Clustering method to use, default is KMeans
    :type clustering_method: `str`
    :return: separated_clusters, separated_reduced_activations
    :rtype: `tuple`
    """
    # pylint: disable=E0001
    from sklearn.cluster import KMeans

    separated_clusters = []
    separated_reduced_activations = []

    if clustering_method == 'KMeans':
        clusterer = KMeans(n_clusters=nb_clusters)
    else:
        raise ValueError(clustering_method + " clustering method not supported.")

    for activation in separated_activations:
        # Apply dimensionality reduction
        nb_activations = np.shape(activation)[1]
        if nb_activations > nb_dims:
            reduced_activations = reduce_dimensionality(activation, nb_dims=nb_dims, reduce=reduce)
        else:
            logger.info("Dimensionality of activations = %i less than nb_dims = %i. Not applying dimensionality "
                        "reduction.", nb_activations, nb_dims)
            reduced_activations = activation
        separated_reduced_activations.append(reduced_activations)

        # Get cluster assignments
        clusters = clusterer.fit_predict(reduced_activations)
        separated_clusters.append(clusters)

    return separated_clusters, separated_reduced_activations


def reduce_dimensionality(activations, nb_dims=10, reduce='FastICA'):
    """
    Reduces dimensionality of the activations provided using the specified number of dimensions and reduction technique.

    :param activations: Activations to be reduced
    :type activations: `numpy.ndarray`
    :param nb_dims: number of dimensions to reduce activation to via PCA
    :type nb_dims: `int`
    :param reduce: Method to perform dimensionality reduction, default is FastICA
    :type reduce: `str`
    :return: array with the activations reduced
    :rtype: `numpy.ndarray`
    """
    # pylint: disable=E0001
    from sklearn.decomposition import FastICA, PCA
    if reduce == 'FastICA':
        projector = FastICA(n_components=nb_dims, max_iter=1000, tol=0.005)
    elif reduce == 'PCA':
        projector = PCA(n_components=nb_dims)
    else:
        raise ValueError(reduce + " dimensionality reduction method not supported.")

    reduced_activations = projector.fit_transform(activations)
    return reduced_activations
