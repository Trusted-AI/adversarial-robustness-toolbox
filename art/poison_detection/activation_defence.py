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
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import os.path

import numpy as np

from art.poison_detection.distance_analyzer import DistanceAnalyzer
from art.poison_detection.ground_truth_evaluator import GroundTruthEvaluator
from art.poison_detection.poison_filtering_defence import PoisonFilteringDefence
from art.poison_detection.size_analyzer import SizeAnalyzer
from art.visualization import create_sprite, save_image

logger = logging.getLogger(__name__)


class ActivationDefence(PoisonFilteringDefence):
    """
    Method from [Chen et al., 2018] performing poisoning detection based on activations clustering.
    Paper link: https://arxiv.org/abs/1811.03728
    """
    defence_params = ['nb_clusters', 'clustering_method', 'nb_dims', 'reduce', 'cluster_analysis']
    valid_clustering = ['KMeans']
    valid_reduce = ['PCA', 'FastICA', 'TSNE']
    valid_analysis = ['smaller', 'distance']
    TOO_SMALL_ACTIVATIONS = 32  # Threshold used to print a warning when activations are not enough

    def __init__(self, classifier, x_train, y_train):
        """
        Create an :class:ActivationDefence object with the provided classifier.

        :param classifier: model evaluated for poison
        :type classifier: :class:`Classifier`
        :param x_train: dataset used to train `classifier`
        :type x_train: `np.ndarray`
        :param y_train: labels used to train `classifier`
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

    def evaluate_defence(self, is_clean, **kwargs):
        """
        Returns confusion matrix.

        :param is_clean: ground truth, where is_clean[i]=1 means that x_train[i] is clean and is_clean[i]=0 means
                         x_train[i] is poisonous
        :type is_clean: :class `list`
        :param kwargs: a dictionary of defence-specific parameters
        :type kwargs: `dict`
        :return: JSON object with confusion matrix
        :rtype: `jsonObject`
        """
        self.set_params(**kwargs)

        if len(self.activations_by_class) == 0:
            activations = self._get_activations()
            self.activations_by_class = self._segment_by_class(activations, self.y_train)

        self.clusters_by_class, self.red_activations_by_class = self.cluster_activations()
        self.assigned_clean_by_class = self.analyze_clusters()

        # Now check ground truth:
        self.is_clean_by_class = self._segment_by_class(is_clean, self.y_train)
        self.errors_by_class, conf_matrix_json = self.evaluator.analyze_correctness(self.assigned_clean_by_class,
                                                                                    self.is_clean_by_class)
        return conf_matrix_json

    def detect_poison(self, **kwargs):
        """
        Returns poison detected.

        :param kwargs: a dictionary of detection-specific parameters
        :type kwargs: `dict`
        :return: 1) confidence_level, 2) is_clean_lst : type List[int], where is_clean_lst[i]=1 means that x_train[i]
                there is clean and is_clean_lst[i]=0, means that x_train[i] was classified as poison
        :rtype: `tuple`
        """
        self.set_params(**kwargs)

        if len(self.activations_by_class) == 0:
            activations = self._get_activations()
            self.activations_by_class = self._segment_by_class(activations, self.y_train)
        self.clusters_by_class, self.red_activations_by_class = self.cluster_activations()
        self.assigned_clean_by_class = self.analyze_clusters()
        # Here, assigned_clean_by_class[i][j] is 1 if the jth datapoint in the ith class was
        # determined to be clean by activation cluster

        # Build an array that matches the original indexes of x_train
        n_train = len(self.x_train)
        indices_by_class = self._segment_by_class(np.arange(n_train), self.y_train)
        self.is_clean_lst = [0] * n_train
        self.confidence_level = [1] * n_train
        for i, (assigned_clean, dp) in enumerate(zip(self.assigned_clean_by_class, indices_by_class)):
            for j, (assignment, index_dp) in enumerate(zip(assigned_clean, dp)):
                if assignment == 1:
                    self.is_clean_lst[index_dp] = 1

        return self.confidence_level, self.is_clean_lst

    def cluster_activations(self, **kwargs):
        """
        Clusters activations and returns cluster_by_class and red_activations_by_class,
        where cluster_by_class[i][j] is the cluster to which the j-th datapoint in the
        ith class belongs and the correspondent activations reduced by class
        red_activations_by_class[i][j]

        :param kwargs: a dictionary of cluster-specific parameters
        :type kwargs: `dict`
        :return: clusters per class and activations by class
        :rtype: `tuple`
        """
        self.set_params(**kwargs)
        if len(self.activations_by_class) == 0:
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
        This function analyzes the clusters according to the provided method

        :param kwargs: a dictionary of cluster-analysis-specific parameters
        :type kwargs: `dict`
        :return: assigned_clean_by_class, an array of arrays that contains what data points where classified as clean.
        :rtype: `np.ndarray`
        """
        self.set_params(**kwargs)

        if len(self.clusters_by_class) == 0:
            self.cluster_activations()

        if self.cluster_analysis == 'smaller':
            analyzer = SizeAnalyzer()
            self.assigned_clean_by_class = analyzer.analyze_clusters(self.clusters_by_class)
        elif self.cluster_analysis == 'distance':
            analyzer = DistanceAnalyzer()
            self.assigned_clean_by_class = analyzer.analyze_clusters(self.clusters_by_class,
                                                                     separated_activations=self.red_activations_by_class)
        return self.assigned_clean_by_class

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
        :return: sprites_by_class: Array with sprite images sprites_by_class, where sprites_by_class[i][j] contains the
                 sprite of class i cluster j.
        :rtype: sprites_by_class: `np.ndarray`
        """
        self.set_params(**kwargs)

        if len(self.clusters_by_class) == 0:
            self.cluster_activations()

        x_raw_by_class = self._segment_by_class(x_raw, self.y_train)
        x_raw_by_cluster = [[[] for x in range(self.nb_clusters)] for y in range(self.classifier.nb_classes)]

        # Get all data in x_raw in the right cluster
        for n_class, cluster in enumerate(self.clusters_by_class):
            for j, assigned_cluster in enumerate(cluster):
                x_raw_by_cluster[n_class][assigned_cluster].append(x_raw_by_class[n_class][j])

        # Now create sprites:
        sprites_by_class = [[[] for x in range(self.nb_clusters)] for y in range(self.classifier.nb_classes)]
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
        Find activations from :class:`Classifier`
        """
        logger.info('Getting activations')

        nb_layers = len(self.classifier.layer_names)
        activations = self.classifier.get_activations(self.x_train, layer=nb_layers - 1)

        # wrong way to get activations activations = self.classifier.predict(self.x_train, logits=True)
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
        n_classes = self.classifier.nb_classes
        by_class = [[] for _ in range(n_classes)]
        for indx, feature in enumerate(features):
            if n_classes > 2:
                assigned = np.argmax(feature)
            else:
                assigned = int(feature)
            by_class[assigned].append(data[indx])

        return [np.asarray(i) for i in by_class]


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
    from sklearn.cluster import KMeans
    from sklearn.decomposition import FastICA, PCA

    separated_clusters = []
    separated_reduced_activations = []

    if reduce == 'FastICA':
        projector = FastICA(n_components=nb_dims, max_iter=1000, tol=0.005)
    elif reduce == 'PCA':
        projector = PCA(n_components=nb_dims)
    else:
        raise ValueError(reduce + " dimensionality reduction method not supported.")

    if clustering_method == 'KMeans':
        clusterer = KMeans(n_clusters=nb_clusters)
    else:
        raise ValueError(clustering_method + " clustering method not supported.")

    for i, ac in enumerate(separated_activations):
        # Apply dimensionality reduction
        nb_activations = np.shape(ac)[1]
        if nb_activations > nb_dims:
            reduced_activations = projector.fit_transform(ac)
        else:
            logger.info("Dimensionality of activations = %i less than nb_dims = %i. Not applying dimensionality "
                        "reduction.", nb_activations, nb_dims)
            reduced_activations = ac
        separated_reduced_activations.append(reduced_activations)

        # Get cluster assignments
        clusters = clusterer.fit_predict(reduced_activations)
        separated_clusters.append(clusters)

    return separated_clusters, separated_reduced_activations
