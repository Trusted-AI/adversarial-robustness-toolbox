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

import numpy as np
from art.poison_detection.poison_filtering_defence import PoisonFilteringDefence
from art.poison_detection.clustering_handler import ClusteringHandler
from art.poison_detection.size_analyzer import SizeAnalyzer
from art.poison_detection.distance_analyzer import DistanceAnalyzer
from art.poison_detection.ground_truth_evaluator import GroundTruthEvaluator


class ActivationDefence(PoisonFilteringDefence):
    """
    Class performing Activation Analysis Defence
    """
    defence_params = ['n_clusters', 'clustering_method', 'ndims', 'reduce', 'cluster_analysis']
    valid_clustering = ['KMeans']
    valid_reduce = ['PCA', 'FastICA', 'TSNE']
    valid_analysis = ['smaller', 'distance']
    TOO_SMALL_ACTIVATIONS = 32  # Threshold used to print a warning when activations are not enough

    def __init__(self, classifier, x_train, y_train, verbose=True):
        """
        Create an ActivationDefence object with the provided classifier

        :param classifier: model evaluated for poison
        :type classifier: :class:`Classifier`
        :param x_train: dataset used to train `classifier`
        :type x_train: :class:`numpy.ndarray`
        :param y_train: labels used to train `classifier`
        :type y_train: :class:`numpy.ndarray`
        :param verbose: When True prints more information
        :type verbose: `bool`
        """
        super(ActivationDefence, self).__init__(classifier, x_train, y_train, verbose)
        kwargs = {'n_clusters': 2, 'clustering_method': "KMeans", 'ndims': 10, 'reduce': 'PCA',
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
                                                                                    self.is_clean_by_class,
                                                                                    verbose=self.verbose)
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
        :return: `tuple`
        """
        self.set_params(**kwargs)
        if len(self.activations_by_class) == 0:
            activations = self._get_activations()
            self.activations_by_class = self._segment_by_class(activations, self.y_train)

        my_clust = ClusteringHandler()
        [self.clusters_by_class, self.red_activations_by_class] = my_clust.cluster_activations(
            self.activations_by_class,
            n_clusters=self.n_clusters,
            ndims=self.ndims,
            reduce=self.reduce,
            clustering_method=self.clustering_method)

        return self.clusters_by_class, self.red_activations_by_class

    def analyze_clusters(self, **kwargs):
        """
        This function analyzes the clusters according to the provided method

        :param kwargs: a dictionary of cluster-analysis-specific parameters
        :type kwargs: `dict`
        :return: Assigned_clean_by_class, an array of arrays that contains what data points where classified as clean.
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

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and applies defence-specific checks before saving them as attributes.
        If a parameter is not provided, it takes its default value.

        :param n_clusters: Number of clusters to be produced. Should be greater than 2.
        :type n_clusters: `int`
        :param clustering_method: Clustering method to use
        :type clustering_method: `string`
        :param ndims: Number of dimensions to project on
        :type ndims: `int`
        :param reduce: Reduction technique
        :type reduce: `str`
        :param cluster_analysis: Method to analyze the clusters
        :type cluster_analysis: `str`
        """
        # Save defence-specific parameters
        super(ActivationDefence, self).set_params(**kwargs)

        if self.n_clusters <= 1:
            raise ValueError(
                "Wrong number of clusters, should be greater or equal to 2. Provided: " + str(self.n_clusters))
            return False
        if self.ndims <= 0:
            raise ValueError("Wrong number of dimensions ")
            return False
        if self.clustering_method not in self.valid_clustering:
            raise ValueError("Unsupported clustering method: " + self.clustering_method)
            return False
        if self.reduce not in self.valid_reduce:
            raise ValueError("Unsupported reduction method: " + self.reduce)
            return False
        if self.cluster_analysis not in self.valid_analysis:
            raise ValueError("Unsupported method for cluster analysis method: " + self.cluster_analysis)
            return False

        return True

    def _get_activations(self):
        """
        Find activations from class:Classifier
        """
        print('Getting activations..')

        nb_layers = len(self.classifier.layer_names)
        activations = self.classifier.get_activations(self.x_train, layer=nb_layers - 1)

        # wrong way to get activations activations = self.classifier.predict(self.x_train, logits=True)
        nodes_last_layer = np.shape(activations)[1]

        if nodes_last_layer <= self.TOO_SMALL_ACTIVATIONS:
            print("WARNING: Number of activations in last layer is too small... method may not work properly. "
                  "Size: " + str(nodes_last_layer))
        return activations

    def _segment_by_class(self, data, features):
        """
        Returns segmented data according to specified features

        :param data: to be segmented
        :type data: :class:`numpy.ndarray`
        :param features: features used to segment data
                       e.g., segment according to predicted label or to y_train
        :type features: class:`numpy.ndarray`
        """
        n_classes = self.classifier.nb_classes
        by_class = [[] for i in range(n_classes)]
        for indx, feature in enumerate(features):
            if n_classes > 2:
                assigned = np.argmax(feature)
            else:
                assigned = int(feature)
            by_class[assigned].append(data[indx])

        return [np.asarray(i) for i in by_class]
