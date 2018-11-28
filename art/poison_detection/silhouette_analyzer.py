from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np

from sklearn.metrics import silhouette_score
from art.poison_detection.clustering_analyzer import ClusteringAnalyzer

logger = logging.getLogger(__name__)


class SilhouetteAnalyzer(ClusteringAnalyzer):
    """
    Computes a silhouette score for each class to determine how cohesive resulting clusters are.
    A low silhouette score indicates that the clustering does not fit the data well, and the class can be considered
    to be unpoisoned. Conversely, a high silhouette score indicates that the clusters reflect true splits in the data.
    The method concludes that a cluster is poison based on the silhouette score and the cluster relative size.
    """
    params = ['reduced_activations_by_class', 'size_threshold', 'silhouette_threshold']

    def __init__(self):
        """
        Create an ClusteringAnalyzer object
        """
        super(SilhouetteAnalyzer, self).__init__()
        self.silhouette_threshold = 0.1
        self.size_threshold = 0.35

    def analyze_clusters(self, separated_clusters, **kwargs):
        """
        Analyze clusters to determine level of suspiciousness of poison based on the cluster's relative size
        and silhouette score. If the relative size is too small, below a size_threshold and at the same time
        the silhouette score is higher than silhouette_threshold, the cluster is classified as poisonous.
        If the above thresholds are not provided, the default ones will be used.

        :param separated_clusters: list where separated_clusters[i] is the cluster assignments for the ith class
        :type separated_clusters: `list`
        :param reduced_activations_by_class: list where separated_activations[i] is a 1D array of [0,1] for [poison,clean]
        :type reduced_activations_by_class: `list`
        :param size_threshold: (optional) threshold used to define when a cluster is substantially smaller. A default
        value is used if the parameter is not provided.
        :type size_threshold: `float`
        :param silhouette_threshold: (optional) threshold used to define when a cluster is cohesive. Default
        value is used if the parameter is not provided.
        :type silhouette_threshold: `float`
        :param kwargs: a dictionary of analysis-specific parameters, including optional threshold parameters:
        silhouette_threshold and size_threshold
        :type kwargs: `dict`
        :return: all_assigned_clean, summary_poison_clusters:
        where all_assigned_clean[i] is a 1D boolean array indicating whether
        a given data point was determined to be clean (as opposed to poisonous)
        summary_poison_clusters: array, where  summary_poison_clusters[i][j]=1 if cluster j of class j was classified as
        poison
        """
        self.set_params(**kwargs)

        all_assigned_clean = []
        nb_classes = len(separated_clusters)
        nb_clusters = len(np.unique(separated_clusters[0]))
        summary_poison_clusters = [[[] for x in range(nb_clusters)] for y in range(nb_classes)]

        for i, (clusters, activations) in enumerate(zip(separated_clusters,self.reduced_activations_by_class)):
            bins = np.bincount(clusters)
            if np.size(bins) > 2:
                raise ValueError("Analyzer does not support more than two clusters.")
            percentages = bins / float(np.sum(bins))
            poison_clusters = np.where(percentages < self.size_threshold)
            clean_clusters = np.where(percentages >= self.size_threshold)

            if np.shape(poison_clusters)[1] != 0:
                # Only compute this score when the the relative size of the clusters is suspicious
                silhouette_avg = silhouette_score(activations, clusters)

                if silhouette_avg > self.silhouette_threshold:
                    # In this case the cluster is considered poisonous
                    clean_clusters = np.where(percentages >= self.size_threshold)
                    print('computed silhouette score: ', silhouette_avg)
                else:
                    poison_clusters = [[]]
                    clean_clusters = np.where(percentages >= 0)

            for p_id in poison_clusters[0]:
                summary_poison_clusters[i][p_id] = 1
            for c_id in clean_clusters[0]:
                summary_poison_clusters[i][c_id] = 0

            assigned_clean = self.assign_class(clusters, clean_clusters, poison_clusters)
            all_assigned_clean.append(assigned_clean)

        return np.asarray(all_assigned_clean), summary_poison_clusters

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and applies specific checks before saving them as attributes.

        :param size_threshold: (optional) threshold used to define when a cluster is substantially smaller. A default
        value is used if the parameter is not provided.
        :type size_threshold: `float`
        :param silhouette_threshold: (optional) threshold used to define when a cluster is cohesive. Default
        value is used if the parameter is not provided.
        :type silhouette_threshold: `float`
        """
        super(SilhouetteAnalyzer, self).set_params(**kwargs)
