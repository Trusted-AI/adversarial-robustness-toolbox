from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np

from art.poison_detection.clustering_analyzer import ClusteringAnalyzer

logger = logging.getLogger(__name__)


class RelativeSizeAnalyzer(ClusteringAnalyzer):
    """
    Designates as poisonous the cluster with less number of items on it.
    """
    params = ['size_threshold']

    def __init__(self):
        """
        Create an ClusteringAnalyzer object
        """
        super(RelativeSizeAnalyzer, self).__init__()
        self.size_threshold = 0.35

    def analyze_clusters(self, separated_clusters, **kwargs):
        """
        Assigns a cluster as poisonous if the smaller one contains less than threshold of the data.
        This method assumes only 2 clusters

        :param separated_clusters: list where separated_clusters[i] is the cluster assignments for the ith class
        :type separated_clusters: `list`
        :param size_threshold: (optional) threshold used to define when a cluster is substantially smaller. A default
        value is used if the parameter is not provided.
        :type size_threshold: `float`
        :param kwargs: a dictionary of defence-specific parameters
        :type kwargs: `dict`
        :return: all_assigned_clean, summary_poison_clusters:
        where all_assigned_clean[i] is a 1D boolean array indicating whether
        a given data point was determined to be clean (as opposed to poisonous) and
        summary_poison_clusters: array, where  summary_poison_clusters[i][j]=1 if cluster j of class i was classified as
        poison, otherwise 0
        """
        all_assigned_clean = []
        nb_classes = len(separated_clusters)
        nb_clusters = len(np.unique(separated_clusters[0]))
        summary_poison_clusters = [[[] for x in range(nb_clusters)] for y in range(nb_classes)]

        for i, clusters in enumerate(separated_clusters):
            bins = np.bincount(clusters)
            if np.size(bins) > 2:
                raise ValueError(" RelativeSizeAnalyzer does not support more than two clusters.")
            percentages = bins / float(np.sum(bins))
            poison_clusters = np.where(percentages < self.size_threshold)
            clean_clusters = np.where(percentages >= self.size_threshold)

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
        """
        super(RelativeSizeAnalyzer, self).set_params(**kwargs)