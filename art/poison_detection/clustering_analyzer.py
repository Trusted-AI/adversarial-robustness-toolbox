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

import numpy as np

logger = logging.getLogger(__name__)


class ClusteringAnalyzer:
    """
    Class for all methodologies implemented to analyze clusters and determine whether they are poisonous.
    """

    def __init__(self):
        """
        Constructor
        """

    @staticmethod
    def assign_class(clusters, clean_clusters, poison_clusters):
        """
        Determines whether each data point in the class is in a clean or poisonous cluster

        :param clusters: clusters[i] indicates which cluster the i'th data point is in
        :type clusters: `list`
        :param clean_clusters: list containing the clusters designated as clean
        :type clean_clusters: `list`
        :param poison_clusters: list containing the clusters designated as poisonous
        :type poison_clusters `list`
        :return: assigned_clean: assigned_clean[i] is a boolean indicating whether the ith data point is clean
        """
        assigned_clean = np.empty(np.shape(clusters))
        assigned_clean[np.isin(clusters, clean_clusters)] = 1
        assigned_clean[np.isin(clusters, poison_clusters)] = 0
        return assigned_clean

    def analyze_by_size(self, separated_clusters):
        """
        Designates as poisonous the cluster with less number of items on it.

        :param separated_clusters: list where separated_clusters[i] is the cluster assignments for the ith class
        :type separated_clusters: `list`
        :return: all_assigned_clean, summary_poison_clusters:
        where all_assigned_clean[i] is a 1D boolean array indicating whether
        a given data point was determined to be clean (as opposed to poisonous) and
        summary_poison_clusters: array, where  summary_poison_clusters[i][j]=1 if cluster j of class i was classified as
        poison, otherwise 0
        :rtype: all_assigned_clean: `ndarray`, summary_poison_clusters: `list`
        """
        all_assigned_clean = []
        nb_classes = len(separated_clusters)
        nb_clusters = len(np.unique(separated_clusters[0]))
        summary_poison_clusters = [[[] for x in range(nb_clusters)] for y in range(nb_classes)]

        for i, clusters in enumerate(separated_clusters):
            # assume that smallest cluster is poisonous and all others are clean
            poison_clusters = [np.argmin(np.bincount(clusters))]
            clean_clusters = list(set(clusters) - set(poison_clusters))

            for p_id in poison_clusters:
                summary_poison_clusters[i][p_id] = 1
            for c_id in clean_clusters:
                summary_poison_clusters[i][c_id] = 0

            assigned_clean = self.assign_class(clusters, clean_clusters, poison_clusters)
            all_assigned_clean.append(assigned_clean)

        return np.asarray(all_assigned_clean), summary_poison_clusters

    def analyze_by_distance(self, separated_clusters, separated_activations):
        """
        Assigns a cluster as poisonous if its median activation is closer to the median activation for another class
        than it is to the median activation of its own class. Currently, this function assumed there are only
        two clusters per class.

        :param separated_clusters: list where separated_clusters[i] is the cluster assignments for the ith class
        :type separated_clusters: `list`
        :param separated_activations: list where separated_activations[i] is a 1D array of [0,1] for [poison,clean]
        :type separated_clusters: `list`
        :return: all_assigned_clean, summary_poison_clusters:
        where all_assigned_clean[i] is a 1D boolean array indicating whether
        a given data point was determined to be clean (as opposed to poisonous) and
        summary_poison_clusters: array, where  summary_poison_clusters[i][j]=1 if cluster j of class i was classified as
        poison, otherwise 0
        :rtype: all_assigned_clean: `ndarray`, summary_poison_clusters: `list`
        """

        all_assigned_clean = []
        cluster_centers = []

        nb_classes = len(separated_clusters)
        nb_clusters = len(np.unique(separated_clusters[0]))
        summary_poison_clusters = [[[] for x in range(nb_clusters)] for y in range(nb_classes)]

        # assign centers
        for t, activations in enumerate(separated_activations):
            cluster_centers.append(np.median(activations, axis=0))

        for i, (clusters, ac) in enumerate(zip(separated_clusters, separated_activations)):
            clusters = np.array(clusters)

            cluster0_center = np.median(ac[np.where(clusters == 0)], axis=0)
            cluster1_center = np.median(ac[np.where(clusters == 1)], axis=0)

            cluster0_distance = np.linalg.norm(cluster0_center - cluster_centers[i])
            cluster1_distance = np.linalg.norm(cluster1_center - cluster_centers[i])

            cluster0_is_poison = False
            cluster1_is_poison = False

            for k, center in enumerate(cluster_centers):
                if k == i:
                    pass
                else:
                    cluster0_distance_to_k = np.linalg.norm(cluster0_center - center)
                    cluster1_distance_to_k = np.linalg.norm(cluster1_center - center)
                    if cluster0_distance_to_k < cluster0_distance and cluster1_distance_to_k > cluster1_distance:
                        cluster0_is_poison = True
                    if cluster1_distance_to_k < cluster1_distance and cluster0_distance_to_k > cluster0_distance:
                        cluster1_is_poison = True

            poison_clusters = []

            if cluster0_is_poison:
                poison_clusters.append(0)
                summary_poison_clusters[i][0] = 1
            else:
                summary_poison_clusters[i][0] = 0

            if cluster1_is_poison:
                poison_clusters.append(1)
                summary_poison_clusters[i][1] = 1
            else:
                summary_poison_clusters[i][1] = 0

            clean_clusters = list(set(clusters) - set(poison_clusters))
            assigned_clean = self.assign_class(clusters, clean_clusters, poison_clusters)
            all_assigned_clean.append(assigned_clean)

        all_assigned_clean = np.asarray(all_assigned_clean)

        return all_assigned_clean, summary_poison_clusters

    def analyze_by_relative_size(self, separated_clusters, size_threshold=0.35):
        """
        Assigns a cluster as poisonous if the smaller one contains less than threshold of the data.
        This method assumes only 2 clusters

        :param separated_clusters: list where separated_clusters[i] is the cluster assignments for the ith class
        :type separated_clusters: `list`
        :param size_threshold: (optional) threshold used to define when a cluster is substantially smaller. A default
        value is used if the parameter is not provided.
        :type size_threshold: `float`
        :return: all_assigned_clean, summary_poison_clusters:
        where all_assigned_clean[i] is a 1D boolean array indicating whether
        a given data point was determined to be clean (as opposed to poisonous) and
        summary_poison_clusters: array, where  summary_poison_clusters[i][j]=1 if cluster j of class i was classified as
        poison, otherwise 0
        :rtype: all_assigned_clean: `ndarray`, summary_poison_clusters: `list`
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
            poison_clusters = np.where(percentages < size_threshold)
            clean_clusters = np.where(percentages >= size_threshold)

            for p_id in poison_clusters[0]:
                summary_poison_clusters[i][p_id] = 1
            for c_id in clean_clusters[0]:
                summary_poison_clusters[i][c_id] = 0

            assigned_clean = self.assign_class(clusters, clean_clusters, poison_clusters)
            all_assigned_clean.append(assigned_clean)

        return np.asarray(all_assigned_clean), summary_poison_clusters

    def analyze_by_sihouette_score(self, separated_clusters, reduced_activations_by_class, size_threshold=0.35,
                                   silhouette_threshold=0.1):
        """
        Computes a silhouette score for each class to determine how cohesive resulting clusters are.
        A low silhouette score indicates that the clustering does not fit the data well, and the class can be considered
        to be unpoisoned. Conversely, a high silhouette score indicates that the clusters reflect true splits in the data.
        The method concludes that a cluster is poison based on the silhouette score and the cluster relative size.
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
        :return: all_assigned_clean, summary_poison_clusters:
        where all_assigned_clean[i] is a 1D boolean array indicating whether
        a given data point was determined to be clean (as opposed to poisonous)
        summary_poison_clusters: array, where  summary_poison_clusters[i][j]=1 if cluster j of class j was classified as
        poison
        :rtype: all_assigned_clean: `ndarray`, summary_poison_clusters: `list`
        """
        from sklearn.metrics import silhouette_score

        all_assigned_clean = []
        nb_classes = len(separated_clusters)
        nb_clusters = len(np.unique(separated_clusters[0]))
        summary_poison_clusters = [[[] for x in range(nb_clusters)] for y in range(nb_classes)]

        for i, (clusters, activations) in enumerate(zip(separated_clusters, reduced_activations_by_class)):
            bins = np.bincount(clusters)
            if np.size(bins) > 2:
                raise ValueError("Analyzer does not support more than two clusters.")
            percentages = bins / float(np.sum(bins))
            poison_clusters = np.where(percentages < size_threshold)
            clean_clusters = np.where(percentages >= size_threshold)

            if np.shape(poison_clusters)[1] != 0:
                # Only compute this score when the the relative size of the clusters is suspicious
                silhouette_avg = silhouette_score(activations, clusters)

                if silhouette_avg > silhouette_threshold:
                    # In this case the cluster is considered poisonous
                    clean_clusters = np.where(percentages >= size_threshold)
                    logger.info('computed silhouette score: ', silhouette_avg)
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
