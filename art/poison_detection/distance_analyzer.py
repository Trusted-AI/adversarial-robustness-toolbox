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

from art.poison_detection.clustering_analyzer import ClusteringAnalyzer

logger = logging.getLogger(__name__)


class DistanceAnalyzer(ClusteringAnalyzer):
    """
    Assigns a cluster as poisonous if its median activation is closer to the median activation for another class
    than it is to the median activation of its own class. Currently, this function assumed there are only
    two clusters per class.
    """
    params = ['separated_activations']

    def __init__(self):
        """
        Create an ClusteringAnalyzer object
        """
        super(DistanceAnalyzer, self).__init__()

    def analyze_clusters(self, separated_clusters, **kwargs):
        """
        Analyze clusters to determine level of suspiciousness of poison.

        :param separated_clusters: list where separated_clusters[i] is the cluster assignments for the ith class
        :type separated_clusters: `list`
        :param separated_activations: list where separated_activations[i] is a 1D array of [0,1] for [poison,clean]
        :type separated_clusters: `list`
        :param kwargs: a dictionary of analysis-specific parameters
        :type kwargs: `dict`
        :return: all_assigned_clean: array where all_assigned_clean[i] is a 1D boolean array indicating whether
        a given data point was determined to be clean (as opposed to poisonous).
        :rtype: all_assigned_clean `ndarray`
        """
        self.set_params(**kwargs)

        all_assigned_clean = []
        cluster_centers = []

        # assign centers
        for t, activations in enumerate(self.separated_activations):
            cluster_centers.append(np.median(activations, axis=0))

        for i, (clusters, ac) in enumerate(zip(separated_clusters, self.separated_activations)):
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
            if cluster1_is_poison:
                poison_clusters.append(1)

            clean_clusters = list(set(np.unique(clusters)) - set(poison_clusters))
            assigned_clean = self.assign_class(clusters, clean_clusters, poison_clusters)
            all_assigned_clean.append(assigned_clean)

        all_assigned_clean = np.asarray(all_assigned_clean)

        return all_assigned_clean

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and applies specific checks before saving them as attributes.

        :param separated_activations: list where separated_activations[i] is a 1D array of [0,1] for [poison,clean]
        :type separated_activations: `list`
        """
        super(DistanceAnalyzer, self).set_params(**kwargs)
