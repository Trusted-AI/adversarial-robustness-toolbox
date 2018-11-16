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


class SizeAnalyzer(ClusteringAnalyzer):
    """
    Designates as poisonous the cluster with less number of items on it.
    """

    def __init__(self):
        """
        Create an ClusteringAnalyzer object
        """
        super(SizeAnalyzer, self).__init__()

    def analyze_clusters(self, separated_clusters, **kwargs):
        """
        Designates as poisonous the cluster with less number of items on it.

        :param separated_clusters: list where separated_clusters[i] is the cluster assignments for the ith class
        :type separated_clusters: `list`
        :param kwargs: a dictionary of defence-specific parameters
        :type kwargs: `dict`
        :return: all_assigned_clean: array where all_assigned_clean[i] is a 1D boolean array indicating whether
        a given data point was determined to be clean (as opposed to poisonous)
        :rtype: all_assigned_clean: `ndarray`
        """
        all_assigned_clean = []

        for i, clusters in enumerate(separated_clusters):
            # assume that smallest cluster is poisonous and all others are clean
            poison_clusters = [np.argmin(np.bincount(clusters))]
            clean_clusters = list(set(np.unique(clusters)) - set(poison_clusters))

            assigned_clean = self.assign_class(clusters, clean_clusters, poison_clusters)
            all_assigned_clean.append(assigned_clean)

        return np.asarray(all_assigned_clean)
