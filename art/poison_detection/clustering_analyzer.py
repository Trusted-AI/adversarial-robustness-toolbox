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

import abc
import sys

import numpy as np

# Ensure compatibility with Python 2 and 3 when using ABCMeta
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str('ABC'), (), {})


class ClusteringAnalyzer(ABC):
    """
      Base class for all methodologies implemented to analyze clusters and determine whether they are poisonous
    """

    def __init__(self):
        """
        Constructor
        """

    @abc.abstractmethod
    def analyze_clusters(self, separated_clusters, **kwargs):
        """
        Analyzes the provided clusters

        :param separated_clusters: list where separated_clusters[i] is the cluster assignments for the ith class
        :type `list`
        :param kwargs: a dictionary of cluster-analysis-specific parameters
        :type kwargs: `dict`
        :return:
        """
        raise NotImplementedError

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

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and apply attack-specific checks before saving them as attributes.

        :param kwargs: a dictionary of attack-specific parameters
        :type kwargs: `dict`
        :return: `True` when parsing was successful
        """
        for key, value in kwargs.items():
            if key in self.params:
                setattr(self, key, value)
        return True
