# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
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
This module implements Subpopulation Poisoning Attacks to poison data used in ML models.
"""
from __future__ import absolute_import, division, print_function, unicode_literals, annotations

import logging

import numpy as np
from sklearn.cluster import KMeans

from art.attacks.attack import Attack

logger = logging.getLogger(__name__)

class PoisoningAttackSubpopulationPoisoning(Attack):
    """
    Implementation of the ClusterMatch Subpopulation Data Poisoning Attack introduced in Jagielski et al., 2021.

    Identified subpopulations within a dataset via ClusterMatch filter function,
    and performs label flipping on data points from a subpopulation to create poison instances to
    be injected into the training dataset.
    | Paper link: https://arxiv.org/abs/2006.14026
    """
    def __init__(
            self,
            aux_data,
            aux_labels,
            test_data,
            test_labels,
            n_classes,
            n_clusters = 100,
            poison_rates = [0.5, 1, 2]
    ):
        """
        Initialise a Subpopulation Poisoning Attack.
        """
        super().__init__()
        self.aux_data = np.array(aux_data),
        self.aux_labels = np.array(aux_labels)
        self.test_data = np.array(test_data)
        self.test_labels = np.array(test_labels)
        self.n_classes = n_classes
        self.n_clusters = n_clusters
        self.poison_rates = poison_rates

        self._check_params()

    def cluster_match(self):
        """
        ClusterMatch filter function for identifying natural subpopulations in data,
        using KMeans clustering with default 100 clusters.
        Optional (but recommended) preprocessing for high-dimensional data: extract
        features from data, then using PCA or t-SNE to reduce dimensionality 
        before clustering.
        Assumes classes are determined by integers, and are 0-indexed
        Returns:
            - poison_dict: a dictionary with keys as tuples of (subpopulation index, poison rate index)
            - values as dictionaries containing:
                - x_test_samples: test data samples from the subpopulation
                - y_test_samples: test data labels from the subpopulation
                - x_poison_samples: poison data samples from the subpopulation - to be added to training data
                - y_poison_samples: poison data labels from the subpopulation - to be added to training data
                - cluster count: number of samples in the subpopulation
                - aux indices: dataset indices of the auxiliary data samples in the subpopulation
                - test indices: dataset indices of the test data samples in the subpopulation
                - poison rate: the rate of poison samples to be added to training data
                - poison count: number of poison samples to be added to training data
        """

        self.kmeans_model = KMeans(n_clusters=self.n_clusters, random_state=0)
        self.kmeans_model.fit(self.aux_data)

        test_km = self.kmeans_model.predict(self.test_data)

        cluster_indices, cluster_counts = np.unique(self.kmeans_model.labels_, return_counts=True)

        subpopulations = [(subpop, count) for subpop, count in zip(cluster_indices, cluster_counts)]

        poison_dict = {}

        for i, (index, count) in enumerate(subpopulations):

            test_indices = np.where(test_km == index)[0]
            x_test_samples, y_test_samples = self.test_data[test_indices], self.test_labels[test_indices]

            aux_indices = np.where(self.kmeans_model.labels_ == index)[0]
            x_aux_samples, y_aux_samples = self.aux_data[aux_indices], self.aux_labels[aux_indices]

            for j, pois_count in enumerate([int(count * rate) for rate in self.poison_rates]):
                
                poison_indices = np.random.choice(x_aux_samples.shape[0], pois_count, replace=False)
                x_poison_samples = x_aux_samples[poison_indices]
                y_poison_samples = np.full(len(x_poison_samples), np.random.randint(0, self.n_classes))

                poison_dict[(i, j)] = {
                    "x_test_samples": x_test_samples,
                    "y_test_samples": y_test_samples,
                    "x_poison_samples": x_poison_samples,
                    "y_poison_samples": y_poison_samples,
                    "cluster count": count,
                    "aux indices": aux_indices,
                    "test indices": test_indices,
                    "poison rate": self.poison_rates[j],
                    "poison count": pois_count,
                }

        return poison_dict
    
    def _check_params(self):
        if len(self.aux_data) == 0:
            raise ValueError("Auxiliary data must be provided. It should be a subset of the training data.")
        if len(self.aux_labels) == 0:
            raise ValueError("Auxiliary labels must be provided.")
        if len(self.test_data) == 0:
            raise ValueError("Test data must be provided.")
        if len(self.test_labels) == 0:
            raise ValueError("Test labels must be provided.")
        if len(self.n_classes) == 0:
            raise ValueError("Number of classes must be non-zero.")
        if self.n_clusters < 2:
            raise ValueError("There must be at least two clusters.")
        if len(self.poison_rates) == 0:
            raise ValueError("Poison rates must be provided.")
        for i in self.poison_rates:
            if i <= 0:
                raise ValueError("Poison rates must be non-zero and positive.")