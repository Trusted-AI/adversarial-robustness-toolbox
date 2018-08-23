from __future__ import absolute_import, division, print_function, unicode_literals

from sklearn.decomposition import FastICA, PCA
from sklearn.cluster import KMeans
import numpy as np


class ClusteringHandler:
    """
    Class in charge of clustering activations by class
    """
    def __init__(self):
        """
        Constructor
        """

    @staticmethod
    def cluster_activations(separated_activations, nb_clusters=2, nb_dims=10, reduce='FastICA',
                            clustering_method='KMeans'):
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
                print("Dimensionality of activations = %i less than nb_dims = %i. Not applying dimensionality "
                      "reduction..." % (nb_activations, nb_dims))
                reduced_activations = ac
            separated_reduced_activations.append(reduced_activations)

            # Get cluster assignments
            clusters = clusterer.fit_predict(reduced_activations)
            separated_clusters.append(clusters)

        return separated_clusters, separated_reduced_activations
