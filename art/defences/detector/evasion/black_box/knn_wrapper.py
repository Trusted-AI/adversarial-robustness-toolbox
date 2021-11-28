# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2021
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
This module implements wrapper for sklearn knn algorithm.
"""
import logging
from typing import Callable, Union, Tuple, Any

import numpy as np
from sklearn.neighbors import NearestNeighbors

from art.defences.detector.evasion.black_box.contrastive_loss import np_contrastive_loss
from art.defences.detector.evasion.black_box.memory_queue import MemoryQueue

logger = logging.getLogger(__name__)


class NearestNeighborsWrapper:
    """
    Wrapper to sklearn NearestNeighbors class
    """

    def __init__(
        self,
        k_neighbors: int,
        memory_queue: MemoryQueue,
        distance_function: Callable = np_contrastive_loss,
    ):
        """
        Wrapper to sklearn NearestNeighbors class

        :param k_neighbors: Number of neighbors to witch we calculate mean distance
        :param memory_queue: Memory queue from where we feed data to algorithm
        :param distance_function: Distance function between elements
        """
        self._k_neighbors = k_neighbors
        self._distance_function = distance_function
        self._memory_queue = memory_queue
        self._knn = NearestNeighbors(n_neighbors=self._k_neighbors)

    def __call__(
        self, query: np.ndarray, with_distances: bool = False
    ) -> Union[Tuple[np.ndarray, Any], np.ndarray]:
        """
        :param query: Query for which we calculate mean distance
        :returns: Mean distance between K nearest neighbors and query
        """
        k_distances, _ = self._knn.kneighbors(query)
        k_distance = np.mean(k_distances)
        if with_distances:
            return k_distance, k_distances
        return k_distance

    def update_memory(self):
        self._knn.fit(self._memory_queue.get_memory())
