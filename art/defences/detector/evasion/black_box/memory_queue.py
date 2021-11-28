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
This module implements memory queue used in black-box detection algorithm.
"""
import logging
import numpy as np
from typing import Tuple


logger = logging.getLogger(__name__)


class MemoryQueue:
    """
    Memory queue for black-box detector.
    """

    def __init__(
        self,
        max_size: int,
        element_shape: Tuple[int, ...],
        initial_elements: np.ndarray = None,
    ):
        """
        Memory queue for black-box detector.

        :param max_size: Maximum number of elements stored in memory queue
        :param initial_elements: Initial elements for memory queue
        """
        self._max_size = max_size
        self._memory = initial_elements
        self._element_shape = element_shape

    def extend(self, elements: np.ndarray):
        """
        Extends queue with new elements
        """
        assert (
            self._element_shape == elements.shape[1:]
        ), f"Shapes does not match! {self._element_shape} != {elements.shape[1:]}"
        if self._memory is not None:
            new_memory = np.concatenate([self._memory, elements], axis=0)
            self._memory = new_memory[-self._max_size :]
        else:
            self._memory = elements[-self._max_size :]

    def get_memory(self):
        return self._memory

    def __len__(self):
        return self._memory.shape[0]
