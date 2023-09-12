# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2022
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
This module implements (De)Randomized Smoothing certifications against adversarial patches.

| Paper link: https://arxiv.org/abs/2110.07719

| Paper link: https://arxiv.org/abs/2002.10733
"""

from __future__ import absolute_import, division, print_function, unicode_literals

from abc import ABC, abstractmethod
import numpy as np


class DeRandomizedSmoothingMixin(ABC):
    """
    Mixin class for smoothed estimators.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        """
        Create a derandomized smoothing wrapper.
        """
        super().__init__(*args, **kwargs)  # type: ignore

    @abstractmethod
    def _predict_classifier(self, x: np.ndarray, batch_size: int, training_mode: bool, **kwargs) -> np.ndarray:
        """
        Perform prediction for a batch of inputs.

        :param x: Input samples.
        :param batch_size: Size of batches.
        :param training_mode: `True` for model set to training mode and `'False` for model set to evaluation mode.
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: np.ndarray, batch_size: int = 128, training_mode: bool = False, **kwargs) -> np.ndarray:
        """
        Performs cumulative predictions over every ablation location

        :param x: Unablated image
        :param batch_size: the batch size for the prediction
        :param training_mode: if to run the classifier in training mode
        :return: cumulative predictions after sweeping over all the ablation configurations.
        """
        raise NotImplementedError
