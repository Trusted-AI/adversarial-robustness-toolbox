# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
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
This module implements Adversarial Embeddings as described by Tan, Shokri (2019).

| Paper link: https://arxiv.org/abs/1905.13409
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from abc import ABC

import logging
from typing import Union

import numpy as np

from art.attacks.poisoning import PoisoningAttackBackdoor

logger = logging.getLogger(__name__)


class AdversarialEmbeddingMixin(ABC):
    """
    Implementation of Adversarial Embedding as introduced by Tan, Shokri (2019).

    | Paper link: https://arxiv.org/abs/1905.13409
    """

    def __init__(self, feature_layer: Union[int, str], backdoor: PoisoningAttackBackdoor, target: np.ndarray, *args,
                 pp_poison: float = 0.05, discriminator_layer_1: int = 256, discriminator_layer_2: int = 128,
                 regularization=10, verbose=False, detect_threshold=0.8, **kwargs,) -> None:
        """
        Create a Adversarial Backdoor Embedding wrapper. This defines the discriminator network and new loss and fit
        function.

        :param feature_layer: The layer of the original network to extract features from
        :param backdoor: The backdoor attack to use in training
        :param target: The target label to poison
        :param pp_poison: The percentage of training data to poison
        :param discriminator_layer_1: The size of the first discriminator layer
        :param discriminator_layer_2: The size of the second discriminator layer
        :param regularization: The regularization constant for the backdoor recognition part of the loss function
        :param verbose: If true, output whether predictions are suspected backdoors
        :param detect_threshold: The probability threshold for detecting backdoors in verbose mode
        """
        super().__init__(*args, **kwargs)  # type: ignore
        self.feature_layer = feature_layer
        self.backdoor = backdoor
        self.target = target
        self.pp_poison = pp_poison
        self.discriminator_layer_1 = discriminator_layer_1
        self.discriminator_layer_2 = discriminator_layer_2
        self.regularization = regularization
        self.verbose = verbose
        self.detect_threshold = detect_threshold

    # pylint: disable=W0221
    def predict(self, x: np.ndarray, batch_size: int = 128, **kwargs) -> np.ndarray:
        """
        Perform prediction of the given classifier for a batch of inputs, taking an expectation over transformations.

        :param x: Test set.
        :param batch_size: Batch size.
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        """
        raise NotImplementedError

    def fit(self, x: np.ndarray, y: np.ndarray, batch_size: int = 128, nb_epochs: int = 10, **kwargs) -> None:
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Training data.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :param batch_size: Batch size.
        :param nb_epochs: Number of epochs to use for training.
        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported for PyTorch
               and providing it takes no effect.
        """
        raise NotImplementedError
