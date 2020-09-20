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
This module implements the transforming defence mechanism of defensive distillation.

| Paper link: https://arxiv.org/abs/1511.04508
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, TYPE_CHECKING

import numpy as np

from art.defences.transformer.transformer import Transformer
from art.utils import is_probability

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE

logger = logging.getLogger(__name__)


class DefensiveDistillation(Transformer):
    """
    Implement the defensive distillation mechanism.

    | Paper link: https://arxiv.org/abs/1511.04508
    """

    params = ["batch_size", "nb_epochs"]

    def __init__(self, classifier: "CLASSIFIER_TYPE", batch_size: int = 128, nb_epochs: int = 10) -> None:
        """
        Create an instance of the defensive distillation defence.

        :param classifier: A trained classifier.
        :param batch_size: Size of batches.
        :param nb_epochs: Number of epochs to use for training.
        """
        super().__init__(classifier=classifier)
        self._is_fitted = True
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self._check_params()

    def __call__(self, x: np.ndarray, transformed_classifier: "CLASSIFIER_TYPE") -> "CLASSIFIER_TYPE":
        """
        Perform the defensive distillation defence mechanism and return a robuster classifier.

        :param x: Dataset for training the transformed classifier.
        :param transformed_classifier: A classifier to be transformed for increased robustness. Note that, the
            objective loss function used for fitting inside the input transformed_classifier must support soft labels,
            i.e. probability labels.
        :return: The transformed classifier.
        """
        # Check if the trained classifier produces probability outputs
        preds = self.classifier.predict(x=x, batch_size=self.batch_size)
        are_probability = [is_probability(y) for y in preds]
        all_probability = np.sum(are_probability) == preds.shape[0]

        if not all_probability:
            raise ValueError("The input trained classifier do not produce probability outputs.")

        # Check if the transformed classifier produces probability outputs
        transformed_preds = transformed_classifier.predict(x=x, batch_size=self.batch_size)
        are_probability = [is_probability(y) for y in transformed_preds]
        all_probability = np.sum(are_probability) == transformed_preds.shape[0]

        if not all_probability:
            raise ValueError("The input transformed classifier do not produce probability outputs.")

        # Train the transformed classifier with soft labels
        transformed_classifier.fit(x=x, y=preds, batch_size=self.batch_size, nb_epochs=self.nb_epochs)

        return transformed_classifier

    def fit(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> None:
        """
        No parameters to learn for this method; do nothing.
        """
        pass

    def _check_params(self) -> None:
        if not isinstance(self.batch_size, (int, np.int)) or self.batch_size <= 0:
            raise ValueError("The size of batches must be a positive integer.")

        if not isinstance(self.nb_epochs, (int, np.int)) or self.nb_epochs <= 0:
            raise ValueError("The number of epochs must be a positive integer.")
