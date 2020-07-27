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

| Paper link: http://people.cs.uchicago.edu/~ravenben/publications/abstracts/backdoor-sp19.html
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, TYPE_CHECKING, Union

import numpy as np

from art.defences.transformer.transformer import Transformer
from art.estimators.certification.neural_cleanse.keras import KerasNeuralCleanse
from art.estimators.classification import KerasClassifier

if TYPE_CHECKING:
    from art.estimators.classification.classifier import Classifier

logger = logging.getLogger(__name__)


class NeuralCleanse(Transformer):
    """
    Implement the defensive distillation mechanism.

    | Paper link: https://arxiv.org/abs/1511.04508
    """

    params = ["steps", "init_cost", "norm", "learning_rate", "attack_success_threshold", "patience", "early_stop",
              "early_stop_threshold", "early_stop_patience", "cost_multiplier", "batch_size"]

    def __init__(self, classifier: "Classifier") -> None:
        """
        Create an instance of the defensive distillation defence.

        :param classifier: A trained classifier.
        :param batch_size: Size of batches.
        :param nb_epochs: Number of epochs to use for training.
        """
        super().__init__(classifier=classifier)
        self._is_fitted = True
        self._check_params()

    def __call__(self, x: np.ndarray, transformed_classifier: "Classifier", mitigation_type: str = "unlearning",
                 steps: int = 1000, init_cost: float = 1e-3, norm: Union[int, float] = 2,
                 learning_rate: float = 0.1, attack_success_threshold: float = 0.99, patience: int = 5,
                 early_stop: bool = True, early_stop_threshold: float = 0.99, early_stop_patience: int = 10,
                 cost_multiplier: float = 1.5, batch_size: int = 32) -> "Classifier":
        """
        Perform the defensive distillation defence mechanism and return a robuster classifier.

        :param x: Dataset for training the transformed classifier.
        :param transformed_classifier: A classifier to be transformed for increased robustness. Note that, the
            objective loss function used for fitting inside the input transformed_classifier must support soft labels,
            i.e. probability labels.
        :return: The transformed classifier.
        """
        if isinstance(transformed_classifier, KerasClassifier):
            transformed_classifier = KerasNeuralCleanse(model=transformed_classifier.model, steps=steps,
                                                        init_cost=init_cost, norm=norm, learning_rate=learning_rate,
                                                        attack_success_threshold=attack_success_threshold,
                                                        patience=patience, early_stop=early_stop,
                                                        early_stop_threshold=early_stop_threshold,
                                                        early_stop_patience=early_stop_patience,
                                                        cost_multiplier=cost_multiplier, batch_size=batch_size)
            return transformed_classifier
        else:
            raise NotImplementedError

    def fit(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> None:
        """
        No parameters to learn for this method; do nothing.
        """
        pass

    def _check_params(self) -> None:
        # TODO: finish this
        pass
        # if not isinstance(self.batch_size, (int, np.int)) or self.batch_size <= 0:
        #     raise ValueError("The size of batches must be a positive integer.")
        #
        # if not isinstance(self.nb_epochs, (int, np.int)) or self.nb_epochs <= 0:
        #     raise ValueError("The number of epochs must be a positive integer.")
