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
This module implements a Hidden Trigger Backdoor attack on Neural Networks.

| Paper link: https://arxiv.org/abs/1910.00033
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np

from art.attacks.attack import PoisoningAttackWhiteBox
from art.attacks.poisoning.backdoor_attack import PoisoningAttackBackdoor
from art.estimators import BaseEstimator, NeuralNetworkMixin
from art.estimators.classification.classifier import ClassifierMixin
from art.estimators.classification.pytorch import PyTorchClassifier
from art.estimators.classification.keras import KerasClassifier
from art.estimators.classification.tensorflow import TensorFlowV2Classifier

from art.attacks.poisoning.hidden_trigger_backdoor.hidden_trigger_backdoor_pytorch import (
    HiddenTriggerBackdoorPyTorch,
)
from art.attacks.poisoning.hidden_trigger_backdoor.hidden_trigger_backdoor_keras import (
    HiddenTriggerBackdoorKeras,
)

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_NEURALNETWORK_TYPE

logger = logging.getLogger(__name__)


class HiddenTriggerBackdoor(PoisoningAttackWhiteBox):
    """
    Implementation of Hidden Trigger Backdoor Attack by Saha et al 2019.
    "Hidden Trigger Backdoor Attacks

    | Paper link: https://arxiv.org/abs/1910.00033
    """

    attack_params = PoisoningAttackWhiteBox.attack_params + [
        "target",
        "backdoor",
        "feature_layer",
        "source",
        "eps",
        "learning_rate",
        "decay_coeff",
        "decay_iter",
        "stopping_tol",
        "max_iter",
        "poison_percent",
        "batch_size",
        "verbose",
        "print_iter",
    ]

    _estimator_requirements = (BaseEstimator, NeuralNetworkMixin, ClassifierMixin)

    def __init__(
        self,
        classifier: "CLASSIFIER_NEURALNETWORK_TYPE",
        target: np.ndarray,
        source: np.ndarray,
        feature_layer: Union[str, int],
        backdoor: PoisoningAttackBackdoor,
        eps: float = 0.1,
        learning_rate: float = 0.001,
        decay_coeff: float = 0.95,
        decay_iter: Union[int, List[int]] = 2000,
        stopping_threshold: float = 10,
        max_iter: int = 5000,
        batch_size: float = 100,
        poison_percent: float = 0.1,
        is_index: bool = False,
        verbose: bool = True,
        print_iter: int = 100,
    ) -> None:
        """
        Creates a new Hidden Trigger Backdoor poisoning attack.

        :param classifier: A trained neural network classifier.
        :param target: The target class/indices to poison. Triggers added to inputs not in the target class will
                       result in misclassifications to the target class. If an int, it represents a label.
                       Otherwise, it is an array of indices.
        :param source: The class/indices which will have a trigger added to cause misclassification
                       If an int, it represents a label. Otherwise, it is an array of indices.
        :param feature_layer: The name of the feature representation layer
        :param backdoor: A PoisoningAttackBackdoor that adds a backdoor trigger to the input.
        :param eps: Maximum perturbation that the attacker can introduce.
        :param learning_rate: The learning rate of clean-label attack optimization.
        :param decay_coeff: The decay coefficient of the learning rate.
        :param decay_iter: The number of iterations before the learning rate decays
        :param stopping_threshold: Stop iterations after loss is less than this threshold.
        :param max_iter: The maximum number of iterations for the attack.
        :param batch_size: The number of samples to draw per batch.
        :param poison_percent: The percentage of the data to poison. This is ignored if indices are provided
        :param is_index: If true, the source and target params are assumed to represent indices rather
                         than a class label. poison_percent is ignored if true.
        :param verbose: Show progress bars.
        :param print_iter: The number of iterations to print the current loss progress.
        """
        super().__init__(classifier=classifier)  # type: ignore
        self.target = target
        self.source = source
        self.feature_layer = feature_layer
        self.backdoor = backdoor
        self.eps = eps
        self.learning_rate = learning_rate
        self.decay_coeff = decay_coeff
        self.decay_iter = decay_iter
        self.stopping_threshold = stopping_threshold
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.poison_percent = poison_percent
        self.is_index = is_index
        self.verbose = verbose
        self.print_iter = print_iter
        self._check_params()

        if isinstance(self.estimator, PyTorchClassifier):
            self._attack = HiddenTriggerBackdoorPyTorch(
                classifier=classifier,  # type: ignore
                target=target,
                source=source,
                backdoor=backdoor,
                feature_layer=feature_layer,
                eps=eps,
                learning_rate=learning_rate,
                decay_coeff=decay_coeff,
                decay_iter=decay_iter,
                stopping_threshold=stopping_threshold,
                max_iter=max_iter,
                batch_size=batch_size,
                poison_percent=poison_percent,
                is_index=is_index,
                verbose=verbose,
                print_iter=print_iter,
            )

        elif isinstance(self.estimator, (KerasClassifier, TensorFlowV2Classifier)):
            self._attack = HiddenTriggerBackdoorKeras(  # type: ignore
                classifier=classifier,  # type: ignore
                target=target,
                source=source,
                backdoor=backdoor,
                feature_layer=feature_layer,
                eps=eps,
                learning_rate=learning_rate,
                decay_coeff=decay_coeff,
                decay_iter=decay_iter,
                stopping_threshold=stopping_threshold,
                max_iter=max_iter,
                batch_size=batch_size,
                poison_percent=poison_percent,
                is_index=is_index,
                verbose=verbose,
                print_iter=print_iter,
            )

        else:
            raise ValueError("Only Pytorch, Keras, and TensorFlowV2 classifiers are supported")

    def poison(  # pylint: disable=W0221
        self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calls perturbation function on the dataset x and returns only the perturbed inputs and their
        indices in the dataset.

        :param x: An array in the shape NxCxWxH with the points to draw source and target samples from.
                  Source indicates the class(es) that the backdoor would be added to to cause
                  misclassification into the target label.
                  Target indicates the class that the backdoor should cause misclassification into.
        :param y: The labels of the provided samples. If none, we will use the classifier to label the
                  data.
        :return: An tuple holding the `(poisoning_examples, poisoning_labels)`.
        """

        return self._attack.poison(x, y, **kwargs)

    def _check_params(self) -> None:

        if not isinstance(self.target, np.ndarray) or not isinstance(self.source, np.ndarray):
            raise ValueError("Target and source must be arrays")

        if np.array_equal(self.target, self.source):
            raise ValueError("Target and source values can't be the same")

        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be strictly positive")

        if not isinstance(self.backdoor, PoisoningAttackBackdoor):
            raise TypeError("Backdoor must be of type PoisoningAttackBackdoor")

        if self.eps < 0:
            raise ValueError("The perturbation size `eps` has to be non-negative.")

        if not isinstance(self.feature_layer, (str, int)):
            raise TypeError("Feature layer should be a string or int")

        if isinstance(self.feature_layer, int):
            if not 0 <= self.feature_layer < len(self.estimator.layer_names):
                raise ValueError("feature_layer is not a non-negative integer")

        if self.decay_coeff <= 0:
            raise ValueError("Decay coefficient must be positive")

        if not 0 < self.poison_percent <= 1:
            raise ValueError("poison_percent must be between 0 (exclusive) and 1 (inclusive)")

        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be of type bool.")
