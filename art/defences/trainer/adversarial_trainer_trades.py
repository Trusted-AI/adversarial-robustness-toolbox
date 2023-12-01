# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2023
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
This module implements adversarial training with TRADES protocol.

| Paper link: https://proceedings.mlr.press/v97/zhang19p.html

| It was noted that this protocol uses a modified loss called TRADES loss which is a combination of cross entropy
loss on clean data and KL divergence loss between clean data and adversarial data. Consequently, framework specific
implementations are being provided in ART.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import abc
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np

from art.defences.trainer.trainer import Trainer
from art.attacks.attack import EvasionAttack
from art.data_generators import DataGenerator

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_LOSS_GRADIENTS_TYPE


class AdversarialTrainerTRADES(Trainer, abc.ABC):
    """
    This is abstract class for different backend-specific implementations of TRADES protocol
    for adversarial training.

    | Paper link: https://proceedings.mlr.press/v97/zhang19p.html
    """

    def __init__(
        self,
        classifier: "CLASSIFIER_LOSS_GRADIENTS_TYPE",
        attack: EvasionAttack,
        beta: float = 6.0,
    ):
        """
        Create an :class:`.AdversarialTrainerTRADES` instance.

        :param classifier: Model to train adversarially.
        :param attack: attack to use for data augmentation in adversarial training
        :param beta: The scaling factor controlling tradeoff between clean loss and adversarial loss
        """
        self._attack = attack
        self._beta = beta
        super().__init__(classifier)

    @abc.abstractmethod
    def fit(  # pylint: disable=W0221
        self,
        x: np.ndarray,
        y: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        batch_size: int = 128,
        nb_epochs: int = 20,
        **kwargs
    ):
        """
        Train a model adversarially with TRADES. See class documentation for more information on the exact procedure.

        :param x: Training set.
        :param y: Labels for the training set.
        :param validation_data: Tuple consisting of validation data, (x_val, y_val)
        :param batch_size: Size of batches.
        :param nb_epochs: Number of epochs to use for trainings.
        :param kwargs: Dictionary of framework-specific arguments. These will be passed as such to the `fit` function of
               the target classifier.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fit_generator(self, generator: DataGenerator, nb_epochs: int = 20, **kwargs):
        """
        Train a model adversarially using a data generator.
        See class documentation for more information on the exact procedure.

        :param generator: Data generator.
        :param nb_epochs: Number of epochs to use for trainings.
        :param kwargs: Dictionary of framework-specific arguments. These will be passed as such to the `fit` function of
               the target classifier.
        """
        raise NotImplementedError

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform prediction using the adversarially trained classifier.

        :param x: Input samples.
        :param kwargs: Other parameters to be passed on to the `predict` function of the classifier.
        :return: Predictions for test set.
        """
        return self._classifier.predict(x, **kwargs)
