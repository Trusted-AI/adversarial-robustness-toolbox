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
This module implements adversarial training with Oracle Aligned Adversarial Training (OAAT) protocol
for adversarial training for defence against larger perturbations.

| Paper link: https://link.springer.com/chapter/10.1007/978-3-031-20065-6_18

| It was noted that this protocol uses double perturbation mechanism i.e, perturbation on the input samples and then
perturbation on the model parameters. Consequently, framework specific implementations are being provided in ART.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import abc
from typing import Optional, Tuple, TYPE_CHECKING, Sequence

import numpy as np

from art.defences.trainer.trainer import Trainer
from art.attacks.attack import EvasionAttack
from art.data_generators import DataGenerator

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_LOSS_GRADIENTS_TYPE


class AdversarialTrainerOAAT(Trainer):
    """
    This is abstract class for different backend-specific implementations of OAAT protocol.

    | Paper link: https://link.springer.com/chapter/10.1007/978-3-031-20065-6_18
    """

    def __init__(
        self,
        classifier: "CLASSIFIER_LOSS_GRADIENTS_TYPE",
        proxy_classifier: "CLASSIFIER_LOSS_GRADIENTS_TYPE",
        lpips_classifier: "CLASSIFIER_LOSS_GRADIENTS_TYPE",
        list_avg_models: Sequence["CLASSIFIER_LOSS_GRADIENTS_TYPE"],
        attack: EvasionAttack,
        train_params: dict,
    ):
        """
        Create an :class:`.AdversarialTrainerOAAT` instance.

        :param classifier: Model to train adversarially.
        :param proxy_classifier: Model for adversarial weight perturbation.
        :param lpips_classifier: Weight averaging model for calculating activations.
        :param list_avg_models: list of models for weight averaging.
        :param attack: attack to use for data augmentation in adversarial training
        :param train_params: parameters' dictionary related to adversarial training
        """
        self._attack = attack
        self._proxy_classifier = proxy_classifier
        self._lpips_classifier = lpips_classifier
        self._list_avg_models = list_avg_models
        self._train_params = train_params
        self._apply_wp = False
        self._apply_lpips_pert = False
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
        Train a model adversarially with OAAT. See class documentation for more information on the exact procedure.

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
    def fit_generator(  # pylint: disable=W0221
        self,
        generator: DataGenerator,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        nb_epochs: int = 20,
        **kwargs
    ):
        """
        Train a model adversarially with OAAT using a data generator.
        See class documentation for more information on the exact procedure.

        :param generator: Data generator.
        :param validation_data: Tuple consisting of validation data, (x_val, y_val)
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
