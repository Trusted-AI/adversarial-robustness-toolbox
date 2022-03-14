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
This module implements adversarial training following Madry's Protocol.

| Paper link: https://arxiv.org/abs/1706.06083

| Please keep in mind the limitations of defences. While adversarial training is widely regarded as a promising,
    principled approach to making classifiers more robust (see https://arxiv.org/abs/1802.00420), very careful
    evaluations are required to assess its effectiveness case by case (see https://arxiv.org/abs/1902.06705).
"""
import logging
from typing import Optional, Union, TYPE_CHECKING

import numpy as np

from art.defences.trainer.trainer import Trainer
from art.defences.trainer.adversarial_trainer import AdversarialTrainer
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_LOSS_GRADIENTS_TYPE


logger = logging.getLogger(__name__)


class AdversarialTrainerMadryPGD(Trainer):
    """
    Class performing adversarial training following Madry's Protocol.

    | Paper link: https://arxiv.org/abs/1706.06083

    | Please keep in mind the limitations of defences. While adversarial training is widely regarded as a promising,
        principled approach to making classifiers more robust (see https://arxiv.org/abs/1802.00420), very careful
        evaluations are required to assess its effectiveness case by case (see https://arxiv.org/abs/1902.06705).
    """

    def __init__(
        self,
        classifier: "CLASSIFIER_LOSS_GRADIENTS_TYPE",
        nb_epochs: int = 391,
        batch_size: int = 128,
        eps: Union[int, float] = 8,
        eps_step: Union[int, float] = 2,
        max_iter: int = 7,
        num_random_init: int = 1,
    ) -> None:
        """
        Create an :class:`.AdversarialTrainerMadryPGD` instance.

        Default values are for CIFAR-10 in pixel range 0-255.

        :param classifier: Classifier to train adversarially.
        :param nb_epochs: Number of training epochs.
        :param batch_size: Size of the batch on which adversarial samples are generated.
        :param eps: Maximum perturbation that the attacker can introduce.
        :param eps_step: Attack step size (input variation) at each iteration.
        :param max_iter: The maximum number of iterations.
        :param num_random_init: Number of random initialisations within the epsilon ball. For num_random_init=0
                                starting at the original input.
        """
        super().__init__(classifier=classifier)  # type: ignore
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs

        # Setting up adversary and perform adversarial training:
        self.attack = ProjectedGradientDescent(
            classifier,
            eps=eps,
            eps_step=eps_step,
            max_iter=max_iter,
            num_random_init=num_random_init,
        )

        self.trainer = AdversarialTrainer(classifier, self.attack, ratio=1.0)  # type: ignore

    def fit(  # pylint: disable=W0221
        self, x: np.ndarray, y: np.ndarray, validation_data: Optional[np.ndarray] = None, **kwargs
    ) -> None:
        """
        Train a model adversarially. See class documentation for more information on the exact procedure.

        :param x: Training data.
        :param y: Labels for the training data.
        :param validation_data: Validation data.
        :param kwargs: Dictionary of framework-specific arguments.
        """
        self.trainer.fit(
            x, y, validation_data=validation_data, nb_epochs=self.nb_epochs, batch_size=self.batch_size, **kwargs
        )

    def get_classifier(self) -> "CLASSIFIER_LOSS_GRADIENTS_TYPE":
        return self.trainer.get_classifier()
