# MIT License
#
# Copyright (C) IBM Corporation 2020
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

from art.defences.trainer.trainer import Trainer
from art.defences.trainer.adversarial_trainer import AdversarialTrainer
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent

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
        self, classifier, nb_epochs=391, batch_size=128, eps=8, eps_step=2, max_iter=7, num_random_init=True, **kwargs
    ):
        """
        Create an :class:`.AdversarialTrainerMadryPGD` instance.

        Default values are for CIFAR-10 in pixel range 0-255.

        :param classifier: Classifier to train adversarially.
        :type classifier: :class:`.Classifier`
        :param nb_epochs: Number of training epochs.
        :type nb_epochs: `int`
        :param batch_size: Size of the batch on which adversarial samples are generated.
        :type batch_size: `int`
        :param eps: Maximum perturbation that the attacker can introduce.
        :type eps: `float`
        :param eps_step: Attack step size (input variation) at each iteration.
        :type eps_step: `float`
        :param max_iter: The maximum number of iterations.
        :type max_iter: `int`
        :param num_random_init: Number of random initialisations within the epsilon ball. For num_random_init=0
            starting at the original input.
        :type num_random_init: `int`
        """

        self.batch_size = batch_size
        self.nb_epochs = nb_epochs

        # Setting up adversary and perform adversarial training:
        self.attack = ProjectedGradientDescent(
            classifier, eps=eps, eps_step=eps_step, max_iter=max_iter, num_random_init=num_random_init
        )

        self.trainer = AdversarialTrainer(classifier, self.attack, ratio=1.0)

    def fit(self, x, y, validation_data=None, **kwargs):
        """
        Train a model adversarially. See class documentation for more information on the exact procedure.

        :param x: Training data.
        :type x: `np.ndarray`
        :param y: Labels for the training data.
        :type y: `np.ndarray`
        :param kwargs: Dictionary of framework-specific arguments.
        :type kwargs: `dict`
        :return: `None`
        """
        self.trainer.fit(
            x, y, validation_data=validation_data, nb_epochs=self.nb_epochs, batch_size=self.batch_size, **kwargs
        )

    def get_classifier(self):
        return self.trainer.get_classifier()
