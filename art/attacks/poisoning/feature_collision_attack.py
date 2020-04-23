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
This module implements clean-label attacks on Neural Networks.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np

from art.attacks.attack import PoisoningAttackWhiteBox
from art.classifiers import ClassifierNeuralNetwork, Classifier
# from art.utils import tensor_norm

from functools import reduce

logger = logging.getLogger(__name__)


class FeatureCollisionAttack(PoisoningAttackWhiteBox):
    """
    Close implementation of Feature Collision Poisoning Attack by Shafahi, Huang, et al 2018.
    "Poison Frogs! Targeted Clean-Label Poisoning Attacks on Neural Networks"

    | Paper link: https://arxiv.org/pdf/1804.00792.pdf
    """

    attack_params = PoisoningAttackWhiteBox.attack_params + [
        "classifier",
        "target",
        "feature_layer",
        "learning_rate",
        "max_iter",
        "similarity_coeff"
    ]

    def __init__(self, classifier, target, eps, feature_layer, learning_rate=0.01, max_iter=50, similatiry_coeff=0.25,
                 **kwargs):
        """
        Initialize an SVM poisoning attack

        :param classifier: A trained neural network classifier
        :type classifier: (`art.classifiers.ClassifierNeuralNetwork`, `art.classifiers.Classifier`)
        :param target: The target input to misclassify at test time
        :type target: `np.ndarray`
        :param eps: The strength of the attack
        :type eps: `float`
        :param feature_layer: The name of the feature representation layer
        :type feature_layer: `str`
        :param learning_rate: The learning rate of clean-label attack optimization
        :type learning_rate: `float`
        :param max_iter: The maximum number of iterations for the attack
        :type max_iter: `int`
        :param similarity_coeff: The maximum number of iterations for the attack
        :type similarity_coeff: `float`
        :param kwargs: Extra optional keyword arguments
        """
        super().__init__(classifier)

        if not isinstance(classifier, (ClassifierNeuralNetwork, Classifier)):
            raise TypeError("Classifier must be a neural network")

        self.classifier = classifier
        self.target = target
        self.feature_layer = feature_layer
        self.learning_rate = learning_rate
        self.eps = eps
        self.max_iter = max_iter
        self.similarity_coeff = similatiry_coeff

        self.set_params(**kwargs)

    def poison(self, x, y=None, **kwargs):
        """
        Iteratively finds optimal attack points starting at values at x

        :param x: The base images to begin the poison process
        :type x: `np.ndarray`
        :param y: Not used  in this attack (clean-label)
        :return: An tuple holding the (poisoning examples, poisoning labels).
        :rtype: `(np.ndarray, np.ndarray)`
        """

        num_poison = len(x)

        if num_poison == 0:
            raise ValueError("Must input at least one poison point")

        # TODO: ensure class of x does not match class of target

        old_attack = x

        # TODO: change to while with convergence (add convergence params)
        for _ in range(self.max_iter):
            new_attack = self.forward_step(old_attack)  # TODO: pass feature reps in here
            new_attack = self.backward_step(x, new_attack)

            old_attack = new_attack

        return old_attack

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and apply attack-specific checks before saving them as attributes.

        :param kwargs: a dictionary of attack-specific parameters
        :type kwargs: `dict`
        :return: `True` when parsing was successful
        """
        super(FeatureCollisionAttack, self).set_params(**kwargs)
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be strictly positive")
        if self.eps <= 0:
            raise ValueError("Value of eps must be strictly positive")
        if self.max_iter <= 1:
            raise ValueError("Value of max_iter at least 1")
        if not isinstance(self.classifier, ClassifierNeuralNetwork):
            raise TypeError("Classifier must be a neural network")
        if not isinstance(self.classifier, Classifier):
            raise TypeError("Classifier must be a valid ART classifier")

    def forward_step(self, poison):
        """
        Forward part of forward-backward splitting algorithm
        :param poison: the current poison samples
        :type poison: `np.ndarray`
        :return: poison example closer in feature representation to target space
        :rtype: `np.ndarray`
        """
        target_feature_rep = self.classifier.get_activations(self.target, self.feature_layer, 1, intermediate=True)
        poison_feature_rep = self.classifier.get_activations(poison, self.feature_layer, 1, intermediate=True)

        attack_loss = self.classifier.normalize_tensor(poison_feature_rep - target_feature_rep)

        attack_grad, = self.classifier.custom_gradient(attack_loss, self.classifier.get_input_layer(), poison)

        poison -= self.learning_rate * attack_grad

        return poison

    def backward_step(self, base, poison):
        """
        Backward part of forward-backward splitting algorithm
        :param base: the base image that the poison was initialized with
        :type base: `np.ndarray`
        :param poison: the current poison samples
        :type poison: `np.ndarray`
        :return: poison example closer in feature representation to target space
        :rtype: `np.ndarray`
        """
        num_features = reduce(lambda x, y: x * y, poison.shape)
        # TODO: replace 2048 with dynamic shape of feature representation
        beta = self.similarity_coeff * (2048.0 / num_features) ** 2
        poison = (poison + self.learning_rate * beta * base) / (1 + beta * self.learning_rate)
        return np.clip(poison, self.classifier.clip_values[0], self.classifier.clip_values[1])
