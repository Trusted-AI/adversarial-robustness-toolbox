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
from tqdm import tqdm

from art.attacks.attack import PoisoningAttackWhiteBox
from art.estimators import BaseEstimator, NeuralNetworkMixin
from art.utils import tensor_norm

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
        "decay_coeff",
        "stopping_tol",
        "num_old_obj",
        "max_iter",
        "similarity_coeff",
        "watermarking",
    ]

    _estimator_requirements = (BaseEstimator, NeuralNetworkMixin)

    def __init__(self, classifier, target, feature_layer, learning_rate=500 * 255.0, decay_coeff=0.5,
                 stopping_tol=1e-10, num_old_obj=40, max_iter=120, similarity_coeff=0.25, watermark=0.35,
                 **kwargs):
        """
        Initialize an SVM poisoning attack

        :param classifier: A trained neural network classifier
        :type classifier: (`art.estimators.NeuralNetworkMixin`, `art.estimators.BaseEstimator`)
        :param target: The target input to misclassify at test time
        :type target: `np.ndarray`
        :param feature_layer: The name of the feature representation layer
        :type feature_layer: `str`
        :param learning_rate: The learning rate of clean-label attack optimization
        :type learning_rate: `float`
        :param decay_coeff: The decay coefficient of the learning rate
        :type decay_coeff: `float`
        :param stopping_tol: The tolerance for relative change in objective function
        :type stopping_tol: `float`
        :param num_old_obj: The number of old objective values to store
        :type num_old_obj: `int`
        :param max_iter: The maximum number of iterations for the attack
        :type max_iter: `int`
        :param similarity_coeff: The maximum number of iterations for the attack
        :type similarity_coeff: `float`
        :param watermark: Whether The opacity of the watermarked target image
        :type watermark: `float`
        :param kwargs: Extra optional keyword arguments
        """
        super().__init__(classifier)

        if not isinstance(classifier, (NeuralNetworkMixin, BaseEstimator)):
            raise TypeError("Classifier must be a neural network")

        self.classifier = classifier
        self.target = target
        self.feature_layer = feature_layer
        self.learning_rate = learning_rate
        self.decay_coeff = decay_coeff
        self.stopping_tol = stopping_tol
        self.num_old_obj = num_old_obj
        self.max_iter = max_iter
        self.similarity_coeff = similarity_coeff
        self.watermark = watermark

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
        final_attacks = []
        if num_poison == 0:
            raise ValueError("Must input at least one poison point")

        # TODO: ensure class of x does not match class of target
        target_features = self.classifier.get_activations(self.target, self.feature_layer, 1)
        for init_attack in x:
            old_attack = np.expand_dims(np.copy(init_attack), axis=0)
            poison_features = self.classifier.get_activations(old_attack, self.feature_layer, 1)
            old_objective = self.objective(poison_features, target_features, init_attack, old_attack)
            last_m_objectives = [old_objective]
            # TODO: change to while with convergence (add convergence params)
            for i in tqdm(range(self.max_iter)):
                # forward step
                new_attack = self.forward_step(old_attack)

                # backward step
                new_attack = self.backward_step(np.expand_dims(init_attack, axis=0), new_attack)

                rel_change_val = np.linalg.norm(new_attack - old_attack) / np.linalg.norm(new_attack)
                if rel_change_val < self.stopping_tol:
                    print("stopped after " + str(i) + " iterations due to small changes")
                    break

                np.expand_dims(new_attack, axis=0)
                new_feature_rep = self.classifier.get_activations(new_attack, self.feature_layer, 1)
                new_objective = self.objective(new_feature_rep, target_features, init_attack, new_attack)
                # np.linalg.norm(new_feature_rep - target_features) + beta * np.linalg.norm(new_attack - x)

                avg_of_last_m = sum(last_m_objectives) / float(min(self.num_old_obj, i + 1))

                # If the objective went up, then learning rate is too big.  Chop it, and throw out the latest iteration
                if new_objective >= avg_of_last_m and (i % self.num_old_obj / 2 == 0):
                    self.learning_rate *= self.decay_coeff
                else:
                    old_attack = new_attack

                if i < self.num_old_obj - 1:
                    last_m_objectives.append(new_objective)
                else:
                    # first remove the oldest obj then append the new obj
                    del last_m_objectives[0]
                    last_m_objectives.append(new_objective)

            # Watermarking
            final_poison = np.clip(old_attack[0] + 0.35 * self.target, *self.classifier.clip_values)
            final_attacks.append(final_poison)

        return np.array(final_attacks)

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
        if self.max_iter <= 1:
            raise ValueError("Value of max_iter at least 1")
        if not isinstance(self.classifier, NeuralNetworkMixin):
            raise TypeError("Classifier must be a neural network")
        if not isinstance(self.classifier, BaseEstimator):
            raise TypeError("Classifier must be a valid ART estimator")

    def forward_step(self, poison):
        """
        Forward part of forward-backward splitting algorithm
        :param poison: the current poison samples
        :type poison: `np.ndarray`
        :return: poison example closer in feature representation to target space
        :rtype: `np.ndarray`
        """
        # target_feature_rep = self.classifier.get_activations(self.target, self.feature_layer, 1, intermediate=True)
        # poison_feature_rep = self.classifier.get_activations(poison, self.feature_layer, 1, intermediate=True)
        #
        # attack_loss = self.classifier.normalize_tensor(poison_feature_rep - target_feature_rep)
        target_placeholder, target_feature_rep = self.classifier.get_activations(self.target, self.feature_layer, 1,
                                                                                 intermediate=True)
        poison_placeholder, poison_feature_rep = self.classifier.get_activations(poison, self.feature_layer, 1,
                                                                                 intermediate=True)
        attack_loss = self.classifier.normalize_tensor(poison_feature_rep - target_feature_rep)
        # attack_loss = tensor_norm([poison_feature_rep - target_feature_rep])
        attack_grad, = self.classifier.custom_gradient(attack_loss, [poison_placeholder, target_placeholder],
                                                       [poison, self.target])

        poison -= self.learning_rate * attack_grad[0]

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
        num_features = reduce(lambda x, y: x * y, base.shape)
        # TODO: replace 2048 with dynamic shape of feature representation
        beta = self.similarity_coeff * (2048.0 / num_features) ** 2
        poison = (poison + self.learning_rate * beta * base) / (1 + beta * self.learning_rate)
        low, high = self.classifier.clip_values
        return np.clip(poison, low, high)

    def objective(self, poison_feature_rep, target_feature_rep, base_image, poison):
        """
        Objective function of the attack

        :param poison_feature_rep: The output of
        :param target_feature_rep:
        :param base_image:
        :param poison:
        :return: `float`
        """
        prod_sum = lambda shape_tuple: reduce(lambda dim1, dim2: dim1 * dim2, shape_tuple)
        num_features = prod_sum(base_image.shape)
        num_activations = prod_sum(poison_feature_rep.shape)
        beta = self.similarity_coeff * (num_activations / num_features) ** 2
        return np.linalg.norm(poison_feature_rep - target_feature_rep) + beta * np.linalg.norm(poison - base_image)
