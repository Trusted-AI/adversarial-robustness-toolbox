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
This module implements clean-label attacks on Neural Networks.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from functools import reduce

import numpy as np
from tqdm import tqdm

from art.attacks.attack import PoisoningAttackWhiteBox
from art.estimators import BaseEstimator, NeuralNetworkMixin
from art.estimators.classification import ClassifierMixin, KerasClassifier

logger = logging.getLogger(__name__)


class FeatureCollisionAttack(PoisoningAttackWhiteBox):
    """
    Close implementation of Feature Collision Poisoning Attack by Shafahi, Huang, et al 2018.
    "Poison Frogs! Targeted Clean-Label Poisoning Attacks on Neural Networks"

    This implementation dynamically calculates the dimension of the feature layer, and doesn't hardcode this
    value to 2048 as done in the paper. Thus we recommend using larger values for the similarity_coefficient.

    | Paper link: https://arxiv.org/abs/1804.00792
    """

    attack_params = PoisoningAttackWhiteBox.attack_params + [
        "target",
        "feature_layer",
        "learning_rate",
        "decay_coeff",
        "stopping_tol",
        "num_old_obj",
        "max_iter",
        "similarity_coeff",
        "watermark",
    ]

    _estimator_requirements = (BaseEstimator, NeuralNetworkMixin, ClassifierMixin, KerasClassifier)

    def __init__(self, classifier, target, feature_layer, learning_rate=500 * 255.0, decay_coeff=0.5,
                 stopping_tol=1e-10, num_old_obj=40, max_iter=120, similarity_coeff=256, watermark=None):
        """
        Initialize an Feature Collision Clean-Label poisoning attack

        :param classifier: A trained neural network classifier
        :type classifier: (`art.estimators.NeuralNetworkMixin`, `art.estimators.BaseEstimator`)
        :param target: The target input to misclassify at test time
        :type target: `np.ndarray`
        :param feature_layer: The name of the feature representation layer
        :type feature_layer: `str` or `int`
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
        """
        super().__init__(classifier)

        if not isinstance(classifier, (NeuralNetworkMixin, BaseEstimator)):
            raise TypeError("Classifier must be a neural network")

        kwargs = {
            "classifier": classifier,
            "target": target,
            "feature_layer": feature_layer,
            "learning_rate": learning_rate,
            "decay_coeff": decay_coeff,
            "stopping_tol": stopping_tol,
            "num_old_obj": num_old_obj,
            "max_iter": max_iter,
            "similarity_coeff": similarity_coeff,
            "watermark": watermark
        }

        FeatureCollisionAttack.set_params(self, **kwargs)

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

        target_features = self.estimator.get_activations(self.target, self.feature_layer, 1)
        for init_attack in x:
            old_attack = np.expand_dims(np.copy(init_attack), axis=0)
            poison_features = self.estimator.get_activations(old_attack, self.feature_layer, 1)
            old_objective = self.objective(poison_features, target_features, init_attack, old_attack)
            last_m_objectives = [old_objective]
            # TODO: change to while with convergence (add convergence params)
            for i in tqdm(range(self.max_iter)):
                # forward step
                new_attack = self.forward_step(old_attack)

                # backward step
                new_attack = self.backward_step(np.expand_dims(init_attack, axis=0), poison_features, new_attack)

                rel_change_val = np.linalg.norm(new_attack - old_attack) / np.linalg.norm(new_attack)
                if rel_change_val < self.stopping_tol:
                    logger.info("stopped after " + str(i) + " iterations due to small changes")
                    break

                np.expand_dims(new_attack, axis=0)
                new_feature_rep = self.estimator.get_activations(new_attack, self.feature_layer, 1)
                new_objective = self.objective(new_feature_rep, target_features, init_attack, new_attack)

                avg_of_last_m = sum(last_m_objectives) / float(min(self.num_old_obj, i + 1))

                # Increasing objective means then learning rate is too big.  Chop it, and throw out the latest iteration
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
            watermark = self.watermark * self.target if self.watermark else 0
            final_poison = np.clip(old_attack + watermark, *self.estimator.clip_values)
            final_attacks.append(final_poison)

        return np.vstack(final_attacks), self.estimator.predict(x)

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and apply attack-specific checks before saving them as attributes.

        :param kwargs: a dictionary of attack-specific parameters
        :type kwargs: `dict`
        :return: `True` when parsing was successful
        """
        super().set_params(**kwargs)
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be strictly positive")
        if self.max_iter < 1:
            raise ValueError("Value of max_iter at least 1")
        if not (isinstance(self.feature_layer, str) or isinstance(self.feature_layer, int)):
            raise TypeError("Feature layer should be a string or int")
        if self.decay_coeff <= 0:
            raise ValueError("Decay coefficient must be positive")
        if self.stopping_tol <= 0:
            raise ValueError("Stopping tolerance must be positive")
        if self.num_old_obj <= 0:
            raise ValueError("Number of old stored objectives must be positive")
        if self.max_iter <= 0:
            raise ValueError("Number of old stored objectives must be positive")
        if self.watermark and not (isinstance(self.watermark, float) and 0 <= self.watermark < 1):
            raise ValueError("Watermark must be between 0 and 1")

    def forward_step(self, poison):
        """
        Forward part of forward-backward splitting algorithm

        :param poison: the current poison samples
        :type poison: `np.ndarray`
        :return: poison example closer in feature representation to target space
        :rtype: `np.ndarray`
        """
        target_placeholder, target_feature_rep = self.estimator.get_activations(self.target, self.feature_layer, 1,
                                                                                intermediate=True)
        poison_placeholder, poison_feature_rep = self.estimator.get_activations(poison, self.feature_layer, 1,
                                                                                intermediate=True)
        attack_loss = tensor_norm(poison_feature_rep - target_feature_rep)
        attack_grad, = self.estimator.custom_gradient(attack_loss, [poison_placeholder, target_placeholder],
                                                      [poison, self.target])

        poison -= self.learning_rate * attack_grad[0]

        return poison

    def backward_step(self, base, feature_rep, poison):
        """
        Backward part of forward-backward splitting algorithm

        :param base: the base image that the poison was initialized with
        :type base: `np.ndarray`
        :param poison: the current poison samples
        :type poison: `np.ndarray`
        :param feature_rep: numpy activations at the target layer
        :type feature_rep: `np.ndarray`
        :return: poison example closer in feature representation to target space
        :rtype: `np.ndarray`
        """
        num_features = reduce(lambda x, y: x * y, base.shape)
        dim_features = feature_rep.shape[-1]
        beta = self.similarity_coeff * (dim_features / num_features) ** 2
        poison = (poison + self.learning_rate * beta * base) / (1 + beta * self.learning_rate)
        low, high = self.estimator.clip_values
        return np.clip(poison, low, high)

    def objective(self, poison_feature_rep, target_feature_rep, base_image, poison):
        """
        Objective function of the attack

        :param poison_feature_rep: The numpy activations of the poison image
        :type poison_feature_rep: `np.ndarray`
        :param target_feature_rep: The numpy activations of the target image
        :type target_feature_rep: `np.ndarray`
        :param base_image: The initial image used to poison
        :type base_image: `np.ndarray`
        :param poison: The current poison image
        :type poison: `np.ndarray`
        :return: The objective of the optimization
        :return: `float`
        """
        num_features = prod_sum(base_image.shape)
        num_activations = prod_sum(poison_feature_rep.shape)
        beta = self.similarity_coeff * (num_activations / num_features) ** 2
        return np.linalg.norm(poison_feature_rep - target_feature_rep) + beta * np.linalg.norm(poison - base_image)


def prod_sum(shape):
    """
    Multiples the values of a shape tuple

    :param shape: a shape tuple
    :type: integer tuple
    :return: product of each dimension
    """
    return reduce(lambda dim1, dim2: dim1 * dim2, shape)


def get_class_name(obj):
    """
    Get the full class name of an object
    :param obj: a python object
    :type obj: `object`
    :return: a qualified class name
    :rtype: `str`
    """
    module = obj.__class__.__module__
    if module is None or module == str.__class__.__module__:
        return obj.__class__.__name__
    else:
        return module + '.' + obj.__class__.__name__


def tensor_norm(tensor, norm_type=2):
    """
    Compute the norm of a tensor

    :param tensor: a tensor from a supported ART neural network
    :param norm_type: order fo the norm
    :type norm_type: `int` or `string`
    :return: a tensor with the norm applied
    """
    tf_tensor_types = ('tensorflow.python.framework.ops.Tensor', 'tensorflow.python.framework.ops.EagerTensor')
    torch_tensor_types = ()
    mxnet_tensor_types = ()
    supported_types = tf_tensor_types + torch_tensor_types + mxnet_tensor_types
    tensor_type = get_class_name(tensor)
    if tensor_type not in supported_types:
        raise TypeError("Tensor type `" + tensor_type + "` is not supported")
    elif tensor_type in tf_tensor_types:
        import tensorflow as tf
        return tf.norm(tensor, ord=norm_type)
    elif tensor_type in torch_tensor_types:
        import torch
        return torch.norm(tensor, p=norm_type)
    elif tensor_type in mxnet_tensor_types:
        import mxnet
        return mxnet.ndarray.norm(tensor, ord=norm_type)
