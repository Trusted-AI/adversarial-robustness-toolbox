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

from functools import reduce
import logging
from typing import Optional, Tuple, Union, TYPE_CHECKING, List

import numpy as np
from tqdm.auto import trange

from art.attacks.attack import PoisoningAttackWhiteBox
from art.estimators import BaseEstimator, NeuralNetworkMixin
from art.estimators.classification.classifier import ClassifierMixin
from art.estimators.classification.pytorch import PyTorchClassifier

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_NEURALNETWORK_TYPE

logger = logging.getLogger(__name__)


class BullseyePolytopeAttack(PoisoningAttackWhiteBox):
    """
    Implementation of Bullseye Polytope Attack by Aghakhani, et. al. 2020.
    "Bullseye Polytope: A Scalable Clean-Label Poisoning Attack with Improved Transferability"


    This implementation dynamically calculates the dimension of the feature layer, and doesn't hardcode this
    value to 2048 as done in the paper. Thus we recommend using larger values for the similarity_coefficient.

    | Paper link: https://arxiv.org/abs/2005.00191
    """

    attack_params = PoisoningAttackWhiteBox.attack_params + [
        "target",
        "feature_layer",
        "learning_rate",
        "decay_coeff",
        "stopping_tol",
        "obj_threshold",
        "num_old_obj",
        "max_iter",
        "similarity_coeff",
        "transfer",
        "verbose",
    ]

    _estimator_requirements = (BaseEstimator, NeuralNetworkMixin, ClassifierMixin, PyTorchClassifier)

    def __init__(
            self,
            classifier: Union["CLASSIFIER_NEURALNETWORK_TYPE", List["CLASSIFIER_NEURALNETWORK_TYPE"]],
            target: np.ndarray,
            feature_layer: Union[Union[str, int], List[Union[str, int]]],
            opt: str = 'adam', # SGD, adam or signedadam https://github.com/ucsb-seclab/BullseyePoison/blob/65af294fd4136d15282360d5f65b44ae9390444b/trainer.py#L239
            max_iter: int = 4000,
            learning_rate: float = 4e-2,
            momentum: float = 0.9,
            decay_iter: Union[int, List[int]] = 1000,
            decay_coeff: float = 0.5,
            epsilon: float = 1e-10,
            norm: Union[float, str] = 'inf',
            dropout: int = 0.3,
            transfer: bool = True,
            verbose: bool = True,
    ):
        """
        Initialize an Feature Collision Clean-Label poisoning attack

        :param classifier: The proxy classifiers used for the attack. Can be a single classifier or list of classifiers
                           with varying architectures.
        :param target: The target input(s) of shape (N, W, H, C) to misclassify at test time. Multiple targets will be averaged
        :param feature_layer: The name(s) of the feature representation layer(s).
        :param opt: The optimizer to use for the attack. Can be 'adam', 'sgd', or 'signedadam'
        :param max_iter: The maximum number of iterations for the attack.
        :param learning_rate: The learning rate of clean-label attack optimization.
        :param momentum: The momentum of clean-label attack optimization.
        :param decay_iter: Which iterations to decay the learning rate.
                           Can be a integer (every N interations) or list of integers [0, 500, 1500]
        :param decay_coeff: The decay coefficient of the learning rate.
        :param epsilon: The perturbation budget
        :param norm: The norm of the epsilon-ball
        :param dropout: Dropout to apply while training
        :param transfer: True for transfer learning. False for end-to-end training.
        :param verbose: Show progress bars.
        """
        super().__init__(classifier=classifier)  # type: ignore
        self.target = target
        self.opt = opt
        self.momentum = momentum
        self.decay_iter = decay_iter
        self.epsilon = epsilon
        self.norm = norm
        self.dropout = dropout
        self.transfer = transfer
        self.feature_layer = feature_layer
        self.learning_rate = learning_rate
        self.decay_coeff = decay_coeff
        self.max_iter = max_iter
        self.verbose = verbose
        self._check_params()

        self.target_placeholder, self.target_feature_rep = self.estimator.get_activations(
            self.target, self.feature_layer, 1, framework=True
        )
        self.poison_placeholder, self.poison_feature_rep = self.estimator.get_activations(
            self.target, self.feature_layer, 1, framework=True
        )
        self.attack_loss = tensor_norm(self.poison_feature_rep - self.target_feature_rep)

    def poison(self, x: np.ndarray, y: Optional[np.ndarray] = None, fetch_nearest: bool = False, num_poison: Optional[int] = None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Iteratively finds optimal attack points starting at values at x

        :param x: The base images to begin the poison process.
        :param y: Not used in this attack (clean-label).
        :return: An tuple holding the (poisoning examples, poisoning labels).
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

            for i in trange(self.max_iter, desc="Feature collision", disable=not self.verbose):
                # forward step
                new_attack = self.forward_step(old_attack)

                # backward step
                new_attack = self.backward_step(np.expand_dims(init_attack, axis=0), poison_features, new_attack)

                rel_change_val = np.linalg.norm(new_attack - old_attack) / np.linalg.norm(new_attack)
                if rel_change_val < self.stopping_tol or self.obj_threshold and old_objective <= self.obj_threshold:
                    logger.info("stopped after %d iterations due to small changes", i)
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
                    old_objective = new_objective

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

    def forward_step(self, poison: np.ndarray) -> np.ndarray:
        """
        Forward part of forward-backward splitting algorithm.

        :param poison: the current poison samples.
        :return: poison example closer in feature representation to target space.
        """
        (attack_grad,) = self.estimator.custom_loss_gradient(
            self.attack_loss,
            [self.poison_placeholder, self.target_placeholder],
            [poison, self.target],
            name="feature_collision_" + str(self.feature_layer),
        )
        poison -= self.learning_rate * attack_grad[0]

        return poison

    def backward_step(self, base: np.ndarray, feature_rep: np.ndarray, poison: np.ndarray) -> np.ndarray:
        """
        Backward part of forward-backward splitting algorithm

        :param base: The base image that the poison was initialized with.
        :param feature_rep: Numpy activations at the target layer.
        :param poison: The current poison samples.
        :return: Poison example closer in feature representation to target space.
        """
        num_features = reduce(lambda x, y: x * y, base.shape)
        dim_features = feature_rep.shape[-1]
        beta = self.similarity_coeff * (dim_features / num_features) ** 2
        poison = (poison + self.learning_rate * beta * base) / (1 + beta * self.learning_rate)
        low, high = self.estimator.clip_values
        return np.clip(poison, low, high)

    def objective(
        self, poison_feature_rep: np.ndarray, target_feature_rep: np.ndarray, base_image: np.ndarray, poison: np.ndarray
    ) -> float:
        """
        Objective function of the attack

        :param poison_feature_rep: The numpy activations of the poison image.
        :param target_feature_rep: The numpy activations of the target image.
        :param base_image: The initial image used to poison.
        :param poison: The current poison image.
        :return: The objective of the optimization.
        """
        num_features = base_image.size
        num_activations = poison_feature_rep.size
        beta = self.similarity_coeff * (num_activations / num_features) ** 2
        return np.linalg.norm(poison_feature_rep - target_feature_rep) + beta * np.linalg.norm(poison - base_image)

    def _check_params(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be strictly positive")

        if self.max_iter < 1:
            raise ValueError("Value of max_iter at least 1")

        if not isinstance(self.feature_layer, (str, int)):
            raise TypeError("Feature layer should be a string or int")

        if self.decay_coeff <= 0:
            raise ValueError("Decay coefficient must be positive")

        if self.stopping_tol <= 0:
            raise ValueError("Stopping tolerance must be positive")

        if self.obj_threshold and self.obj_threshold <= 0:
            raise ValueError("Objective threshold must be positive")

        if self.num_old_obj <= 0:
            raise ValueError("Number of old stored objectives must be positive")

        if self.max_iter <= 0:
            raise ValueError("Number of old stored objectives must be positive")

        if self.watermark and not (isinstance(self.watermark, float) and 0 <= self.watermark < 1):
            raise ValueError("Watermark must be between 0 and 1")

        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be of type bool.")


def get_class_name(obj: object) -> str:
    """
    Get the full class name of an object.

    :param obj: A Python object.
    :return: A qualified class name.
    """
    module = obj.__class__.__module__

    if module is None or module == str.__class__.__module__:
        return obj.__class__.__name__

    return module + "." + obj.__class__.__name__


def tensor_norm(tensor, norm_type: Union[int, float, str] = 2):
    """
    Compute the norm of a tensor.

    :param tensor: A tensor from a supported ART neural network.
    :param norm_type: Order of the norm.
    :return: A tensor with the norm applied.
    """
    tf_tensor_types = ("tensorflow.python.framework.ops.Tensor", "tensorflow.python.framework.ops.EagerTensor")
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
