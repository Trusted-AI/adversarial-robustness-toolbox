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
This module implements a Hidden Trigger Backdoor attack on Neural Networks.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from functools import reduce
import logging
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
from scipy.spatial import distance
from tqdm.auto import trange

from art.attacks.attack import PoisoningAttackWhiteBox
from art.attacks.poisoning.backdoor_attack import PoisoningAttackBackdoor
from art.estimators import BaseEstimator, NeuralNetworkMixin
from art.estimators.classification.classifier import ClassifierMixin
from art.estimators.classification.keras import KerasClassifier

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_NEURALNETWORK_TYPE

logger = logging.getLogger(__name__)


class LossMeter(object):
    """Computes and stores the average and current loss value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class HiddenTriggerBackdoorKeras(PoisoningAttackWhiteBox):
    """
    Implementation of Hidden Trigger Backdoor Attack by Saha et al 2019.
    "Hidden Trigger Backdoor Attacks

    | Paper link: https://arxiv.org/abs/1910.00033
    """

    attack_params = PoisoningAttackWhiteBox.attack_params + ["target"]

    _estimator_requirements = (BaseEstimator, NeuralNetworkMixin, ClassifierMixin, KerasClassifier)

    def __init__(
        self,
        classifier: "CLASSIFIER_NEURALNETWORK_TYPE",
        target: Union[int, np.ndarray],
        source: Union[int, np.ndarray],
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
    ) -> None:
        """
        Creates a new Hidden Trigger Backdoor poisoning attack

        :param classifier: A trained neural network classifier.
        :param target: The target class/indices to poison. Triggers added to inputs not in the target class will result in
                       misclassifications to the target class. If an int, it represents a label. Otherwise, it is an
                       array of indicies.
        :param source: The class/indicies which will have a trigger added to cause misclassification
                       If an int, it represents a label. Otherwise, it is an array of indicies.
        :param feature_layer: The name of the feature representation layer.
        :param eps: Maximum perturbation that the attacker can introduce.
        :param learning_rate: The learning rate of clean-label attack optimization.
        :param decay_coeff: The decay coefficient of the learning rate.
        :param decay_iter: The number of iterations before the learning rate decays
        :param stopping_threshold: Stop iterations after loss is less than this threshold.
        :param max_iter: The maximum number of iterations for the attack.
        :param batch_size: The number of samples to draw per batch.
        :param poison_percent: The percentage of the data to poison. This is ignored if indices are provided
        :param is_index: If true, the source and target params are assumed to represent indices rather than a class label.
                         poison_percent is ignored if true
        :param verbose: Show progress bars.
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

    def poison(  # pylint: disable=W0221
        self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calls perturbation function on input x and returns the perturbed input and poison labels for the data.

        :param x: An array with the points to draw source and target samples from. Source indicates the class(es) that the
        backdoor would be added to to cause misclassification into the target label. Target indicates the class that the
        backdoor should cause misclassification into.
        :param y: The labels of the provided samples. If none, we will use the classifier to label the data.
        :return: An tuple holding the `(poisoning_examples, poisoning_labels)`.
        """
        data = np.copy(x)
        estimated_labels = self.classifier.predict(data) if y is None else np.copy(y)

        # Get indices of target class
        if not self.is_index:
            poison_class = self.target
            poison_indices = np.where(np.all(estimated_labels == poison_class, axis=1))[0]
            num_poison = int(np.ceil(self.poison_percent * len(poison_indices)))
            if num_poison == 0:
                raise ValueError("No data points with target label found")

            poison_indices = np.random.choice(poison_indices, num_poison, replace=False)
        else:
            poison_class = estimated_labels[self.target[0]]
            poison_indices = self.target
            if not np.all(np.all(estimated_labels[poison_indices] == poison_class, axis=1)):
                raise ValueError("The target indices do not share the same label")

            num_poison = len(poison_indices)

        # Get indices of source class
        if not self.is_index:
            trigger_class = self.source
            trigger_indices = np.where(np.all(estimated_labels == trigger_class, axis=1))[0]
            num_trigger = min(len(trigger_indices), num_poison)
            if num_trigger == 0:
                raise ValueError("No data points with source label found")
            elif num_trigger < num_poison:
                raise ValueError(
                    "There must be at least as many images with the source label as the target. Maybe try reducing poison_percent or providing fewer target indices"
                )

            # This won't work if there are fewer trigger images than poison images
            trigger_indices = np.random.choice(trigger_indices, num_poison, replace=False)
            num_trigger = len(trigger_indices)

        # Otherwise, we treat it as an index
        else:
            print(len(self.source))
            trigger_class = estimated_labels[self.target[0]]
            trigger_indices = self.source
            if not np.all(np.all(estimated_labels[trigger_indices] == trigger_class, axis=1)):
                raise ValueError("The target indices do not share the same label")

            num_trigger = len(trigger_indices)
            if num_trigger < num_poison:
                raise ValueError(
                    "There must be at least as many images with the source label as the target. Maybe try reducing poison_percent or providing fewer target indices"
                )

        logger.info("Number of poison inputs:{}".format(num_poison_img))
        logger.info("Number of trigger inputs:{}".format(num_trigger_img))

        batches = int(np.ceil(num_poison_img / float(self.batch_size)))

        # Extra stuff for debugging
        losses = LossMeter()
        final_poison = np.copy(data[poison_indices])

        original_images = np.copy(data[poison_indices])

        # FIX THINGS BELOW THIS LINE
        for batch_id in trange(batches, desc="Hidden Trigger", disable=not self.verbose):

            # TODO Do something here so it loops if one of the batches ends
            batch_index_1, batch_index_2 = batch_id * batch_size, (batch_id + 1) * batch_size
            target_samples = data[target_indices[batch_index_1:batch_index_2]]
            source_samples = data[source_indices[batch_index_1:batch_index_2]]

            poison = np.zeros_like(target_samples)

            # First, we add the backdoor to the source samples and get the feature representation
            source_samples, _ = self.backdoor.poison(source_samples, self.target, broadcast=True)

            # If attack_loss is none, then we need to define it and also initialize the placeholders
            if attack_loss is None:
                target_placeholder, target_features = self.estimator.get_activations(
                    target_samples, self.feature_layer, 1, framework=True
                )
                source_placeholder, source_features = self.estimator.get_activations(
                    source_samples, self.feature_layer, 1, framework=True
                )

                attack_loss = tensor_norm(target_features - source_features)

            source_features = self.estimator.get_activations(source_samples, self.feature_layer, 1)

            for i in range(self.max_iter):
                lr = self.learning_rate * (self.decay_coeff ** (i // self.decay_iter))

                target_features = self.estimator.get_activations(target_samples + poison, self.feature_layer, 1)

                # Compute distance between features and match samples
                # We are swapping the samples and the features unlike in the original implementation because
                # we are computing the loss gradient using ART, which needs the inputs rather than the features
                source_samples_copy = np.copy(source_samples)
                source_features_copy = np.copy(source_features)  # Assuming this is numpy array
                dist = distance.cdist(source_features, target_features)
                for _ in range(len(source_features)):
                    min_index = np.squeeze((dist == np.min(dist)).nonzero())
                    source_samples[min_index[1]] = source_samples_copy[min_index[0]]
                    source_features[min_index[1]] = source_feature_copy[min_index[0]]
                    dist[min_index[0], min_index[1]] = 1e5

                loss = np.linalg.norm(source_features - target_features)
                (attack_grad,) = self.estimator.custom_loss_gradient(
                    attack_loss,
                    [source_placeholder, target_placeholder],
                    [target_samples + poison, source_samples],
                    name="hidden_trigger" + str(self.feature_layer),
                )

                # Update the poison and clip
                poison -= lr * attack_grad[0]
                poison = np.clip(poison, -1 * self.eps, self.eps)
                poisoned_samples = np.clip(target_samples + poison, *self.estimator.clip_values)
                poison = poisoned_samples - target_samples

            data[target_indices[batch_index_1:batch_index_2]] = poisoned_samples

        return data, estimated_labels


def tensor_norm(tensor, norm_type: Union[int, float, str] = 2):  # pylint: disable=R1710
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
    if tensor_type not in supported_types:  # pragma: no cover
        raise TypeError("Tensor type `" + tensor_type + "` is not supported")

    if tensor_type in tf_tensor_types:
        import tensorflow as tf

        return tf.norm(tensor, ord=norm_type)

    if tensor_type in torch_tensor_types:  # pragma: no cover
        import torch

        return torch.norm(tensor, p=norm_type)

    if tensor_type in mxnet_tensor_types:  # pragma: no cover
        import mxnet

        return mxnet.ndarray.norm(tensor, ord=norm_type)
