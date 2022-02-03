# MIT License
#
# Copyright (c) 2020 Aniruddha Saha, Akshayvarun Subramanya, Hamed Pirsiavash
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

import logging
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
from tqdm.auto import trange

from art.attacks.attack import PoisoningAttackWhiteBox
from art.attacks.poisoning.backdoor_attack import PoisoningAttackBackdoor
from art.estimators import BaseEstimator, NeuralNetworkMixin
from art.estimators.classification.classifier import ClassifierMixin

if TYPE_CHECKING:
    # pylint: disable=C0412
    from art.estimators.classification.pytorch import PyTorchClassifier

logger = logging.getLogger(__name__)


class LossMeter:
    """
    Computes and stores the average and current loss value
    """

    def __init__(self):
        """
        Create loss tracker
        """
        self.reset()

    def reset(self):
        """
        Reset loss tracker
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Update loss tracker
        :param val: Loss value to add to tracker
        :param n: Number of elements contributing to val
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class HiddenTriggerBackdoorPyTorch(PoisoningAttackWhiteBox):
    """
    Implementation of Hidden Trigger Backdoor Attack by Saha et al 2019.
    "Hidden Trigger Backdoor Attacks

    | Paper link: https://arxiv.org/abs/1910.00033
    """

    attack_params = PoisoningAttackWhiteBox.attack_params + ["target"]

    _estimator_requirements = (BaseEstimator, NeuralNetworkMixin, ClassifierMixin)

    def __init__(
        self,
        classifier: "PyTorchClassifier",
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
        Creates a new Hidden Trigger Backdoor poisoning attack for PyTorch.

        :param classifier: A trained neural network classifier.
        :param target: The target class/indices to poison. Triggers added to inputs not in the target class will
                       result in misclassifications to the target class. If an int, it represents a label.
                       Otherwise, it is an array of indicies.
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
        :param is_index: If true, the source and target params are assumed to represent indices rather
                         than a class label. poison_percent is ignored if true
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
        Calls perturbation function on the dataset x and returns only the perturbed input and their
        indices in the dataset.
        :param x: An array in the shape NxCxWxH with the points to draw source and target samples from.
                  Source indicates the class(es) that the backdoor would be added to to cause
                  misclassification into the target label.
                  Target indicates the class that the backdoor should cause misclassification into.
        :param y: The labels of the provided samples. If none, we will use the classifier to label the
                  data.
        :return: An tuple holding the `(poison samples, indices in x that the poison samples should replace)`.
        """
        import torch  # lgtm [py/repeated-import]

        data = np.copy(x)
        estimated_labels = self.estimator.predict(data) if y is None else np.copy(y)

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
            if num_trigger < num_poison:
                raise ValueError("There must be at least as many images with the source label as the target.")

            # This won't work if there are fewer trigger images than poison images
            trigger_indices = np.random.choice(trigger_indices, num_poison, replace=False)
            num_trigger = len(trigger_indices)

        # Otherwise, we treat it as an index
        else:
            trigger_class = estimated_labels[self.target[0]]
            trigger_indices = self.source
            num_trigger = len(trigger_indices)
            if num_trigger < num_poison:
                raise ValueError("There must be at least as many images with the source label as the target.")

        logger.info("Number of poison inputs: %d", num_poison)
        logger.info("Number of trigger inputs: %d", num_trigger)

        batches = int(np.ceil(num_poison / float(self.batch_size)))

        losses = LossMeter()
        final_poison = np.copy(data[poison_indices])

        original_images = torch.from_numpy(np.copy(data[poison_indices])).to(self.estimator.device)

        for batch_id in trange(batches, desc="Hidden Trigger", disable=not self.verbose):

            cur_index = self.batch_size * batch_id
            offset = min(self.batch_size, num_poison - cur_index)
            poison_batch_indices = poison_indices[cur_index : cur_index + offset]
            trigger_batch_indices = trigger_indices[cur_index : cur_index + offset]

            poison_samples = torch.from_numpy(data[poison_batch_indices]).to(self.estimator.device)

            # First, we add the backdoor to the source samples and get the feature representation
            trigger_samples, _ = self.backdoor.poison(data[trigger_batch_indices], self.target, broadcast=True)
            trigger_samples = torch.from_numpy(trigger_samples).to(self.estimator.device)

            feat1 = self.estimator.get_activations(trigger_samples, self.feature_layer, 1, framework=True)
            feat1 = feat1.detach().clone()

            for i in range(self.max_iter):
                poison_samples.requires_grad_()
                if isinstance(self.decay_iter, int):
                    decay_exp = i // self.decay_iter
                else:
                    max_index = [ii for ii, _ in enumerate(self.decay_iter) if self.decay_iter[ii] <= i]
                    if len(max_index) == 0:
                        decay_exp = 0
                    else:
                        decay_exp = max(max_index) + 1
                learning_rate = self.learning_rate * (self.decay_coeff ** decay_exp)

                # Compute the feature representation of the current poisons and
                # identify the closest trigger sample for each poison
                feat2 = self.estimator.get_activations(poison_samples, self.feature_layer, 1, framework=True)
                feat11 = feat1.clone()
                dist = torch.cdist(feat1, feat2)

                for _ in range(feat2.size(0)):
                    dist_min_index = (dist == torch.min(dist)).nonzero().squeeze()
                    feat1[dist_min_index[1]] = feat11[dist_min_index[0]]
                    dist[dist_min_index[0], dist_min_index[1]] = 1e5

                loss = torch.norm(feat1 - feat2) ** 2
                losses.update(loss.item(), trigger_samples.size(0))
                loss.backward()

                # Update the poison and clip
                poison_samples = poison_samples - learning_rate * poison_samples.grad
                pert = poison_samples - original_images[cur_index : cur_index + offset]
                pert = torch.clamp(pert, -self.eps, self.eps).detach_()
                poison_samples = pert + original_images[cur_index : cur_index + offset]
                poison_samples = poison_samples.clamp(*self.estimator.clip_values)

                if i % 100 == 0:
                    print(
                        "Epoch: {:2d} | batch: {} | i: {:5d} | LR: {:2.5f} | \
                        Loss Val: {:5.3f} | Loss Avg: {:5.3f}".format(
                            0, batch_id, i, learning_rate, losses.val, losses.avg
                        )
                    )

                if loss.item() < self.stopping_threshold or i == (self.max_iter - 1):
                    print("Max_Loss: {}".format(loss.item()))
                    final_poison[cur_index : cur_index + offset] = poison_samples.detach().cpu().numpy()
                    break

        return final_poison, poison_indices

    def _check_params(self) -> None:

        if self.is_index and not (isinstance(self.target, np.ndarray) and isinstance(self.source, np.ndarray)):
            raise ValueError("Target and source values must be an array of indices")

        if (isinstance(self.target, int) and (self.target == self.source)) or (
            isinstance(self.target, np.ndarray) and np.array_equal(self.target, self.source)
        ):
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
                raise ValueError("feature_layer is not a non-negative integer and can't be greater than the number of layers")

        if self.decay_coeff <= 0:
            raise ValueError("Decay coefficient must be positive")

        if not 0 < self.poison_percent <= 1:
            raise ValueError("poison_percent must be between 0 (exclusive) and 1 (inclusive)")

        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be of type bool.")