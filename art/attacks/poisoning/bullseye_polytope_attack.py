# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2021
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
This module implements Bullseye Polytope clean-label attacks on Neural Networks.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, Tuple, Union, TYPE_CHECKING, List

import numpy as np
import time
from tqdm.auto import trange

from art.attacks.attack import PoisoningAttackWhiteBox
from art.estimators import BaseEstimator, NeuralNetworkMixin
from art.estimators.classification.classifier import ClassifierMixin
from art.estimators.classification.pytorch import PyTorchClassifier

if TYPE_CHECKING:
    import torch
    from art.utils import CLASSIFIER_NEURALNETWORK_TYPE

logger = logging.getLogger(__name__)


class BullseyePolytopeAttackPyTorch(PoisoningAttackWhiteBox):
    """
    Implementation of Bullseye Polytope Attack by Aghakhani, et. al. 2020.
    "Bullseye Polytope: A Scalable Clean-Label Poisoning Attack with Improved Transferability"

    This implementation is based on UCSB's original code here: https://github.com/ucsb-seclab/BullseyePoison

    | Paper link: https://arxiv.org/abs/2005.00191
    """

    attack_params = PoisoningAttackWhiteBox.attack_params + [
        "target",
        "feature_layer",
        "opt" "max_iter",
        "learning_rate",
        "momentum",
        "decay_iter",
        "decay_coeff",
        "epsilon",
        "norm",
        "dropout",
        "endtoend",
        "verbose",
    ]

    _estimator_requirements = (BaseEstimator, NeuralNetworkMixin, ClassifierMixin, PyTorchClassifier)

    def __init__(
        self,
        classifier: Union["CLASSIFIER_NEURALNETWORK_TYPE", List["CLASSIFIER_NEURALNETWORK_TYPE"]],
        target: np.ndarray,
        feature_layer: Union[Union[str, int], List[Union[str, int]]],
        opt: str = "adam",
        max_iter: int = 4000,
        learning_rate: float = 4e-2,
        momentum: float = 0.9,
        decay_iter: Union[int, List[int]] = 10000,
        decay_coeff: float = 0.5,
        epsilon: float = 0.1,
        dropout: int = 0.3,
        net_repeat: int = 1,
        endtoend: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize an Feature Collision Clean-Label poisoning attack

        :param classifier: The proxy classifiers used for the attack. Can be a single classifier or list of classifiers
                           with varying architectures.
        :param target: The target input(s) of shape (N, W, H, C) to misclassify at test time. Multiple targets will be
                       averaged.
        :param feature_layer: The name(s) of the feature representation layer(s).
        :param opt: The optimizer to use for the attack. Can be 'adam' or 'sgd'
        :param max_iter: The maximum number of iterations for the attack.
        :param learning_rate: The learning rate of clean-label attack optimization.
        :param momentum: The momentum of clean-label attack optimization.
        :param decay_iter: Which iterations to decay the learning rate.
                           Can be a integer (every N iterations) or list of integers [0, 500, 1500]
        :param decay_coeff: The decay coefficient of the learning rate.
        :param epsilon: The perturbation budget
        :param dropout: Dropout to apply while training
        :param net_repeat: The number of times to repeat prediction on each network
        :param endtoend: True for end-to-end training. False for transfer learning.
        :param verbose: Show progress bars.
        """
        self.subsistute_networks: List["CLASSIFIER_NEURALNETWORK_TYPE"] = [classifier] if not isinstance(
            classifier, list
        ) else classifier

        super().__init__(classifier=self.subsistute_networks[0])  # type: ignore
        self.target = target
        self.opt = opt
        self.momentum = momentum
        self.decay_iter = decay_iter
        self.epsilon = epsilon
        self.dropout = dropout
        self.net_repeat = net_repeat
        self.endtoend = endtoend
        self.feature_layer = feature_layer
        self.learning_rate = learning_rate
        self.decay_coeff = decay_coeff
        self.max_iter = max_iter
        self.verbose = verbose
        self._check_params()

    def poison(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Iteratively finds optimal attack points starting at values at x

        :param x: The base images to begin the poison process.
        :param y: Target label
        :return: An tuple holding the (poisoning examples, poisoning labels).
        """
        import torch

        class PoisonBatch(torch.nn.Module):
            """
            Implementing this to work with PyTorch optimizers.
            """

            def __init__(self, base_list):
                super(PoisonBatch, self).__init__()
                base_batch = torch.stack(base_list, 0)
                self.poison = torch.nn.Parameter(base_batch.clone())

            def forward(self):
                return self.poison

        base_tensor_list = [torch.from_numpy(sample).to(self.estimator.device) for sample in x]
        poison_batch = PoisonBatch([torch.from_numpy(np.copy(sample)).to(self.estimator.device) for sample in x])
        opt_method = self.opt.lower()

        if opt_method == "sgd":
            logger.info("Using SGD to craft poison samples")
            optimizer = torch.optim.SGD(poison_batch.parameters(), lr=self.learning_rate, momentum=self.momentum)
        elif opt_method == "adam":
            logger.info("Using Adam to craft poison samples")
            optimizer = torch.optim.Adam(poison_batch.parameters(), lr=self.learning_rate, betas=(self.momentum, 0.999))

        base_tensor_batch = torch.stack(base_tensor_list, 0)
        base_range01_batch = base_tensor_batch

        # Because we have turned on DP for the substitute networks,
        # the target image's feature becomes random.
        # We can try enforcing the convex polytope in one of the multiple realizations of the feature,
        # but empirically one realization is enough.
        target_feat_list = []
        # Coefficients for the convex combination.
        # Initializing from the coefficients of last step gives faster convergence.
        s_init_coeff_list = []
        n_poisons = len(x)
        for n, net in enumerate(self.subsistute_networks):
            # End to end training
            if self.endtoend:
                block_feats = [
                    feat.detach() for feat in net.get_activations(x, layer=self.feature_layer, framework=True)
                ]
                target_feat_list.append(block_feats)
                s_coeff = [
                    torch.ones(n_poisons, 1).to(self.estimator.device) / n_poisons for _ in range(len(block_feats))
                ]
            else:
                target_feat_list.append(net.get_activations(x, layer=self.feature_layer, framework=True).detach())
                s_coeff = torch.ones(n_poisons, 1).to(self.estimator.device) / n_poisons

            s_init_coeff_list.append(s_coeff)

        for ite in trange(self.max_iter):
            if ite % self.decay_iter == 0 and ite != 0:
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= self.decay_coeff
                print(
                    "%s Iteration %d, Adjusted lr to %.2e"
                    % (time.strftime("%Y-%m-%d %H:%M:%S"), ite, self.learning_rate)
                )

            poison_batch.zero_grad()
            total_loss = loss_from_center(
                self.subsistute_networks,
                target_feat_list,
                poison_batch,
                self.net_repeat,
                self.endtoend,
                self.feature_layer,
            )
            total_loss.backward()
            optimizer.step()

            # clip the perturbations into the range
            perturb_range01 = torch.clamp((poison_batch.poison.data - base_tensor_batch), -self.epsilon, self.epsilon)
            perturbed_range01 = torch.clamp(
                base_range01_batch.data + perturb_range01.data,
                self.estimator.clip_values[0],
                self.estimator.clip_values[1],
            )
            poison_batch.poison.data = perturbed_range01

        if y is None:
            raise ValueError("You must pass in the target label as y")

        return get_poison_tuples(poison_batch, y)

    def _check_params(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be strictly positive")

        if self.max_iter < 1:
            raise ValueError("Value of max_iter at least 1")

        if not isinstance(self.feature_layer, (str, int)):
            raise TypeError("Feature layer should be a string or int")

        if self.opt.lower() not in ["adam", "sgd"]:
            raise ValueError("Optimizer must be 'adam' or 'sgd'")

        if 1 < self.momentum < 0:
            raise ValueError("Momentum must be between 0 and 1")

        if self.decay_iter < 0:
            raise ValueError("decay_iter must be at least 0")

        if self.epsilon <= 0:
            raise ValueError("epsilon must be at least 0")

        if 1 < self.dropout < 0:
            raise ValueError("dropout must be between 0 and 1")

        if self.net_repeat < 1:
            raise ValueError("net_repeat must be at least 1")

        if not 0 <= self.feature_layer < len(self.estimator.layer_names):
            raise ValueError("Invalid feature layer")

        if 1 < self.decay_coeff < 0:
            raise ValueError("Decay coefficient must be between zero and one")


def get_poison_tuples(poison_batch, poison_label):
    """
    Includes the labels
    """
    poison = [
        poison_batch.poison.data[num_p].unsqueeze(0).detach().cpu().numpy()
        for num_p in range(poison_batch.poison.size(0))
    ]
    return np.vstack(poison), poison_label


def loss_from_center(
    subs_net_list, target_feat_list, poison_batch, net_repeat, end2end, feature_layer
) -> "torch.Tensor":
    import torch

    if end2end:
        loss = 0
        for net, center_feats in zip(subs_net_list, target_feat_list):
            if net_repeat > 1:
                poisons_feats_repeats = [
                    net.get_activations(poison_batch(), layer=feature_layer, framework=True) for _ in range(net_repeat)
                ]
                BLOCK_NUM = len(poisons_feats_repeats[0])
                poisons_feats = []
                for block_idx in range(BLOCK_NUM):
                    poisons_feats.append(
                        sum([poisons_feat_r[block_idx] for poisons_feat_r in poisons_feats_repeats]) / net_repeat
                    )
            elif net_repeat == 1:
                poisons_feats = net.get_activations(poison_batch(), layer=feature_layer, framework=True)
            else:
                assert False, "net_repeat set to {}".format(net_repeat)

            net_loss = 0
            for pfeat, cfeat in zip(poisons_feats, center_feats):
                diff = torch.mean(pfeat, dim=0) - cfeat
                diff_norm = torch.norm(diff, dim=0)
                cfeat_norm = torch.norm(cfeat, dim=0)
                diff_norm = diff_norm / cfeat_norm
                net_loss += torch.mean(diff_norm)
            loss += net_loss / len(center_feats)
        loss = loss / len(subs_net_list)

    else:
        loss = 0
        for net, center in zip(subs_net_list, target_feat_list):
            poisons = [
                net.get_activations(poison_batch(), layer=feature_layer, framework=True) for _ in range(net_repeat)
            ]
            poisons = sum(poisons) / len(poisons)

            diff = torch.mean(poisons, dim=0) - center
            diff_norm = torch.norm(diff, dim=1) / torch.norm(center, dim=1)
            loss += torch.mean(diff_norm)

        loss = loss / len(subs_net_list)

    return loss
