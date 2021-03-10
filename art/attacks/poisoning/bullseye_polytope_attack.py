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
from typing import Optional, Tuple, Union, TYPE_CHECKING, List

import numpy as np
import torch
import math
import time
from torch.optim.optimizer import Optimizer
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
        "opt"
        "max_iter",
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
            opt: str = 'adam', # SGD, adam or signedadam https://github.com/ucsb-seclab/BullseyePoison/blob/65af294fd4136d15282360d5f65b44ae9390444b/trainer.py#L239
            max_iter: int = 4000,
            learning_rate: float = 4e-2,
            momentum: float = 0.9,
            decay_iter: Union[int, List[int]] = 1000,
            decay_coeff: float = 0.5,
            epsilon: float = 0.1,
            norm: Union[float, str] = 'inf',
            dropout: int = 0.3,
            net_repeat: int = 1,
            endtoend: bool = True,
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
        :param net_repeat: The number of times to repeat prediction on each network
        :param endtoend: True for end-to-end training. False for transfer learning.
        :param verbose: Show progress bars.
        """
        self.subsistute_networks: List["CLASSIFIER_NEURALNETWORK_TYPE"] = \
            [classifier] if type(classifier) != type([]) else classifier

        super().__init__(classifier=self.subsistute_networks[0])  # type: ignore
        self.target = target
        self.opt = opt
        self.momentum = momentum
        self.decay_iter = decay_iter
        self.epsilon = epsilon
        self.norm = norm
        self.dropout = dropout
        self.net_repeat = net_repeat
        self.endtoend = endtoend
        self.feature_layer = feature_layer
        self.learning_rate = learning_rate
        self.decay_coeff = decay_coeff
        self.max_iter = max_iter
        self.verbose = verbose
        self._check_params()

    def poison(
            self,
            x: np.ndarray,
            y: Optional[np.ndarray] = None,
            fetch_nearest: bool = False,
            num_poison: Optional[int] = None,
            **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Iteratively finds optimal attack points starting at values at x

        :param x: The base images to begin the poison process.
        :param y: Target label
        :return: An tuple holding the (poisoning examples, poisoning labels).
        """
        # target_net.eval()
        base_tensor_list = [torch.from_numpy(sample).to(self.estimator.device) for sample in x]
        poison_batch = PoisonBatch([torch.from_numpy(np.copy(sample)).to(self.estimator.device) for sample in x])
        opt_method = self.opt.lower()

        if opt_method == 'sgd':
            logger.info("Using SGD to craft poison samples")
            optimizer = torch.optim.SGD(poison_batch.parameters(), lr=self.learning_rate, momentum=self.momentum)
        elif opt_method == 'signedadam':
            logger.info("Using Signed Adam to craft poison samples")
            optimizer = SignedAdam(poison_batch.parameters(), lr=self.learning_rate, betas=(self.momentum, 0.999))
        elif opt_method == 'adam':
            logger.info("Using Adam to craft poison samples")
            optimizer = torch.optim.Adam(poison_batch.parameters(), lr=self.learning_rate, betas=(self.momentum, 0.999))

        target = torch.from_numpy(self.target).to(self.estimator.device)
        mean = torch.Tensor((0.4914, 0.4822, 0.4465)).reshape(1, 3, 1, 1)
        std = torch.Tensor((0.2023, 0.1994, 0.2010)).reshape(1, 3, 1, 1)
        # std, mean = std.to(self.estimator.device), mean.to(self.estimator.device)
        base_tensor_batch = torch.stack(base_tensor_list, 0)
        base_range01_batch = base_tensor_batch * std + mean

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
            net = net.model
            net.eval()
            # End to end training
            if self.endtoend:
                block_feats = [feat.detach() for feat in net(x=target, block=True)]
                target_feat_list.append(block_feats)
                s_coeff = [torch.ones(n_poisons, 1).to(self.estimator.device) / n_poisons for _ in range(len(block_feats))]
            else:
                target_feat_list.append(net(x=target, penu=True).detach())
                s_coeff = torch.ones(n_poisons, 1).to(self.estimator.device) / n_poisons

            s_init_coeff_list.append(s_coeff)

        poisons_time = 0
        start_ite = 0  # TODO: where is this called and this isn't zero?
        for ite in trange(start_ite, self.max_iter):
            if ite in self.decay_iter:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= self.decay_coeff
                print("%s Iteration %d, Adjusted lr to %.2e" % (time.strftime("%Y-%m-%d %H:%M:%S"), ite, self.learning_rate))

            poison_batch.zero_grad()
            t = time.time()
            total_loss = loss_from_center(self.subsistute_networks, target_feat_list, poison_batch, self.net_repeat, self.endtoend)
            total_loss.backward()

            optimizer.step()
            poisons_time += int(time.time() - t)

            # clip the perturbations into the range
            perturb_range01 = torch.clamp((poison_batch.poison.data - base_tensor_batch) * std,
                                          -self.epsilon,
                                          self.epsilon)
            perturbed_range01 = torch.clamp(base_range01_batch.data + perturb_range01.data, 0, 1)
            poison_batch.poison.data = (perturbed_range01 - mean) / std

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

        if self.decay_coeff <= 0:
            raise ValueError("Decay coefficient must be positive")

        # if self.stopping_tol <= 0:
        #     raise ValueError("Stopping tolerance must be positive")
        #
        # if self.obj_threshold and self.obj_threshold <= 0:
        #     raise ValueError("Objective threshold must be positive")
        #
        # if self.num_old_obj <= 0:
        #     raise ValueError("Number of old stored objectives must be positive")
        #
        # if self.max_iter <= 0:
        #     raise ValueError("Number of old stored objectives must be positive")
        #
        # if self.watermark and not (isinstance(self.watermark, float) and 0 <= self.watermark < 1):
        #     raise ValueError("Watermark must be between 0 and 1")
        #
        # if not isinstance(self.verbose, bool):
        #     raise ValueError("The argument `verbose` has to be of type bool.")


def get_poison_tuples(poison_batch, poison_label):
    """
    Includes the labels
    """
    poison_tuple = [(poison_batch.poison.data[num_p].detach().cpu(), poison_label) for num_p in
                    range(poison_batch.poison.size(0))]
    poison, labels = zip(*poison_tuple)
    poison = np.array(poison)
    labels = np.array(labels)
    print(f"final poison shape: {poison.shape}")
    print(f"final labels shape: {labels.shape}")
    return poison, labels


def loss_from_center(subs_net_list, target_feat_list, poison_batch, net_repeat, end2end) -> "torch.Tensor":
    if end2end:
        loss = 0
        for net, center_feats in zip(subs_net_list, target_feat_list):
            if net_repeat > 1:
                poisons_feats_repeats = [net(x=poison_batch(), block=True) for _ in range(net_repeat)]
                BLOCK_NUM = len(poisons_feats_repeats[0])
                poisons_feats = []
                for block_idx in range(BLOCK_NUM):
                    poisons_feats.append(
                        sum([poisons_feat_r[block_idx] for poisons_feat_r in poisons_feats_repeats]) / net_repeat)
            elif net_repeat == 1:
                poisons_feats = net(x=poison_batch(), block=True)
            else:
                assert False, "net_repeat set to {}".format(net_repeat)

            net_loss = 0
            for pfeat, cfeat in zip(poisons_feats, center_feats):
                diff = torch.mean(pfeat, dim=0) - cfeat
                diff_norm = torch.norm(diff, dim=1) / torch.norm(cfeat, dim=1)
                net_loss += torch.mean(diff_norm)
            loss += net_loss / len(center_feats)
        loss = loss / len(subs_net_list)

    else:
        loss = 0
        for net, center in zip(subs_net_list, target_feat_list):
            poisons = [net(x=poison_batch(), penu=True) for _ in range(net_repeat)]
            poisons = sum(poisons) / len(poisons)

            diff = torch.mean(poisons, dim=0) - center
            diff_norm = torch.norm(diff, dim=1) / torch.norm(center, dim=1)
            loss += torch.mean(diff_norm)

        loss = loss / len(subs_net_list)

    return loss


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


class SignedAdam(Optimizer):
    """Implements Signed Adam algorithm. Code stolen from
    https://raw.githubusercontent.com/pytorch/pytorch/v0.4.1/torch/optim/adam.py
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(SignedAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SignedAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                #p.data.addcdiv_(-step_size, exp_avg, denom)
                p.data -= step_size * torch.sign(exp_avg / denom)

        return loss