# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2022
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
This module implements SmoothMix.

| Paper link: https://arxiv.org/pdf/1906.04584.pdf
| Authors' implementation: https://github.com/jh-jeong/smoothmix
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Tuple, Union
import numpy as np

import torch
from art.config import ART_NUMPY_DTYPE

logger = logging.getLogger(__name__)


# pylint: disable=W0613
def fit_pytorch(self, x: np.ndarray, y: np.ndarray, batch_size: int, nb_epochs: int, **kwargs) -> None:
    """
    Performs SmoothMix adversarial training on the model

    :param x: Training data.
    :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes)
              or indices of shape (nb_samples,).
    :param batch_size: Batch size.
    :key nb_epochs: Number of epochs to use for training
    """
    import torch.nn.functional as F
    from art.estimators.certification.randomized_smoothing.smoothmix.smooth_pgd_attack import SmoothMixPGD
    import random

    x = x.astype(ART_NUMPY_DTYPE)
    start_epoch = 0

    if self.maxnorm_s is None:
        self.maxnorm_s = self.alpha * self.mix_step

    if self.attack_type == "PGD":
        attacker = SmoothMixPGD(
            steps=self.num_steps,
            mix_step=self.mix_step,
            alpha=self.alpha,
            maxnorm=self.maxnorm,
            maxnorm_s=self.maxnorm_s
        )

    if self.optimizer is None:  # pragma: no cover
        raise ValueError(f"An optimizer is needed to train the model, but none for provided: {self.optimizer}")
    if self.scheduler is None:  # pragma: no cover
        raise ValueError(f"A scheduler is needed to train the model, but none for provided: {self.scheduler}")
    if attacker is None:
        raise ValueError(f"A attacker is needed to smooth adversarially train the model, but \
                            none for provided: {self.attack_type}")

    num_batch = int(np.ceil(len(x) / float(batch_size)))
    ind = np.arange(len(x))

    # Start training
    for epoch_num in range(start_epoch + 1, nb_epochs + 1):
        # Shuffle the examples
        random.shuffle(ind)

        # SmoothMix Training setup
        warmup_v = np.min([1.0, (epoch_num + 1) / self.warmup])
        attacker.maxnorm_s = warmup_v * self.maxnorm_s

        # Put the model in the training mode
        self.model.train()
        self._requires_grad_(self.model, True)  # pylint: disable=W0212

        for num_batch_index in range(num_batch):
            input_batch = torch.from_numpy(
                x[ind[num_batch_index * batch_size : (num_batch_index + 1) * batch_size]]
            ).to(self.device)
            output_batch = torch.from_numpy(
                y[ind[num_batch_index * batch_size : (num_batch_index + 1) * batch_size]]
            ).to(self.device)

            # pylint: disable=W0212
            mini_batches = self._get_minibatches(
                input_batch, output_batch, self.num_noise_vec
            )

            for (inputs, targets) in mini_batches:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                noises = [torch.randn_like(inputs) * self.scale for _ in range(self.num_noise_vec)]

                # Attack and find adversarial examples
                self._requires_grad_(self.model, False)  # pylint: disable=W0212
                self.model.eval()
                inputs, inputs_adv = attacker.attack(
                    self.model,
                    inputs=inputs,
                    labels=targets,
                    noises=noises
                )
                self.model.train()
                self._requires_grad_(self.model, True)  # pylint: disable=W0212

                in_clean_c = torch.cat([inputs + noise for noise in noises], dim=0)
                logits_c = self.model(in_clean_c)
                targets_c = targets.repeat(self.num_noise_vec)

                logits_c_chunk = torch.chunk(logits_c, self.num_noise_vec, dim=0)
                clean_avg_sm = _avg_softmax(logits_c_chunk)

                if isinstance(clean_avg_sm, float):
                    clean_avg_sm = torch.Tensor(clean_avg_sm)

                loss_xent = F.cross_entropy(logits_c, targets_c, reduction='none')

                in_mix, targets_mix = _mixup_data(inputs, inputs_adv, clean_avg_sm, self.nb_classes)

                in_mix_c = torch.cat([in_mix + noise for noise in noises], dim=0)
                targets_mix_c = targets_mix.repeat(self.num_noise_vec, 1)
                logits_mix_c = F.log_softmax(self.model(in_mix_c), dim=1)

                _, top1_idx = torch.topk(clean_avg_sm, 1)
                ind_correct = (top1_idx[:, 0] == targets).float()
                ind_correct = ind_correct.repeat(self.num_noise_vec)

                loss_mixup = F.kl_div(logits_mix_c, targets_mix_c, reduction='none').sum(1)
                loss = loss_xent.mean() + self.eta * warmup_v * (ind_correct * loss_mixup).mean()

                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.scheduler.step()


def fit_tensorflow(self, x: np.ndarray, y: np.ndarray, batch_size: int, nb_epochs: int, **kwargs) -> None:
    """TODO: fill out"""
    raise NotImplementedError


def get_minibatches(x, y, num_batches):
    """
    Generate batches of the training data and target values

    :param x: Training data
    :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
    :param num_batches: The number of batches to generate
    """
    batch_size = len(x) // num_batches
    for i in range(num_batches):
        yield x[i * batch_size : (i + 1) * batch_size], y[i * batch_size : (i + 1) * batch_size]


def _mixup_data(x_1, x_2, y_1, n_classes) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns mixed inputs, pairs of targets, and lambda

    :param x_1: Training data
    :param x_2: Adversarial training data
    :param y_1: Training labels
    :param n_classes: The number of classes
    """
    device = x_1.device

    _eye = torch.eye(n_classes, device=device)
    _unif = _eye.mean(0, keepdim=True)
    lam = torch.rand(x_1.size(0), device=device) / 2

    mixed_x = (1 - lam).view(-1, 1, 1, 1) * x_1 + lam.view(-1, 1, 1, 1) * x_2
    mixed_y = (1 - lam).view(-1, 1) * y_1 + lam.view(-1, 1) * _unif

    return mixed_x, mixed_y


def _avg_softmax(logits) -> Union[torch.Tensor, float]:
    """
    Computes the average softmax for the given logits

    :param logits: the logits to compute the average softmax of
    """
    import torch.nn.functional as F

    m = len(logits)
    softmax = [F.softmax(logit, dim=1) for logit in logits]
    avg_softmax = sum(softmax) / m
    return avg_softmax
