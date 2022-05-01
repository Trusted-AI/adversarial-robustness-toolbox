# MIT License

# Copyright (c) 2021 Jongheon Jeong, Sejun Park, Minkyu Kim, Heung-Chang Lee, Doguk Kim and Jinwoo Shin

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This is authors' implementation of SmoothMix_PGD

| Paper link: https://arxiv.org/pdf/2111.09277.pdf

"""

import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional

class SmoothMix_PGD(object):
    def __init__(self,
                 steps: int,
                 mix_step: int,
                 alpha: Optional[float] = None,
                 maxnorm_s: Optional[float] = None,
                 maxnorm: Optional[float] = None) -> None:
        super(SmoothMix_PGD, self).__init__()
        self.steps = steps
        self.mix_step = mix_step
        self.alpha = alpha
        self.maxnorm = maxnorm
        if maxnorm_s is None:
            self.maxnorm_s = alpha * mix_step
        else:
            self.maxnorm_s = maxnorm_s

    def attack(self, model, inputs, labels, noises=None):
        if inputs.min() < 0 or inputs.max() > 1: raise ValueError('Input values should be in the [0, 1] range.')

        def _batch_l2norm(x):
            x_flat = x.reshape(x.size(0), -1)
            return torch.norm(x_flat, dim=1)

        def _project(x, x0, maxnorm=None):
            if maxnorm is not None:
                eta = x - x0
                eta = eta.renorm(p=2, dim=0, maxnorm=maxnorm)
                x = x0 + eta
            x = torch.clamp(x, 0, 1)
            x = x.detach()
            return x

        adv = inputs.detach()
        init = inputs.detach()
        for i in range(self.steps):
            if i == self.mix_step:
                init = adv.detach()
            adv.requires_grad_()

            softmax = [F.softmax(model(adv + noise), dim=1) for noise in noises]
            avg_softmax = sum(softmax) / len(noises)
            logsoftmax = torch.log(avg_softmax.clamp(min=1e-20))

            loss = F.nll_loss(logsoftmax, labels, reduction='sum')

            grad = torch.autograd.grad(loss, [adv])[0]
            grad_norm = _batch_l2norm(grad).view(-1, 1, 1, 1)
            grad = grad / (grad_norm + 1e-8)
            adv = adv + self.alpha * grad

            adv = _project(adv, inputs, self.maxnorm)
        init = _project(init, inputs, self.maxnorm_s)

        return init, adv