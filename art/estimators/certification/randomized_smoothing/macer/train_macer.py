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
This module implements Smooth Adversarial Attack using PGD and DDN.

| Paper link: https://arxiv.org/pdf/1906.04584.pdf
| Authors' implementation: https://github.com/Hadisalman/smoothing-adversarial
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import TYPE_CHECKING
import numpy as np

from art.config import ART_NUMPY_DTYPE

if TYPE_CHECKING:
    # pylint: disable=C0412
    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor import Preprocessor
    from art.defences.postprocessor import Postprocessor

logger = logging.getLogger(__name__)


def fit_pytorch(self, x: np.ndarray, y: np.ndarray, batch_size: int, nb_epochs: int, **kwargs) -> None:
    import torch
    import torch.nn.functional as F
    from torch.distributions.normal import Normal
    import random

    x = x.astype(ART_NUMPY_DTYPE)
    m = Normal(torch.tensor([0.0]).to(self._device), torch.tensor([1.0]).to(self._device))
    cl_total = 0.0
    rl_total = 0.0
    input_total = 0
    start_epoch = 0

    # Put the model in the training mode
    self.model.train()

    if self.optimizer is None:  # pragma: no cover
        raise ValueError("An optimizer is needed to train the model, but none for provided.")
    if self.scheduler is None:  # pragma: no cover
        raise ValueError("A scheduler is needed to train the model, but none for provided.")

    if kwargs.get('checkpoint') is not None:
        chkpt = kwargs.get('checkpoint')
        cpoint = torch.load(chkpt)
        self.model.load_state_dict(cpoint['net'])
        start_epoch = cpoint['epoch']
        self.scheduler.step(start_epoch)
        print('Loading model from epoch {} and checkpoint {}'.format(str(start_epoch), str(chkpt)))
    num_batch = int(np.ceil(len(x) / float(batch_size)))
    ind = np.arange(len(x))

    # Start training
    for epoch_num in range(start_epoch + 1, nb_epochs + 1):
        # Shuffle the examples
        random.shuffle(ind)
        i = 0
        # Train for one epoch
        for nb in range(num_batch):
            i_batch = torch.from_numpy(x[ind[nb * batch_size : (nb + 1) * batch_size]]).to(self.device)
            o_batch = torch.from_numpy(y[ind[nb * batch_size : (nb + 1) * batch_size]]).to(self.device)
            input_size = len(i_batch)
            input_total += input_size

            new_shape = [input_size * self.gauss_num]
            new_shape.extend(i_batch[0].shape)
            i_batch = i_batch.repeat((1, self.gauss_num, 1, 1)).view(new_shape)
            noise = torch.randn_like(i_batch, device=self.device) * self.scale
            noisy_inputs = i_batch + noise
            outputs = self.model(noisy_inputs)
            outputs = outputs.reshape((input_size, self.gauss_num, self.nb_classes))

            # Classification loss
            outputs_softmax = F.softmax(outputs, dim=2).mean(1)
            outputs_logsoftmax = torch.log(outputs_softmax + 1e-10)  # avoid nan
            classification_loss = F.nll_loss(
                outputs_logsoftmax, o_batch, reduction='sum')

            cl_total += classification_loss.item()

            # Robustness loss
            beta_outputs = outputs * self.beta  # only apply beta to the robustness loss
            beta_outputs_softmax = F.softmax(beta_outputs, dim=2).mean(1)
            top2 = torch.topk(beta_outputs_softmax, 2)
            top2_score = top2[0]
            top2_idx = top2[1]
            indices_correct = (top2_idx[:, 0] == o_batch)  # G_theta
            out0, out1 = top2_score[indices_correct,
                                    0], top2_score[indices_correct, 1]
            robustness_loss = m.icdf(out1) - m.icdf(out0)
            indices = ~torch.isnan(robustness_loss) & ~torch.isinf(
                robustness_loss) & (torch.abs(robustness_loss) <= self.gamma)  # hinge
            out0, out1 = out0[indices], out1[indices]
            robustness_loss = m.icdf(out1) - m.icdf(out0) + self.gamma
            robustness_loss = robustness_loss.sum() * self.scale / 2
            rl_total += robustness_loss.item()

            # Final objective function
            loss = classification_loss + self.lbd * robustness_loss
            loss /= input_size
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            i = i + 1

        self.scheduler.step()

        cl_total /= input_total
        rl_total /= input_total
