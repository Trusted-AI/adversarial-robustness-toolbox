# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2021
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated
# documentation files (the "Software"), to deal in the Software without restriction, including
# without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
# EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE
# SOFTWARE.

"""
This attack is an implementation of the over the air flickering attack from
https://arxiv.org/pdf/2002.05123.pdf.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, TYPE_CHECKING

import torch
from torch import nn
from scipy.special import expit
from tqdm.auto import trange

from art.config import ART_NUMPY_DTYPE
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassGradientsMixin
from art.attacks.attack import EvasionAttack
from art.utils import compute_success, is_probability

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE
logger = logging.getLogger(__name__)


class OverTheAirPyTorch(EvasionAttack):
    attack_params = EvasionAttack.attack_params + [
        "regularization_param",
        "beta_1",
        "beta_2"
        "m"
    ]
    _estimator_requirements = (BaseEstimator, ClassGradientsMixin)

    def __init__(self,
                 classifier: "CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE",
                 regularization_param: float,
                 beta_1: float,
                 beta_2: float,
                 m: float):
        super(OverTheAirPyTorch, self).__init__(estimator=classifier)

        self.regularization_param = regularization_param
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.m = m
        self._check_params()

    def generate(self,
                 X: torch.Tensor,
                 labels: Optional[torch.Tensor] = None,
                 **kwargs) -> torch.Tensor:

        num_epochs = 200
        epoch_print_str = f"{num_epochs}:"

        delta = nn.parameter.Parameter(
            torch.zeros(X[0].shape[1], 3, 1, 1).normal_(mean=0., std=.2).to(
                torch.device('cuda')),
            requires_grad=True
        )

        # All values of delta needs to be within [V_min, V_max], so we get those
        # values here.
        v_min = torch.min(X).item()
        v_max = torch.max(X).item()
        delta = torch.clamp(delta, v_min, v_max)

        # Learning rate from the paper.
        optimizer = torch.optim.Adam([delta], lr=1e-3)

        # They did not specify a learning rate scheduler or patience.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

        # Training Loop
        for i in range(num_epochs):
            optimizer.zero_grad()
            preds = []

            # We do NOT use grad here for speed increase, and because the loss
            # is wrt delta.
            with torch.no_grad():
                # TODO: Add in Batching
                for video in X:
                    preds.append(expit(self.estimator(video + delta, return_loss=False)))

            preds = torch.tensor(preds).to(torch.device('cuda')).squeeze(1)

            # Calculate the adversarial loss
            loss = self.objective(
                delta,
                preds,
                labels.squeeze(1),
            )
            logger.info(f"Epoch {i + 1:>5}/{epoch_print_str:<6} Loss {loss:.7f}")

            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            delta = torch.clamp(delta, v_min, v_max)

        return delta

    def objective(self,
                  delta: torch.Tensor,
                  predictions: torch.Tensor,
                  labels: Optional[torch.Tensor] = None):
        """
            Equation (1): The objective function. Does NOT include the argmin nor constraints from
            equation (2).
            :param predictions:
            :param labels:
            :param delta:
            :param regularization_param:
            :param beta_1:
            :param beta_2:
            :return:
            """
        T = delta.shape[0]
        # The first summation from equation (1)
        regularization_term = self.regularization_param * (
                self.beta_1 * self.thicknessRegularization(delta, T)
                + self.beta_2 * self.roughnessRegularization(delta, T)
        )

        return regularization_term + torch.mean(self.adversarialLoss(predictions, labels, self.m))

    @staticmethod
    def firstTemporalDerivative(X: torch.Tensor) -> torch.Tensor:
        """
        Equation 7 from the paper.
        :param X: `torch.tensor`
            Input tensor. Can be any dimensions, but per the paper it should be
            a 4-dimensional Tensor with dimensions
            (T consecutive frames, H rows, W columns, C color channels).
        :return: `torch.Tensor`
            The first order temporal derivative with dimensions
            (T consecutive frames, H rows, W columns, C color channels).
        """
        # Use dims to ensure that it is only shifted on the first dimensions.
        # Per the paper, we roll x_1,...,x_T in X. Since T is the first
        # dimension of X, we use dim=0.
        return torch.roll(X, 1, dims=0) - torch.roll(X, 0, dims=0)

    @staticmethod
    def secondTemporalDerivative(X: torch.Tensor) -> torch.Tensor:
        """
        Equation 8 from the paper. Defined as:
            Roll(X,-1) - 2*Roll(X, 0) + Roll(X,1)
        :param X: `torch.tensor`
            Input tensor. Can be any dimensions, but per the paper it should be
            a 4-dimensional Tensor with dimensions
            (T consecutive frames, H rows, W columns, C color channels).
        :return: `torch.Tensor`
            The first order temporal derivative with dimensions
            (T consecutive frames, H rows, W columns, C color channels).
        """
        # Use dims to ensure that it is only shifted on the first dimensions.
        # Per the paper, we roll x_1,...,x_T in X. Since T is the first
        # dimension of X, we use dim=0.
        return (
                torch.roll(X, -1, dims=0)
                - 2 * torch.roll(X, 0, dims=0)
                - torch.roll(X, 1, dims=0)
        )

    def roughnessRegularization(self, delta: torch.Tensor, T: int) -> torch.Tensor:
        """
        ROUGH AND ROWDY
        :param delta: `torch.Tensor`
            Delta parameter from the paper
        :param T:
        :return:
            Rough.
        """
        return 1 / (3 * T) * (
                torch.pow(torch.norm(self.firstTemporalDerivative(delta), 2), 2)
                + torch.pow(torch.norm(self.secondTemporalDerivative(delta), 2), 2)
        )

    # TODO: Also, get rid of the garbage I call most of these comments.
    @staticmethod
    def thicknessRegularization(delta: torch.Tensor, T: int) -> torch.Tensor:
        """
        Thickness Function
        :param delta: `torch.Tensor`
            Delta parameter from the paper
        :param T: `int`

        :return: `torch.Tensor`
            The THICKness. Like oatmeal * oatmeal=oatmeal^2
        """
        return torch.pow(torch.norm(delta, 2), 2) / (3 * T)

    @staticmethod
    def adversarialLoss(predictions: torch.Tensor, labels: torch.Tensor, m: float) -> torch.Tensor:
        """

        :param predictions: Logits?
        :param labels:
        :param m:
        :return:
        """
        # Number of samples x Number of Labels
        samples, n = predictions.shape
        pred_mask = torch.ones(samples, n).type(torch.bool)
        pred_mask[torch.arange(end=samples), labels[:]] = False

        # Equation (4) from the paper:
        #   You need the `==` or else pytorch throws a fit.
        #
        #   predictions[pred_mask == False]:
        #       Get the logits for the true labeled class
        #
        #   torch.max(predictions[pred_mask == True].view(samples,m-1), dim=-1)[0]:
        #       Get the max logit for each row that is not the true class.
        l_m = (
                predictions[pred_mask == False]
                - torch.max(predictions[pred_mask == True].view(samples, n - 1), dim=-1)[0]
                + m
        )

        # Equation 3
        return torch.max(torch.zeros(labels.shape).to(predictions.device),
                         torch.min(1 / m * torch.pow(l_m, 2), l_m))
