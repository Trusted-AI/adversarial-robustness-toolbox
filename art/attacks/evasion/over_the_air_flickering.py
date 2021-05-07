# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2021
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
This module contains an implementation of the Over-the-Air Adversarial
Flickering attack on video recognition networks.

Paper link: https://arxiv.org/abs/2002.05123
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, TYPE_CHECKING
import numpy as np
from scipy.special import expit
from tqdm.auto import trange

from art.config import ART_NUMPY_DTYPE
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassGradientsMixin
from art.attacks.attack import EvasionAttack
from art.utils import check_and_transform_label_format

if TYPE_CHECKING:
    import torch
    from art.utils import CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE
logger = logging.getLogger(__name__)

import numpy as np

class OverTheAirFlickeringTorch(EvasionAttack):
    attack_params = EvasionAttack.attack_params + ["regularization_param", "beta_1", "beta_2" "margin"]
    _estimator_requirements = (BaseEstimator, ClassGradientsMixin)

    def __init__(
        self,
        classifier: "CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE",
        regularization_param: float,
        beta_1: float,
        beta_2: float,
        margin: float,
    ):
        """
        Initialize the `OverTheAirFlickeringTorch` attack. Besides the
        `classifier` argument, the rest are hyperparameters from the paper.

        :param classifier: The classifier model.
        :param regularization_param: The hyperparameter `lambda` term in
            equation (1).
        :param beta_1: The hyperparameter `beta_1` term in equation (1).
        :param beta_2: The hyperparameter `beta_2` term in equation (1).
        :param margin: The hyperparameter `m` term in the paper.
        """
        super().__init__(estimator=classifier)

        self.regularization_param = regularization_param
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.margin = margin
        self._check_params()

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> "torch.Tensor":
        """

        :param x: Input videos. Can be any dimensions, but per the paper it
            should be a 5-dimensional Tensor with dimensions

            (N Samples, T consecutive frames, H rows, W columns, C color channels).

        :param y: Labels/Ground Truth. Should be a 1-dimensional tensor with
            dimensions

            (N samples)

            Where each `yi in y` is a class label.

        :param kwargs: Keyword Args

        :return: The perturbed videos `x+delta`. 5-dimensional tensor with
            dimensions

            (N Samples, T consecutive frames, H rows, W columns, C color channels).
        """
        import torch

        y = check_and_transform_label_format(y, self.estimator.nb_classes)

        x = torch.tensor(x, device=self.estimator.device)
        y = torch.tensor(y, device=self.estimator.device)

        num_epochs = 200
        epoch_print_str = f"{num_epochs}:"

        delta = torch.nn.parameter.Parameter(
            torch.zeros(1, 3, 1, 1).normal_(mean=0.0, std=0.2).to(self.estimator.device), requires_grad=True
        )

        # All values of delta needs to be within [V_min, V_max], so we get those
        # values here.
        v_min = torch.min(x).item()
        v_max = torch.max(x).item()

        # Learning rate from the paper.
        optimizer = torch.optim.AdamW([delta], lr=1e-3)

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
                for video in x:
                    preds.append(expit(self.estimator.model(video + delta, return_loss=False)))

            preds = torch.Tensor(preds).to(self.estimator.device).squeeze(1)

            # Calculate the adversarial loss
            loss = self._objective(
                delta,
                preds,
                y.squeeze(1),
            )
            logger.info(f"Epoch {i + 1:>5}/{epoch_print_str:<6} Loss {loss:.7f}")

            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            delta = torch.clamp(delta, v_min, v_max)

        return x + delta.detach().cpu().numpy()

    def _objective(self, delta: "torch.Tensor", predictions: "torch.Tensor", y: Optional["torch.Tensor"] = None):
        """
        Equation (1): The objective function. Does NOT include the argmin nor constraints from
        equation (2).
        :param delta: 3-dimensional perturbation tensor with dimensions

            (H rows, W columns, C color channels).

        :param predictions: Predictions. 1 dimensional tensor with dimensions:

            (N Samples).

        :param y: Labels/Ground Truth. Should be a 1-dimensional tensor with
            dimensions

            (N samples)

        :return: The loss/objective values. Scalar.
        """
        import torch

        T = delta.shape[0]
        # The first summation from equation (1)
        regularization_term = self.regularization_param * (
            self.beta_1 * self._thickness_regularization(delta, T)
            + self.beta_2 * self._roughness_regularization(delta, T)
        )

        return regularization_term + torch.mean(self._adversarial_loss(predictions, y))

    @staticmethod
    def _first_temporal_derivative(x: "torch.Tensor") -> "torch.Tensor":
        """
        Equation 7 from the paper.
        :param x: Input tensor. Can be any dimensions, but per the paper it
            should be a 4-dimensional Tensor with dimensions

            (T consecutive frames, H rows, W columns, C color channels).
        :return: The first order temporal derivative with dimensions

            (T consecutive frames, H rows, W columns, C color channels).
        """
        import torch

        # Use dims to ensure that it is only shifted on the first dimensions.
        # Per the paper, we roll x_1,...,x_T in X. Since T is the first
        # dimension of X, we use dim=0.
        return torch.roll(x, 1, dims=0) - torch.roll(x, 0, dims=0)

    @staticmethod
    def _second_temporal_derivative(X: "torch.Tensor") -> "torch.Tensor":
        """
        Equation 8 from the paper. Defined as:
            Roll(X,-1) - 2*Roll(X, 0) + Roll(X,1)

        :param X: Input tensor. Can be any dimensions, but per the paper it
            should be a 4-dimensional Tensor with dimensions

            (T consecutive frames, H rows, W columns, C color channels).

        :return: The first order temporal derivative with dimensions
            (T consecutive frames, H rows, W columns, C color channels).
        """
        import torch

        # Use dims to ensure that it is only shifted on the first dimensions.
        # Per the paper, we roll x_1,...,x_T in X. Since T is the first
        # dimension of X, we use dim=0.
        return torch.roll(X, -1, dims=0) - 2 * torch.roll(X, 0, dims=0) - torch.roll(X, 1, dims=0)

    def _roughness_regularization(self, delta: "torch.Tensor", T: int) -> "torch.Tensor":
        """
        ROUGH AND ROWDY
        :param delta: Delta parameter from the paper
        :param T:
        :return: Roughness regularization parameter.
        """
        import torch

        return (
            1
            / (3 * T)
            * (
                torch.pow(torch.norm(self._first_temporal_derivative(delta), 2), 2)
                + torch.pow(torch.norm(self._second_temporal_derivative(delta), 2), 2)
            )
        )

    @staticmethod
    def _thickness_regularization(delta: "torch.Tensor", T: int) -> "torch.Tensor":
        """
        Thickness Function
        :param delta: Delta parameter from the paper
        :param T:
        :return: The thickness. Like oatmeal * oatmeal=oatmeal^2
        """
        import torch

        return torch.pow(torch.norm(delta, 2), 2) / (3 * T)

    def _adversarial_loss(self, predictions: "torch.Tensor", y: "torch.Tensor") -> "torch.Tensor":
        """
        C&W adversarial loss

        :param predictions: Predictions. 1 dimensional tensor with dimensions:

            (N Samples).

        :param y: Labels/Ground Truth. Should be a 1-dimensional tensor with
            dimensions

            (N samples)

        :return: The loss.
        """

        import torch

        # Number of samples x Number of Labels
        samples, n = predictions.shape
        pred_mask = torch.ones(samples, n).type(torch.bool)
        pred_mask[torch.arange(end=samples), y[:]] = False

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
            + self.margin
        )

        # Equation 3
        return torch.max(
            torch.zeros(y.shape).to(predictions.device), torch.min(1 / self.margin * torch.pow(l_m, 2), l_m)
        )
