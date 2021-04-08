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
This module implements the Reverse Sigmoid perturbation for the classifier output.

| Paper link: https://arxiv.org/abs/1806.00054
"""
import logging

import numpy as np

from art.defences.postprocessor.postprocessor import Postprocessor

logger = logging.getLogger(__name__)


class ReverseSigmoid(Postprocessor):
    """
    Implementation of a postprocessor based on adding the Reverse Sigmoid perturbation to classifier output.
    """

    params = ["beta", "gamma"]

    def __init__(
        self,
        beta: float = 1.0,
        gamma: float = 0.1,
        apply_fit: bool = False,
        apply_predict: bool = True,
    ) -> None:
        """
        Create a ReverseSigmoid postprocessor.

        :param beta: A positive magnitude parameter.
        :param gamma: A positive dataset and model specific convergence parameter.
        :param apply_fit: True if applied during fitting/training.
        :param apply_predict: True if applied during predicting.
        """
        super().__init__(is_fitted=True, apply_fit=apply_fit, apply_predict=apply_predict)
        self.beta = beta
        self.gamma = gamma
        self._check_params()

    def __call__(self, preds: np.ndarray) -> np.ndarray:
        """
        Perform model postprocessing and return postprocessed output.

        :param preds: model output to be postprocessed.
        :return: Postprocessed model output.
        """
        clip_min = 1e-9
        clip_max = 1.0 - clip_min

        def sigmoid(var_z):
            return 1.0 / (1.0 + np.exp(-var_z))

        preds_clipped = np.clip(preds, clip_min, clip_max)

        if preds.shape[1] > 1:
            perturbation_r = self.beta * (sigmoid(-self.gamma * np.log((1.0 - preds_clipped) / preds_clipped)) - 0.5)
            preds_perturbed = preds - perturbation_r
            preds_perturbed = np.clip(preds_perturbed, 0.0, 1.0)
            alpha = 1.0 / np.sum(preds_perturbed, axis=-1, keepdims=True)
            reverse_sigmoid = alpha * preds_perturbed
        else:
            preds_1 = preds
            preds_2 = 1.0 - preds

            preds_clipped_1 = preds_clipped
            preds_clipped_2 = 1.0 - preds_clipped

            perturbation_r_1 = self.beta * (
                sigmoid(-self.gamma * np.log((1.0 - preds_clipped_1) / preds_clipped_1)) - 0.5
            )
            perturbation_r_2 = self.beta * (
                sigmoid(-self.gamma * np.log((1.0 - preds_clipped_2) / preds_clipped_2)) - 0.5
            )

            preds_perturbed_1 = preds_1 - perturbation_r_1
            preds_perturbed_2 = preds_2 - perturbation_r_2

            preds_perturbed_1 = np.clip(preds_perturbed_1, 0.0, 1.0)
            preds_perturbed_2 = np.clip(preds_perturbed_2, 0.0, 1.0)

            alpha = 1.0 / (preds_perturbed_1 + preds_perturbed_2)
            reverse_sigmoid = alpha * preds_perturbed_1

        return reverse_sigmoid

    def _check_params(self) -> None:
        if self.beta <= 0:
            raise ValueError("Magnitude parameter must be positive.")

        if self.gamma <= 0:
            raise ValueError("Convergence parameter must be positive.")
