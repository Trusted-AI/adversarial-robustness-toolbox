"""
This module implements the total variance minimization defence `TotalVarMin` in PyTorch.

| Paper link: https://openreview.net/forum?id=SyJ7ClWCb

| Please keep in mind the limitations of defences. For more information on the limitations of this defence,
    see https://arxiv.org/abs/1802.00420 . For details on how to evaluate classifier security in general, see
    https://arxiv.org/abs/1902.06705
"""
from __future__ import absolute_import, division, print_function, unicode_literals, annotations

import logging
from typing import TYPE_CHECKING

from tqdm.auto import tqdm

import numpy as np

from art.defences.preprocessor.preprocessor import PreprocessorPyTorch

if TYPE_CHECKING:
    from art.utils import CLIP_VALUES_TYPE

logger = logging.getLogger(__name__)


class TotalVarMinPyTorch(PreprocessorPyTorch):
    """
    Implement the total variance minimization defence approach in PyTorch.

    | Paper link: https://openreview.net/forum?id=SyJ7ClWCb

    | Please keep in mind the limitations of defences. For more information on the limitations of this
        defence, see https://arxiv.org/abs/1802.00420 . For details on how to evaluate classifier security in general,
        see https://arxiv.org/abs/1902.06705
    """

    def __init__(
        self,
        prob: float = 0.3,
        norm: int = 1,
        lamb: float = 0.5,
        max_iter: int = 10,
        channels_first: bool = True,
        clip_values: "CLIP_VALUES_TYPE" | None = None,
        apply_fit: bool = False,
        apply_predict: bool = True,
        verbose: bool = False,
        device_type: str = "gpu"
    ) -> None:
        """
        Create an instance of total variance minimization in PyTorch.

        :param prob: Probability of the Bernoulli distribution.
        :param norm: The norm (positive integer).
        :param lamb: The lambda parameter in the objective function.
        :param max_iter: Maximum number of iterations when performing optimization.
        :param channels_first: Set channels first or last.
        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :param apply_fit: True if applied during fitting/training.
        :param apply_predict: True if applied during predicting.
        :param verbose: Show progress bars.
        :param device_type: Type of device on which the classifier is run, either `gpu` or `cpu`.
        """
        super().__init__(
            device_type=device_type, 
            apply_fit=apply_fit, 
            apply_predict=apply_predict
        )

        self.prob = prob
        self.norm = norm
        self.lamb = lamb
        self.max_iter = max_iter
        self.channels_first = channels_first
        self.clip_values = clip_values
        self.verbose = verbose
        self._check_params()

    def forward(
        self, x: "torch.Tensor", y: "torch.Tensor" | None = None
    ) -> tuple["torch.Tensor", "torch.Tensor" | None]:
        """
        Apply total variance minimization to sample `x`.

        :param x: Sample to compress with shape `(batch_size, channels, height, width)`.
        :param y: Labels of the sample `x`. This function does not affect them in any way.
        :return: Similar samples.
        """
        import torch

        if len(x.shape) != 4:
            raise ValueError("Input `x` must be a 4D tensor (batch, channels, width, height).")

        if not self.channels_first:
            # BHWC -> BCHW
            x = x.permute(0, 3, 1, 2)

        x_preproc = x.clone()

        B, C, H, W = x_preproc.shape

        # Minimize one input at a time (iterate over the batch dimension)
        for i in tqdm(range(B), desc="Variance minimization", disable=not self.verbose):
            mask = (torch.rand_like(x_preproc[i]) < self.prob).float()

            # Skip optimization if mask is all zeros (prob=0.0 case)
            if torch.sum(mask) > 0:
                x_preproc[i] = self._minimize(x_preproc[i], mask)

        if self.clip_values is not None:
            x_preproc = torch.clamp(x_preproc, self.clip_values[0], self.clip_values[1])

        if not self.channels_first:
            # BCHW -> BHWC
            x_preproc = x_preproc.permute(0, 2, 3, 1)

        return x_preproc, y

    def _minimize(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Minimize the total variance objective function for a single 3D image by
        iterating through its channels.

        :param x: Original image.
        :param mask: A matrix that decides which points are kept.
        :return: A new image.
        """
        import torch

        # Create a tensor to hold the final results for each channel
        z_min = x.clone()
        C, H, W = x.shape

        # Iterate over each channel of the single image
        for c in range(C):
            # Skip channel if no mask points in this channel
            if torch.sum(mask[c, :, :]) == 0:
                continue
                
            # Create a separate, optimizable variable for the current channel
            res = x[c, :, :].clone().detach().requires_grad_(True)

            # The optimizer works on this specific channel variable
            optimizer = torch.optim.LBFGS([res], max_iter=self.max_iter)

            def closure():
                optimizer.zero_grad()
                # Loss is calculated only for the current 2D channel
                loss = self._loss_func(z_init=res.flatten(), x=x[c, :, :], mask=mask[c, :, :], norm=self.norm, lamb=self.lamb)
                loss.backward(retain_graph=True)
                return loss

            optimizer.step(closure)

            # Place the optimized channel back into our result tensor
            with torch.no_grad():
                z_min[c, :, :] = res.view_as(z_min[c, :, :])

        return z_min

    @staticmethod
    def _loss_func(z_init: torch.Tensor, x: torch.Tensor, mask: torch.Tensor, norm: float, lamb: float, eps=1e-6) -> torch.Tensor:
        """
        Loss function to be minimized - try to match SciPy implementation closely.

        :param z_init: Initial guess.
        :param x: Original image.
        :param mask: A matrix that decides which points are kept.
        :param norm: The norm (positive integer).
        :param lamb: The lambda parameter in the objective function.
        :return: A single scalar loss value.
        """
        import torch
        
        # Flatten inputs for pixel-wise loss
        x_flat = x.flatten()
        mask_flat = mask.flatten().float()

        # Data fidelity term
        res = torch.sqrt( ((z_init - x_flat)**2 * mask_flat).sum() + eps )

        z2d = z_init.view(x.shape)

        # Total variation terms
        if norm == 1:
            # L1 norm: sum of absolute differences per row/column
            tv_h = lamb * torch.abs(z2d[1:, :] - z2d[:-1, :]).sum(dim=1).sum()
            tv_w = lamb * torch.abs(z2d[:, 1:] - z2d[:, :-1]).sum(dim=0).sum()
        elif norm == 2:
            # L2 norm: sqrt of sum of squares per row/column, then sum
            tv_h = lamb * torch.sqrt(((z2d[1:, :] - z2d[:-1, :])**2).sum(dim=1) + eps).sum()
            tv_w = lamb * torch.sqrt(((z2d[:, 1:] - z2d[:, :-1])**2).sum(dim=0) + eps).sum()
        else:
            # General Lp norm
            tv_h = lamb * torch.pow(torch.abs(z2d[1:, :] - z2d[:-1, :]), norm).sum(dim=1).pow(1/norm).sum()
            tv_w = lamb * torch.pow(torch.abs(z2d[:, 1:] - z2d[:, :-1]), norm).sum(dim=0).pow(1/norm).sum()
        
        tv = tv_h + tv_w

        return res + tv

    def _check_params(self) -> None:
        if not isinstance(self.prob, (float, int)) or self.prob < 0.0 or self.prob > 1.0:
            logger.error("Probability must be between 0 and 1.")
            raise ValueError("Probability must be between 0 and 1.")

        if not isinstance(self.norm, int) or self.norm <= 0:
            logger.error("Norm must be a positive integer.")
            raise ValueError("Norm must be a positive integer.")

        if not isinstance(self.max_iter, int) or self.max_iter <= 0:
            logger.error("Number of iterations must be a positive integer.")
            raise ValueError("Number of iterations must be a positive integer.")

        if self.clip_values is not None:

            if len(self.clip_values) != 2:
                raise ValueError("'clip_values' should be a tuple of 2 floats or arrays containing the allowed data range.")

            if np.array(self.clip_values[0] >= self.clip_values[1]).any():
                raise ValueError("Invalid 'clip_values': min >= max.")

        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be of type bool.")
