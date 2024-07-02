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
This module implements the Projected Gradient Descent attack `ProjectedGradientDescent` as an iterative method in which,
after each iteration, the perturbation is projected on an lp-ball of specified radius (in addition to clipping the
values of the adversarial sample so that it lies in the permitted data range). This is the attack proposed by Madry et
al. for adversarial training.

| Paper link: https://arxiv.org/abs/1706.06083
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, Union, TYPE_CHECKING

import numpy as np
from tqdm.auto import tqdm

from art.config import ART_NUMPY_DTYPE
from art.summary_writer import SummaryWriter
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.estimators.classification.classifier import ClassifierMixin
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent_numpy import (
    ProjectedGradientDescentCommon,
)
from art.utils import compute_success, random_sphere, compute_success_array

if TYPE_CHECKING:
    # pylint: disable=C0412
    import torch
    from art.estimators.classification.pytorch import PyTorchClassifier

logger = logging.getLogger(__name__)


class ProjectedGradientDescentPyTorch(ProjectedGradientDescentCommon):
    """
    The Projected Gradient Descent attack is an iterative method in which, after each iteration, the perturbation is
    projected on an lp-ball of specified radius (in addition to clipping the values of the adversarial sample so that it
    lies in the permitted data range). This is the attack proposed by Madry et al. for adversarial training.

    | Paper link: https://arxiv.org/abs/1706.06083
    """

    _estimator_requirements = (BaseEstimator, LossGradientsMixin, ClassifierMixin)  # type: ignore

    def __init__(
        self,
        estimator: Union["PyTorchClassifier"],
        norm: Union[int, float, str] = np.inf,
        eps: Union[int, float, np.ndarray] = 0.3,
        eps_step: Union[int, float, np.ndarray] = 0.1,
        decay: Optional[float] = None,
        max_iter: int = 100,
        targeted: bool = False,
        num_random_init: int = 0,
        batch_size: int = 32,
        random_eps: bool = False,
        summary_writer: Union[str, bool, SummaryWriter] = False,
        verbose: bool = True,
    ):
        """
        Create a :class:`.ProjectedGradientDescentPyTorch` instance.

        :param estimator: An trained estimator.
        :param norm: The norm of the adversarial perturbation, supporting  "inf", `np.inf` or a real `p >= 1`.
                     Currently, when `p` is not infinity, the projection step only rescales the noise, which may be
                     suboptimal for `p != 2`.
        :param eps: Maximum perturbation that the attacker can introduce.
        :param eps_step: Attack step size (input variation) at each iteration.
        :param random_eps: When True, epsilon is drawn randomly from truncated normal distribution. The literature
                           suggests this for FGSM based training to generalize across different epsilons. eps_step is
                           modified to preserve the ratio of eps / eps_step. The effectiveness of this method with PGD
                           is untested (https://arxiv.org/pdf/1611.01236.pdf).
        :param max_iter: The maximum number of iterations.
        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False).
        :param num_random_init: Number of random initialisations within the epsilon ball. For num_random_init=0 starting
                                at the original input.
        :param batch_size: Size of the batch on which adversarial samples are generated.
        :param summary_writer: Activate summary writer for TensorBoard.
                               Default is `False` and deactivated summary writer.
                               If `True` save runs/CURRENT_DATETIME_HOSTNAME in current directory.
                               If of type `str` save in path.
                               If of type `SummaryWriter` apply provided custom summary writer.
                               Use hierarchical folder structure to compare between runs easily. e.g. pass in
                               ‘runs/exp1’, ‘runs/exp2’, etc. for each new experiment to compare across them.
        :param verbose: Show progress bars.
        """
        if not estimator.all_framework_preprocessing:
            raise NotImplementedError(
                "The framework-specific implementation only supports framework-specific preprocessing."
            )

        if summary_writer and num_random_init > 1:
            raise ValueError("TensorBoard is not yet supported for more than 1 random restart (num_random_init>1).")

        super().__init__(
            estimator=estimator,
            norm=norm,
            eps=eps,
            eps_step=eps_step,
            decay=decay,
            max_iter=max_iter,
            targeted=targeted,
            num_random_init=num_random_init,
            batch_size=batch_size,
            random_eps=random_eps,
            verbose=verbose,
            summary_writer=summary_writer,
        )

        self._batch_id = 0
        self._i_max_iter = 0

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :param mask: An array with a mask broadcastable to input `x` defining where to apply adversarial perturbations.
                     Shape needs to be broadcastable to the shape of x and can also be of the same shape as `x`. Any
                     features for which the mask is zero will not be adversarially perturbed.
        :type mask: `np.ndarray`
        :return: An array holding the adversarial examples.
        """
        import torch

        mask = self._get_mask(x, **kwargs)

        # Ensure eps is broadcastable
        self._check_compatibility_input_and_eps(x=x)

        # Check whether random eps is enabled
        self._random_eps()

        # Set up targets
        targets = self._set_targets(x, y)

        # Create dataset
        if mask is not None:
            # Here we need to make a distinction: if the masks are different for each input, we need to index
            # those for the current batch. Otherwise (i.e. mask is meant to be broadcasted), keep it as it is.
            if len(mask.shape) == len(x.shape):
                dataset = torch.utils.data.TensorDataset(
                    torch.from_numpy(x.astype(ART_NUMPY_DTYPE)),
                    torch.from_numpy(targets.astype(ART_NUMPY_DTYPE)),
                    torch.from_numpy(mask.astype(ART_NUMPY_DTYPE)),
                )

            else:
                dataset = torch.utils.data.TensorDataset(
                    torch.from_numpy(x.astype(ART_NUMPY_DTYPE)),
                    torch.from_numpy(targets.astype(ART_NUMPY_DTYPE)),
                    torch.from_numpy(np.array([mask.astype(ART_NUMPY_DTYPE)] * x.shape[0])),
                )

        else:
            dataset = torch.utils.data.TensorDataset(
                torch.from_numpy(x.astype(ART_NUMPY_DTYPE)),
                torch.from_numpy(targets.astype(ART_NUMPY_DTYPE)),
            )

        data_loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=self.batch_size, shuffle=False, drop_last=False
        )

        # Start to compute adversarial examples
        adv_x = x.astype(ART_NUMPY_DTYPE)

        # Compute perturbation with batching
        for batch_id, batch_all in enumerate(
            tqdm(data_loader, desc="PGD - Batches", leave=False, disable=not self.verbose)
        ):

            self._batch_id = batch_id

            if mask is not None:
                (batch, batch_labels, mask_batch) = batch_all[0], batch_all[1], batch_all[2]
            else:
                (batch, batch_labels, mask_batch) = batch_all[0], batch_all[1], None

            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size

            batch_eps: Union[int, float, np.ndarray]
            batch_eps_step: Union[int, float, np.ndarray]

            # Compute batch_eps and batch_eps_step
            if isinstance(self.eps, np.ndarray) and isinstance(self.eps_step, np.ndarray):
                if len(self.eps.shape) == len(x.shape) and self.eps.shape[0] == x.shape[0]:
                    batch_eps = self.eps[batch_index_1:batch_index_2]
                    batch_eps_step = self.eps_step[batch_index_1:batch_index_2]

                else:
                    batch_eps = self.eps
                    batch_eps_step = self.eps_step

            else:
                batch_eps = self.eps
                batch_eps_step = self.eps_step

            for rand_init_num in range(max(1, self.num_random_init)):
                if rand_init_num == 0:
                    # first iteration: use the adversarial examples as they are the only ones we have now
                    adv_x[batch_index_1:batch_index_2] = self._generate_batch(
                        x=batch, targets=batch_labels, mask=mask_batch, eps=batch_eps, eps_step=batch_eps_step
                    )
                else:
                    adversarial_batch = self._generate_batch(
                        x=batch, targets=batch_labels, mask=mask_batch, eps=batch_eps, eps_step=batch_eps_step
                    )

                    # return the successful adversarial examples
                    attack_success = compute_success_array(
                        self.estimator,
                        batch,
                        batch_labels,
                        adversarial_batch,
                        self.targeted,
                        batch_size=self.batch_size,
                    )
                    adv_x[batch_index_1:batch_index_2][attack_success] = adversarial_batch[attack_success]

        logger.info(
            "Success rate of attack: %.2f%%",
            100 * compute_success(self.estimator, x, targets, adv_x, self.targeted, batch_size=self.batch_size),
        )

        if self.summary_writer is not None:
            self.summary_writer.reset()

        return adv_x

    def _generate_batch(
        self,
        x: "torch.Tensor",
        targets: "torch.Tensor",
        mask: "torch.Tensor",
        eps: Union[int, float, np.ndarray],
        eps_step: Union[int, float, np.ndarray],
    ) -> np.ndarray:
        """
        Generate a batch of adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :param targets: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)`.
        :param mask: An array with a mask to be applied to the adversarial perturbations. Shape needs to be
                     broadcastable to the shape of x. Any features for which the mask is zero will not be adversarially
                     perturbed.
        :param eps: Maximum perturbation that the attacker can introduce.
        :param eps_step: Attack step size (input variation) at each iteration.
        :return: Adversarial examples.
        """
        import torch

        inputs = x.to(self.estimator.device)
        targets = targets.to(self.estimator.device)
        adv_x = torch.clone(inputs)
        momentum = torch.zeros(inputs.shape).to(self.estimator.device)

        if mask is not None:
            mask = mask.to(self.estimator.device)

        for i_max_iter in range(self.max_iter):
            self._i_max_iter = i_max_iter
            adv_x = self._compute_pytorch(
                adv_x, inputs, targets, mask, eps, eps_step, self.num_random_init > 0 and i_max_iter == 0, momentum
            )

        return adv_x.cpu().detach().numpy()

    def _compute_perturbation_pytorch(  # pylint: disable=W0221
        self, x: "torch.Tensor", y: "torch.Tensor", mask: Optional["torch.Tensor"], momentum: "torch.Tensor"
    ) -> "torch.Tensor":
        """
        Compute perturbations.

        :param x: Current adversarial examples.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :param mask: An array with a mask broadcastable to input `x` defining where to apply adversarial perturbations.
                     Shape needs to be broadcastable to the shape of x and can also be of the same shape as `x`. Any
                     features for which the mask is zero will not be adversarially perturbed.
        :return: Perturbations.
        """
        import torch

        # Get gradient wrt loss; invert it if attack is targeted
        grad = self.estimator.loss_gradient(x=x, y=y) * (-1 if self.targeted else 1)

        # Write summary
        if self.summary_writer is not None:  # pragma: no cover
            self.summary_writer.update(
                batch_id=self._batch_id,
                global_step=self._i_max_iter,
                grad=grad.cpu().detach().numpy(),
                patch=None,
                estimator=self.estimator,
                x=x.cpu().detach().numpy(),
                y=y.cpu().detach().numpy(),
                targeted=self.targeted,
            )

        # Check for nan before normalisation an replace with 0
        if torch.any(grad.isnan()):  # pragma: no cover
            logger.warning("Elements of the loss gradient are NaN and have been replaced with 0.0.")
            grad[grad.isnan()] = 0.0

        # Apply mask
        if mask is not None:
            grad = torch.where(mask == 0.0, torch.tensor(0.0).to(self.estimator.device), grad)

        # Compute gradient momentum
        if self.decay is not None:
            # Update momentum in-place (important).
            # The L1 normalization for accumulation is an arbitrary choice of the paper.
            grad_2d = grad.reshape(len(grad), -1)
            norm1 = torch.linalg.norm(grad_2d, ord=1, dim=1, keepdim=True)
            normalized_grad = (grad_2d * norm1.where(norm1 == 0, 1 / norm1)).reshape(grad.shape)
            momentum *= self.decay
            momentum += normalized_grad
            # Use the momentum to compute the perturbation, instead of the gradient
            grad = momentum

        # Apply norm bound
        norm: float = np.inf if self.norm == "inf" else float(self.norm)
        grad_2d = grad.reshape(len(grad), -1)
        if norm == np.inf:
            grad_2d = torch.ones_like(grad_2d)
        elif norm == 1:
            i_max = torch.argmax(grad_2d.abs(), dim=1)
            grad_2d = torch.zeros_like(grad_2d)
            grad_2d[range(len(grad_2d)), i_max] = 1
        elif norm > 1:
            conjugate = norm / (norm - 1)
            q_norm = torch.linalg.norm(grad_2d, ord=conjugate, dim=1, keepdim=True)
            grad_2d = (grad_2d.abs() * q_norm.where(q_norm == 0, 1 / q_norm)) ** (conjugate - 1)

        grad = grad_2d.reshape(grad.shape) * grad.sign()

        assert x.shape == grad.shape

        return grad

    def _apply_perturbation_pytorch(  # pylint: disable=W0221
        self, x: "torch.Tensor", perturbation: "torch.Tensor", eps_step: Union[int, float, np.ndarray]
    ) -> "torch.Tensor":
        """
        Apply perturbation on examples.

        :param x: Current adversarial examples.
        :param perturbation: Current perturbations.
        :param eps_step: Attack step size (input variation) at each iteration.
        :return: Adversarial examples.
        """
        import torch

        eps_step = np.array(eps_step, dtype=ART_NUMPY_DTYPE)
        perturbation_step = torch.tensor(eps_step).to(self.estimator.device) * perturbation
        perturbation_step[torch.isnan(perturbation_step)] = 0
        x = x + perturbation_step
        if self.estimator.clip_values is not None:
            clip_min, clip_max = self.estimator.clip_values
            x = torch.max(
                torch.min(x, torch.tensor(clip_max).to(self.estimator.device)),
                torch.tensor(clip_min).to(self.estimator.device),
            )

        return x

    def _compute_pytorch(
        self,
        x: "torch.Tensor",
        x_init: "torch.Tensor",
        y: "torch.Tensor",
        mask: "torch.Tensor",
        eps: Union[int, float, np.ndarray],
        eps_step: Union[int, float, np.ndarray],
        random_init: bool,
        momentum: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Compute adversarial examples for one iteration.

        :param x: Current adversarial examples.
        :param x_init: An array with the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236).
        :param mask: An array with a mask broadcastable to input `x` defining where to apply adversarial perturbations.
                     Shape needs to be broadcastable to the shape of x and can also be of the same shape as `x`. Any
                     features for which the mask is zero will not be adversarially perturbed.
        :param eps: Maximum perturbation that the attacker can introduce.
        :param eps_step: Attack step size (input variation) at each iteration.
        :param random_init: Random initialisation within the epsilon ball. For random_init=False starting at the
                            original input.
        :return: Adversarial examples.
        """
        import torch

        if random_init:
            n = x.shape[0]
            m = np.prod(x.shape[1:]).item()

            random_perturbation_array = random_sphere(n, m, eps, self.norm).reshape(x.shape).astype(ART_NUMPY_DTYPE)
            random_perturbation = torch.from_numpy(random_perturbation_array).to(self.estimator.device)

            if mask is not None:
                random_perturbation = random_perturbation * mask

            x_adv = x + random_perturbation

            if self.estimator.clip_values is not None:
                clip_min, clip_max = self.estimator.clip_values
                x_adv = torch.max(
                    torch.min(x_adv, torch.tensor(clip_max).to(self.estimator.device)),
                    torch.tensor(clip_min).to(self.estimator.device),
                )

        else:
            x_adv = x

        # Get perturbation
        perturbation = self._compute_perturbation_pytorch(x_adv, y, mask, momentum)

        # Apply perturbation and clip
        x_adv = self._apply_perturbation_pytorch(x_adv, perturbation, eps_step)

        # Do projection
        perturbation = self._projection(x_adv - x_init, eps, self.norm)

        # Recompute x_adv
        x_adv = perturbation + x_init

        return x_adv

    @staticmethod
    def _projection(
        values: "torch.Tensor",
        eps: Union[int, float, np.ndarray],
        norm_p: Union[int, float, str],
        *,
        suboptimal: bool = True,
    ) -> "torch.Tensor":
        """
        Project `values` on the L_p norm ball of size `eps`.

        :param values: Values to clip.
        :param eps: If a scalar, the norm of the L_p ball onto which samples are projected. Equivalently in general,
                    can be any array of non-negatives broadcastable with `values`, and the projection occurs onto the
                    unit ball for the weighted L_{p, w} norm with `w = 1 / eps`. Currently, for any given sample,
                    non-uniform weights are only supported with infinity norm. Example: To specify sample-wise scalar,
                    you can provide `eps.shape = (n_samples,) + (1,) * values[0].ndim`.
        :param norm_p: Lp norm to use for clipping, with `norm_p > 0`. Only 2, `np.inf` and "inf" are supported
                       with `suboptimal=False` for now.
        :param suboptimal: If `True` simply projects by rescaling to Lp ball. Fast but may be suboptimal for
                           `norm_p != 2`.
                       Ignored when `norm_p in [np.inf, "inf"]` because optimal solution is fast. Defaults to `True`.
        :return: Values of `values` after projection.
        """
        import torch

        norm = np.inf if norm_p == "inf" else float(norm_p)
        assert norm > 0

        values_tmp = values.reshape(len(values), -1)  # (n_samples, d)

        eps = np.broadcast_to(eps, values.shape)
        eps = eps.reshape(len(eps), -1)  # (n_samples, d)
        assert np.all(eps >= 0)
        if norm != np.inf and not np.all(eps == eps[:, [0]]):
            raise NotImplementedError(
                "Projection onto the weighted L_p ball is currently not supported with finite `norm_p`."
            )

        if (suboptimal or norm == 2) and norm != np.inf:  # Simple rescaling
            values_norm = torch.linalg.norm(values_tmp, ord=norm, dim=1, keepdim=True)  # (n_samples, 1)
            values_tmp = values_tmp * values_norm.where(
                values_norm == 0, torch.minimum(torch.ones(1), torch.tensor(eps).to(values_tmp.device) / values_norm)
            )
        else:  # Optimal
            if norm == np.inf:  # Easy exact case
                values_tmp = values_tmp.sign() * torch.minimum(
                    values_tmp.abs(), torch.tensor(eps).to(values_tmp.device)
                )
            elif norm >= 1:  # Convex optim
                raise NotImplementedError(
                    "Finite values of `norm_p >= 1` are currently not supported with `suboptimal=False`."
                )
            else:  # Non-convex optim
                raise NotImplementedError("Values of `norm_p < 1` are currently not supported with `suboptimal=False`")

        values = values_tmp.reshape(values.shape).to(values.dtype)

        return values
