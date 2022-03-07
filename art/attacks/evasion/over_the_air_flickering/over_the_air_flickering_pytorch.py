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
This module contains an implementation of the Over-the-Air Adversarial Flickering attack on video recognition networks.

| Paper link: https://arxiv.org/abs/2002.05123
"""

import logging
from typing import Optional, TYPE_CHECKING

import numpy as np
from tqdm.auto import tqdm

from art.config import ART_NUMPY_DTYPE
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin, LossGradientsMixin
from art.attacks.attack import EvasionAttack
from art.utils import check_and_transform_label_format, get_labels_np_array

if TYPE_CHECKING:
    # pylint: disable=C0412
    import torch
    from art.estimators.classification.pytorch import PyTorchClassifier


logger = logging.getLogger(__name__)


class OverTheAirFlickeringPyTorch(EvasionAttack):
    """
    This module contains an implementation of the Over-the-Air Adversarial Flickering attack on video recognition
    networks.

    | Paper link: https://arxiv.org/abs/2002.05123
    """

    attack_params = EvasionAttack.attack_params + [
        "eps_step",
        "max_iter",
        "beta_0",
        "beta_1",
        "beta_2",
        "loss_margin",
        "batch_size",
        "start_frame_index",
        "num_frames",
        "round_samples",
        "targeted",
        "verbose",
    ]

    _estimator_requirements = (BaseEstimator, ClassifierMixin, LossGradientsMixin)

    def __init__(
        self,
        classifier: "PyTorchClassifier",
        eps_step: float = 0.01,
        max_iter: int = 30,
        beta_0: float = 1.0,
        beta_1: float = 0.5,
        beta_2: float = 0.5,
        loss_margin: float = 0.05,
        batch_size: int = 1,
        start_frame_index: int = 0,
        num_frames: Optional[int] = None,
        round_samples: float = 0.0,
        targeted: bool = False,
        verbose: bool = True,
    ):
        """
        Create an instance of the :class:`.OverTheAirFlickeringPyTorch`.

        :param classifier: A trained classifier.
        :param eps_step: The step size per iteration.
        :param max_iter: The number of iterations.
        :param beta_0: Weighting of the sum of all regularisation terms corresponding to `lambda` in the original paper.
        :param beta_1: Weighting of thickness regularisation.
        :param beta_2: Weighting of roughness regularisation.
        :param loss_margin: The loss margin.
        :param batch_size: Batch size.
        :param start_frame_index: The first frame to be perturbed.
        :param num_frames: The number of frames to be perturbed.
        :param round_samples: Granularity of the input values to be enforced if > 0.0.
        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False).
        :param verbose: Show progress bars.
        """
        super().__init__(estimator=classifier)

        self.eps_step = eps_step
        self.max_iter = max_iter
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.loss_margin = loss_margin
        self.batch_size = batch_size
        self.start_frame_index = start_frame_index
        self.num_frames = num_frames
        self.round_samples = round_samples
        self.end_frame_index = (
            self.start_frame_index + self.num_frames if self.num_frames is not None else self.num_frames
        )
        self._targeted = targeted
        self.verbose = verbose
        self._check_params()

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial examples.

        :param x: Original input samples representing videos of format NFHWC.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :return: Adversarial examples.
        """
        import torch  # lgtm [py/repeated-import]

        y = check_and_transform_label_format(y, self.estimator.nb_classes)
        if y is None:
            if self.targeted:  # pragma: no cover
                raise ValueError("Target labels `y` need to be provided for a targeted attack.")

            # Use model predictions as true labels
            logger.info("Using model predictions as true labels.")
            y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))

        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(x.astype(ART_NUMPY_DTYPE)),
            torch.from_numpy(y.astype(ART_NUMPY_DTYPE)),
        )

        data_loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=self.batch_size, shuffle=False, drop_last=False
        )

        x_adv = x.copy().astype(ART_NUMPY_DTYPE)

        # Compute perturbation with batching
        for (batch_id, batch_all) in enumerate(
            tqdm(data_loader, desc="OverTheAirFlickeringPyTorch - Batches", leave=False, disable=not self.verbose)
        ):
            (batch, batch_labels) = batch_all[0], batch_all[1]

            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            x_adv[batch_index_1:batch_index_2] = self._generate_batch(batch, batch_labels)

        return x_adv

    def _generate_batch(self, x: "torch.Tensor", y: "torch.Tensor") -> np.ndarray:
        """
        Generate a batch of adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)`.
        :return: Adversarial examples.
        """
        import torch  # lgtm [py/repeated-import]

        x = x.to(self.estimator.device)
        y = y.to(self.estimator.device)
        x_adv = torch.clone(x)

        for _ in range(self.max_iter):
            x_adv = self._compute_torch(
                x_adv,
                x,
                y,
                self.eps_step,
            )

        return x_adv.cpu().detach().numpy()

    def _compute_torch(
        self,
        x_adv: "torch.Tensor",
        x: "torch.Tensor",
        y: "torch.Tensor",
        eps_step: float,
    ) -> "torch.Tensor":
        """
        Compute adversarial examples for one iteration.

        :param x_adv: Current adversarial examples.
        :param x: An array with the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)`.
        :param eps_step: Attack step size (input variation) at each iteration.
        :return: Adversarial examples.
        """
        # Get perturbation
        perturbation = x_adv - x
        grad = self._compute_perturbation(x_adv, y, perturbation)

        # Apply perturbation and clip
        x_adv = self._apply_perturbation(x_adv, grad, eps_step)

        return x_adv

    def _compute_perturbation(
        self, x: "torch.Tensor", y: "torch.Tensor", perturbation: "torch.Tensor"
    ) -> "torch.Tensor":
        """
        Compute perturbation.

        :param x: Current adversarial examples.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)`.
        :param perturbation: Currently accumulated perturbation
        :return: Perturbations.
        """
        import torch  # lgtm [py/repeated-import]

        # Get gradient wrt loss
        grad = self._get_loss_gradients(x, y, perturbation)

        grad = torch.repeat_interleave(torch.repeat_interleave(grad, x.shape[2], dim=2), x.shape[3], dim=3)

        if self.start_frame_index is not None:
            full_grad = torch.zeros(x.shape, dtype=grad.dtype, device=grad.device)
            full_grad[:, self.start_frame_index : self.end_frame_index, :, :, :] = grad[
                :, self.start_frame_index : self.end_frame_index, :, :, :
            ]
            grad = full_grad

        return grad

    def _get_loss_gradients(self, x: "torch.Tensor", y: "torch.Tensor", perturbation: "torch.Tensor") -> "torch.Tensor":
        """
        Compute custom, framework-specific, regularized loss gradients.
        """
        import torch  # lgtm [py/repeated-import]

        softmax = torch.nn.Softmax(dim=1).to(self.estimator.device)

        grads_batch = []

        for i in range(x.shape[0]):
            eps = torch.autograd.Variable(
                torch.zeros((1, x.shape[1], 1, 1, x.shape[4]), device=self.estimator.device), requires_grad=True
            )
            x_in = x[[i]] + torch.repeat_interleave(torch.repeat_interleave(eps, x.shape[2], dim=2), x.shape[3], dim=3)
            x_in = self._clip_and_round_pytorch(x_in)
            preds, _ = self.estimator._predict_framework(x=x_in)  # pylint: disable=W0212

            # calculate adversarial loss
            y_preds = softmax(preds)[0]
            y_mask = y[i].eq(1)
            label_prob = torch.masked_select(y_preds, y_mask)
            max_non_label_prob = torch.max(y_preds - y[i], dim=0)[0]

            l_1 = torch.zeros(1).to(self.estimator.device)
            l_m = (label_prob - max_non_label_prob) * (1 - 2 * int(self.targeted)) + self.loss_margin
            l_2 = (l_m ** 2) / self.loss_margin
            l_3 = l_m

            adversarial_loss = torch.max(l_1, torch.min(l_2, l_3)[0])[0]

            # calculate regularization terms
            # thickness - loss term
            perturbation_i = perturbation[[i]] + eps
            norm_reg = torch.mean(perturbation_i ** 2) + 1e-12
            perturbation_roll_right = torch.roll(perturbation_i, 1, dims=1)
            perturbation_roll_left = torch.roll(perturbation_i, -1, dims=1)

            # 1st order diff - loss term
            diff_norm_reg = torch.mean((perturbation_i - perturbation_roll_right) ** 2) + 1e-12

            # 2nd order diff - loss term
            laplacian_norm_reg = (
                torch.mean((-2 * perturbation_i + perturbation_roll_right + perturbation_roll_left) ** 2) + 1e-12
            )

            regularization_loss = self.beta_0 * (
                self.beta_1 * norm_reg + self.beta_2 * diff_norm_reg + self.beta_2 * laplacian_norm_reg
            )

            loss = adversarial_loss + regularization_loss

            self.estimator.model.zero_grad()

            # Compute gradients
            loss.backward()
            grads = eps.grad
            grads_batch.append(grads[0, ...])

        grads_batch_tensor = torch.stack(grads_batch)

        return grads_batch_tensor

    def _clip_and_round_pytorch(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Rounds the input to the correct level of granularity. Useful to ensure data passed to classifier can be
        represented in the correct domain, e.g., [0, 255] integers verses [0,1] or [0, 255] floating points.

        :param x: Sample input with shape as expected by the model.
        :return: Clipped and rounded inputs.
        """
        import torch  # lgtm [py/repeated-import]

        if self.estimator.clip_values is not None:
            x = torch.clamp(x, self.estimator.clip_values[0], self.estimator.clip_values[1])

        if self.round_samples != 0.0:
            x = torch.round(x / self.round_samples) * self.round_samples

        return x

    def _apply_perturbation(self, x_adv: "torch.Tensor", grad: "torch.Tensor", eps_step: float) -> "torch.Tensor":
        """
        Apply perturbation on examples.

        :param x: Current adversarial examples.
        :param grad: Current gradients.
        :param eps_step: Attack step size (input variation) at each iteration.
        :return: Adversarial examples.
        """
        import torch  # lgtm [py/repeated-import]

        x_adv = x_adv - eps_step * torch.sign(grad)

        if self.estimator.clip_values is not None:
            clip_min, clip_max = self.estimator.clip_values
            x_adv = torch.max(
                torch.min(x_adv, torch.tensor(clip_max).to(self.estimator.device)),
                torch.tensor(clip_min).to(self.estimator.device),
            )

        return x_adv

    def _check_params(self) -> None:

        if not isinstance(self.eps_step, (int, float)) or self.eps_step <= 0.0:
            raise ValueError("The argument `eps_step` must be positive of type int or float.")

        if not isinstance(self.max_iter, int) or self.max_iter <= 0:
            raise ValueError("The argument `max_iter` must be positive of type int.")

        if not isinstance(self.beta_0, (int, float)) or self.beta_0 < 0.0:
            raise ValueError("The argument `beta_0` must be 0.0 or positive of type int or float.")

        if not isinstance(self.beta_1, (int, float)) or self.beta_1 < 0.0:
            raise ValueError("The argument `beta_1` must be 0.0 or positive of type int or float.")

        if not isinstance(self.beta_2, (int, float)) or self.beta_2 < 0.0:
            raise ValueError("The argument `beta_2` must be 0.0 or positive of type int or float.")

        if not isinstance(self.loss_margin, (int, float)) or self.loss_margin <= 0.0:
            raise ValueError("The argument `loss_margin` must be positive of type int or float.")

        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("The argument `batch_size` must be positive of type int.")

        if not isinstance(self.start_frame_index, int) or self.start_frame_index < 0:
            raise ValueError("The argument `start_frame_index` must be 0 or positive of type int.")

        if self.num_frames is not None and (not isinstance(self.num_frames, int) or self.num_frames <= 0):
            raise ValueError("The argument `num_frames` must be positive of type int.")

        if not isinstance(self.round_samples, (int, float)) or self.round_samples < 0.0:
            raise ValueError("The argument `round_samples` must be 0.0 or positive of type int or float.")
