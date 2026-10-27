# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2026
# Copyright (c) 2021 Maura Pintor
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Fast Minimum-Norm (FMN) attack with L2 and Linf constraints.

Adapted from the authors' MIT-licensed reference implementation:
https://github.com/pralab/Fast-Minimum-Norm-FMN-Attack

| Paper link: https://arxiv.org/abs/2102.12827
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from tqdm.auto import trange

from art.attacks.attack import EvasionAttack
from art.config import ART_NUMPY_DTYPE
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassGradientsMixin, ClassifierMixin
from art.utils import check_and_transform_label_format

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE


class FastMinimumNorm(EvasionAttack):
    """
    Fast Minimum-Norm attack of Pintor et al. (NeurIPS 2021).

    FMN adapts a per-sample perturbation radius while following the gradient of a logit margin. It returns the
    smallest successful perturbation encountered, or the original input if no successful perturbation was found.
    This implementation supports L2 and Linf, targeted and untargeted attacks, and starts at the original input.
    L0, L1, adversarial starting points and restarts are not implemented.

    The estimator must return logits from ``predict`` and the derivatives of those same logits from
    ``class_gradient``. A softmax output layer should be removed before wrapping a neural network.
    One logit per class is required, including two logits for a binary classifier; single-output binary
    classifiers are unsupported. Finite ``clip_values`` are required to bound the adaptive radius.

    Reference: Maura Pintor, Fabio Roli, Wieland Brendel, and Battista Biggio, "Fast Minimum-norm Adversarial
    Attacks through Adaptive Norm Constraints", Advances in Neural Information Processing Systems 34, 2021.

    | Paper link: https://arxiv.org/abs/2102.12827
    """

    attack_params = EvasionAttack.attack_params + [
        "norm",
        "max_iter",
        "max_step_size",
        "min_step_size",
        "gamma",
        "targeted",
        "batch_size",
        "verbose",
    ]
    _estimator_requirements = (BaseEstimator, ClassifierMixin, ClassGradientsMixin)

    def __init__(
        self,
        classifier: "CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE",
        norm: int | float | str = 2,
        max_iter: int = 100,
        max_step_size: float = 1.0,
        min_step_size: float | None = None,
        gamma: float = 0.05,
        targeted: bool = False,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> None:
        """
        Create an FMN attack.

        :param classifier: Classifier exposing logits and their class gradients.
        :param norm: Perturbation norm: 2, np.inf or "inf".
        :param max_iter: Number of margin evaluations and gradient updates per sample. As in the reference
                         implementation, the last update is not evaluated for inclusion in the returned result.
        :param max_step_size: Initial L2-normalised gradient step size, in input units.
        :param min_step_size: Lower endpoint of the cosine step-size schedule; the final executed step approaches
                              this value. None uses max_step_size / 100.
        :param gamma: Initial relative radius adjustment, annealed towards 0.001; in [0.001, 1].
        :param targeted: Whether to generate examples classified as the labels supplied to generate.
        :param batch_size: Internal batch size.
        :param verbose: Show progress bars.
        """
        super().__init__(estimator=classifier)
        self.norm = norm
        self.max_iter = max_iter
        self.max_step_size = max_step_size
        self.min_step_size = min_step_size
        self.gamma = gamma
        self._targeted = targeted
        self.batch_size = batch_size
        self.verbose = verbose
        self._check_params()

    def generate(self, x: np.ndarray, y: np.ndarray | None = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial examples without modifying the inputs.

        :param x: Original inputs within the estimator's clip values.
        :param y: Class indices or one-hot labels. For targeted attacks these are target labels and are required.
                  For untargeted attacks, None uses the model's predictions to avoid label leaking.
        :param kwargs: Unused additional arguments.
        :return: Smallest successful examples found, or original samples where the attack failed.
        """
        x_adv = x.astype(ART_NUMPY_DTYPE, copy=True)
        if x_adv.ndim < 2 or x_adv.shape[1:] != self.estimator.input_shape:
            raise ValueError("Input shape must match the estimator's input_shape.")
        clip_min, clip_max = self.estimator.clip_values
        if not np.all(np.isfinite(x_adv)) or np.any(x_adv < clip_min) or np.any(x_adv > clip_max):
            raise ValueError("Inputs must be finite and within the estimator's clip_values.")
        if self.targeted and y is None:
            raise ValueError("Target labels y are required for targeted attacks.")
        if y is not None:
            if not np.all(np.isfinite(y)) or (
                y.ndim == 2 and y.shape[1] > 1 and y.shape[1] != self.estimator.nb_classes
            ):
                raise ValueError("Labels must be finite and have one column per class when one-hot encoded.")
            y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes, return_one_hot=False)
            if y is None or y.size != len(x_adv):
                raise ValueError("Labels must contain one class per input.")
            if np.any(y != np.floor(y)):
                raise ValueError("Labels must be integer class indices.")
            if np.any(y < 0) or np.any(y >= self.estimator.nb_classes):
                raise ValueError("Labels must be valid class indices.")
            y = y.reshape(-1).astype(int)

        for start in trange(0, len(x_adv), self.batch_size, desc="FMN", disable=not self.verbose):
            batch = x_adv[start : start + self.batch_size].copy()
            labels = self._predict(batch).argmax(axis=1) if y is None else y[start : start + self.batch_size]
            x_adv[start : start + self.batch_size] = self._generate_batch(batch, labels)
        return x_adv

    def _predict(self, x: np.ndarray) -> np.ndarray:
        """Obtain one finite logit per class; single-output binary models are unsupported."""
        logits = self.estimator.predict(x, batch_size=self.batch_size)
        if logits.shape != (len(x), self.estimator.nb_classes) or not np.all(np.isfinite(logits)):
            raise ValueError("The estimator must return one finite logit per class and sample.")
        return logits

    def _generate_batch(self, x: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Run the adaptive-radius search for a batch."""
        norm, dual = (2, 2) if self.norm == 2 else (np.inf, 1)
        clip_min, clip_max = self.estimator.clip_values
        broadcast_shape = (-1,) + (1,) * (x.ndim - 1)
        rows = np.arange(len(x))
        delta = np.zeros_like(x)
        best_delta = delta.copy()
        worst_norm = np.linalg.norm(np.maximum(x - clip_min, clip_max - x).reshape(len(x), -1), ord=norm, axis=1)
        best_norm = worst_norm.copy()
        # The box radius is equivalent to an infinite initial radius after clipping and avoids inf * 0 at gamma=1.
        epsilon = worst_norm.copy()
        adv_found = np.zeros(len(x), dtype=bool)
        min_step = self.max_step_size / 100 if self.min_step_size is None else self.min_step_size

        for iteration in range(self.max_iter):
            annealing = float((1 + np.cos(np.pi * iteration / self.max_iter)) / 2)
            step_size = min_step + (self.max_step_size - min_step) * annealing
            gamma = 0.001 + (self.gamma - 0.001) * annealing
            current = x + delta
            logits = self._predict(current)
            other_logits = logits.copy()
            other_logits[rows, labels] = -np.inf
            other = other_logits.argmax(axis=1)
            # Ascend the negative margin: other - original, or target - other.
            add, subtract = (labels, other) if self.targeted else (other, labels)
            margin = logits[rows, subtract] - logits[rows, add]
            gradients = (
                self.estimator.class_gradient(current, label=add)[:, 0]
                - self.estimator.class_gradient(current, label=subtract)[:, 0]
            )
            if not np.all(np.isfinite(gradients)):
                raise ValueError("Class gradients must be finite.")
            predicted = logits.argmax(axis=1)
            is_adv = predicted == labels if self.targeted else predicted != labels
            current_norm = np.linalg.norm(delta.reshape(len(x), -1), ord=norm, axis=1)
            improved = is_adv & (current_norm <= best_norm)
            best_norm = np.where(improved, current_norm, best_norm)
            best_delta[improved] = delta[improved]
            adv_found |= is_adv

            flat_gradients = gradients.reshape(len(x), -1)
            dual_norm = np.linalg.norm(flat_gradients, ord=dual, axis=1)
            # A flat margin has no local boundary estimate. The finite box radius caps infinity.
            distance = np.divide(abs(margin), dual_norm, out=np.full_like(margin, np.inf), where=dual_norm > 0)
            # Inflate the local boundary estimate by a small float32 rounding guard. For a linear model, exact
            # projection onto the boundary can otherwise repeatedly land on the original side of a tied decision.
            boundary_radius = current_norm + distance
            boundary_radius += 8 * np.finfo(ART_NUMPY_DTYPE).eps * np.maximum(1, boundary_radius)
            epsilon = np.where(
                is_adv,
                np.minimum(epsilon * (1 - gamma), best_norm),
                np.where(adv_found, epsilon * (1 + gamma), boundary_radius),
            )
            epsilon = np.minimum(epsilon, worst_norm)
            grad_norm = np.linalg.norm(flat_gradients, axis=1).reshape(broadcast_shape)
            direction = np.divide(gradients, grad_norm, out=np.zeros_like(gradients), where=grad_norm > 0)
            delta += step_size * direction

            if norm == 2:
                delta_norm = np.linalg.norm(delta.reshape(len(x), -1), axis=1)
                scale = np.minimum(1, np.divide(epsilon, delta_norm, out=np.ones_like(epsilon), where=delta_norm > 0))
                delta *= scale.reshape(broadcast_shape)
            else:
                radius = epsilon.reshape(broadcast_shape)
                delta = np.clip(delta, -radius, radius)
            delta = (np.clip(x + delta, clip_min, clip_max) - x).astype(ART_NUMPY_DTYPE)

        return np.clip(x + best_delta, clip_min, clip_max).astype(ART_NUMPY_DTYPE)

    def _check_params(self) -> None:
        """Validate public parameters and the finite input domain."""
        if self.norm not in (2, np.inf, "inf"):
            raise ValueError("The norm must be 2, np.inf or 'inf'.")
        for name in ("max_iter", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        for name in ("max_step_size", "gamma"):
            scalar = getattr(self, name)
            if (
                isinstance(scalar, bool)
                or not isinstance(scalar, (int, float))
                or not np.isfinite(scalar)
                or scalar <= 0
            ):
                raise ValueError(f"{name} must be a finite positive number.")
        if not 0.001 <= self.gamma <= 1:
            raise ValueError("gamma must be in [0.001, 1].")
        if self.min_step_size is not None and (
            isinstance(self.min_step_size, bool)
            or not isinstance(self.min_step_size, (int, float))
            or not np.isfinite(self.min_step_size)
            or not 0 <= self.min_step_size <= self.max_step_size
        ):
            raise ValueError("min_step_size must be finite and between zero and max_step_size.")
        if not isinstance(self.targeted, bool) or not isinstance(self.verbose, bool):
            raise ValueError("targeted and verbose must be bool.")
        if self.estimator.clip_values is None or not np.all(np.isfinite(self.estimator.clip_values)):
            raise ValueError("FMN requires finite estimator clip_values.")
