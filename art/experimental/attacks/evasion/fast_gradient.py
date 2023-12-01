# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2023
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
This module contains an experimental FGSM attack for multimodal models.
"""
import copy
from collections import UserDict
from typing import Optional, Union, TYPE_CHECKING

import numpy as np

from art.attacks.evasion.fast_gradient import FastGradientMethod
from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.experimental.estimators.hugging_face_multimodal import HuggingFaceMultiModalInput

from art.summary_writer import SummaryWriter
from art.config import ART_NUMPY_DTYPE

from art.utils import random_sphere, projection_l1_1, projection_l1_2

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_LOSS_GRADIENTS_TYPE


def multimodal_projection(
    values: np.ndarray, eps: Union[int, float, np.ndarray], norm_p: Union[int, float, str]
) -> np.ndarray:
    """
    Experimental extension of the projection in art.utils to support multimodal inputs.

    Project `values` on the L_p norm ball of size `eps`.

    :param values: Array of perturbations to clip.
    :param eps: Maximum norm allowed.
    :param norm_p: L_p norm to use for clipping.
            Only 1, 2 , `np.Inf` 1.1 and 1.2 supported for now.
            1.1 and 1.2 compute orthogonal projections on l1-ball, using two different algorithms
    :return: Values of `values` after projection.
    """
    # Pick a small scalar to avoid division by 0
    tol = 10e-8
    values_tmp = values.reshape((values.shape[0], -1))

    if norm_p == 2:
        if isinstance(eps, np.ndarray):
            raise NotImplementedError("The parameter `eps` of type `np.ndarray` is not supported to use with norm 2.")

        values_tmp = values_tmp * np.expand_dims(
            np.minimum(1.0, eps / (np.linalg.norm(values_tmp, axis=1) + tol)), axis=1
        )

    elif norm_p == 1:
        if isinstance(eps, np.ndarray):
            raise NotImplementedError("The parameter `eps` of type `np.ndarray` is not supported to use with norm 1.")

        values_tmp = values_tmp * np.expand_dims(
            np.minimum(1.0, eps / (np.linalg.norm(values_tmp, axis=1, ord=1) + tol)),
            axis=1,
        )
    elif norm_p == 1.1:
        values_tmp = projection_l1_1(values_tmp, eps)
    elif norm_p == 1.2:
        values_tmp = projection_l1_2(values_tmp, eps)

    elif norm_p in [np.inf, "inf"]:
        if isinstance(eps, np.ndarray):
            if isinstance(values_tmp, UserDict):
                eps = eps * np.ones_like(values["pixel_values"].cpu().detach().numpy())
            else:
                eps = eps * np.ones_like(values)
            eps = eps.reshape([eps.shape[0], -1])  # type: ignore

        if isinstance(values_tmp, UserDict):
            sign = np.sign(values_tmp["pixel_values"].cpu().detach().numpy())
            mag = abs(values_tmp["pixel_values"].cpu().detach().numpy())
            values_tmp["pixel_values"] = sign * np.minimum(mag, eps)
        else:
            values_tmp = np.sign(values_tmp) * np.minimum(abs(values_tmp), eps)

    else:
        raise NotImplementedError(
            'Values of `norm_p` different from 1, 2, `np.inf` and "inf" are currently not ' "supported."
        )

    values = values_tmp.reshape(values.shape)

    return values


class FastGradientMethodCLIP(FastGradientMethod):
    """
    Implementation of the FGSM attack operating on the image portion of multimodal inputs
    to the CLIP model.
    """

    attack_params = EvasionAttack.attack_params + [
        "norm",
        "eps",
        "eps_step",
        "targeted",
        "num_random_init",
        "batch_size",
        "minimal",
        "summary_writer",
    ]
    _estimator_requirements = (BaseEstimator, LossGradientsMixin)

    def __init__(
        self,
        estimator: "CLASSIFIER_LOSS_GRADIENTS_TYPE",
        norm: Union[int, float, str] = np.inf,
        eps: Union[int, float, np.ndarray] = 0.3,
        eps_step: Union[int, float, np.ndarray] = 0.1,
        targeted: bool = False,
        num_random_init: int = 0,
        batch_size: int = 32,
        minimal: bool = False,
        summary_writer: Union[str, bool, SummaryWriter] = False,
    ) -> None:

        super().__init__(
            estimator=estimator,
            norm=norm,
            eps=eps,
            eps_step=eps_step,
            targeted=targeted,
            num_random_init=num_random_init,
            batch_size=batch_size,
            minimal=minimal,
            summary_writer=summary_writer,
        )

    def _minimal_perturbation(self, x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Iteratively compute the minimal perturbation necessary to make the class prediction change. Stop when the
        first adversarial example was found.

        :param x: An array with the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes).
        :return: An array holding the adversarial examples.
        """
        partial_stop_condition: Union[bool, np.ndarray, np.bool_]
        current_eps: Union[int, float, np.ndarray]

        adv_x = copy.deepcopy(x)
        # Compute perturbation with implicit batching
        for batch_id in range(int(np.ceil(adv_x.shape[0] / float(self.batch_size)))):
            batch_index_1, batch_index_2 = (
                batch_id * self.batch_size,
                (batch_id + 1) * self.batch_size,
            )
            batch_labels = y[batch_index_1:batch_index_2]

            mask_batch = mask
            if mask is not None:
                # Here we need to make a distinction: if the masks are different for each input, we need to index
                # those for the current batch. Otherwise (i.e. mask is meant to be broadcasted), keep it as it is.
                if len(mask.shape) == len(x.shape):
                    mask_batch = mask[batch_index_1:batch_index_2]

            # Get perturbation
            perturbation = self._compute_perturbation(adv_x[batch_index_1:batch_index_2], batch_labels, mask_batch)

            # Get current predictions
            active_indices = np.arange(len(adv_x[batch_index_1:batch_index_2])) + batch_index_1

            if isinstance(self.eps, np.ndarray) and isinstance(self.eps_step, np.ndarray):
                if len(self.eps.shape) == len(x.shape) and self.eps.shape[0] == x.shape[0]:
                    current_eps = self.eps_step[batch_index_1:batch_index_2]
                    partial_stop_condition = (current_eps <= self.eps[batch_index_1:batch_index_2]).all()

                else:
                    current_eps = self.eps_step
                    partial_stop_condition = (current_eps <= self.eps).all()

            else:
                current_eps = self.eps_step
                partial_stop_condition = current_eps <= self.eps

            while active_indices.size > 0 and partial_stop_condition:
                # Adversarial crafting
                current_x = self._apply_perturbation(x[batch_index_1:batch_index_2], perturbation, current_eps)

                # Update
                adv_x[active_indices] = current_x[active_indices]

                adv_preds = self.estimator.predict(adv_x[batch_index_1:batch_index_2])
                # If targeted active check to see whether we have hit the target, otherwise head to anything but
                if self.targeted:
                    active_indices = (
                        np.where(np.argmax(batch_labels, axis=1) != np.argmax(adv_preds, axis=1))[0] + batch_index_1
                    )
                else:
                    active_indices = (
                        np.where(np.argmax(batch_labels, axis=1) == np.argmax(adv_preds, axis=1))[0] + batch_index_1
                    )

                # Update current eps and check the stop condition
                if isinstance(self.eps, np.ndarray) and isinstance(self.eps_step, np.ndarray):
                    if len(self.eps.shape) == len(x.shape) and self.eps.shape[0] == x.shape[0]:
                        current_eps = current_eps + self.eps_step[batch_index_1:batch_index_2]
                        partial_stop_condition = (current_eps <= self.eps[batch_index_1:batch_index_2]).all()

                    else:
                        current_eps = current_eps + self.eps_step
                        partial_stop_condition = (current_eps <= self.eps).all()

                else:
                    current_eps = current_eps + self.eps_step
                    partial_stop_condition = current_eps <= self.eps

        return adv_x

    def _apply_perturbation(
        self, x: np.ndarray, perturbation: np.ndarray, eps_step: Union[int, float, np.ndarray]
    ) -> np.ndarray:

        perturbation_step = eps_step * perturbation
        if perturbation_step.dtype != object:
            perturbation_step[np.isnan(perturbation_step)] = 0
        else:
            for i, _ in enumerate(perturbation_step):
                perturbation_step_i_array = perturbation_step[i].astype(np.float32)
                if np.isnan(perturbation_step_i_array).any():
                    perturbation_step[i] = np.where(
                        np.isnan(perturbation_step_i_array), 0.0, perturbation_step_i_array
                    ).astype(object)

        x = x + perturbation_step
        if self.estimator.clip_values is not None:
            clip_min, clip_max = self.estimator.clip_values
            if x.dtype == object:
                if isinstance(x, HuggingFaceMultiModalInput):
                    for i_obj in range(x.shape[0]):
                        x[i_obj] = np.clip(x[i_obj]["pixel_values"].cpu().detach().numpy(), clip_min, clip_max)
                else:
                    for i_obj in range(x.shape[0]):
                        x[i_obj] = np.clip(x[i_obj], clip_min, clip_max)
            else:
                x = np.clip(x, clip_min, clip_max)

        return x

    def _compute(
        self,
        x: np.ndarray,
        x_init: np.ndarray,
        y: np.ndarray,
        mask: Optional[np.ndarray],
        eps: Union[int, float, np.ndarray],
        eps_step: Union[int, float, np.ndarray],
        project: bool,
        random_init: bool,
        batch_id_ext: Optional[int] = None,
        decay: Optional[float] = None,
        momentum: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        import torch
        batch_eps: Union[int, float, np.ndarray]
        batch_eps_step: Union[int, float, np.ndarray]
        original_type = x['pixel_values'].dtype

        if random_init:
            n = x.shape[0]
            m = np.prod(x.shape[1:]).item()
            random_perturbation = random_sphere(n, m, eps, self.norm).reshape(x.shape).astype(ART_NUMPY_DTYPE)
            if mask is not None:
                random_perturbation = random_perturbation * (mask.astype(ART_NUMPY_DTYPE))
            x_adv = x.astype(ART_NUMPY_DTYPE) + random_perturbation

            if self.estimator.clip_values is not None:
                clip_min, clip_max = self.estimator.clip_values
                x_adv = np.clip(x_adv, clip_min, clip_max)
        else:
            if x.dtype == object:
                x_adv = copy.deepcopy(x)
            else:
                x_adv = x.astype(ART_NUMPY_DTYPE)

        # Compute perturbation with implicit batching
        x_adv_result = []
        for batch_id in range(int(np.ceil(x.shape[0] / float(self.batch_size)))):
            if batch_id_ext is None:
                self._batch_id = batch_id
            else:
                self._batch_id = batch_id_ext
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            batch_index_2 = min(batch_index_2, x.shape[0])
            batch_labels = y[batch_index_1:batch_index_2]

            mask_batch = mask
            if mask is not None:
                # Here we need to make a distinction: if the masks are different for each input, we need to index
                # those for the current batch. Otherwise (i.e. mask is meant to be broadcasted), keep it as it is.
                if len(mask.shape) == len(x.shape):
                    mask_batch = mask[batch_index_1:batch_index_2]

            # Get perturbation
            perturbation = self._compute_perturbation(
                x_adv[batch_index_1:batch_index_2], batch_labels, mask_batch, decay, momentum
            )

            # Compute batch_eps and batch_eps_step
            if isinstance(eps, np.ndarray) and isinstance(eps_step, np.ndarray):
                if len(eps.shape) == len(x.shape) and eps.shape[0] == x.shape[0]:
                    batch_eps = eps[batch_index_1:batch_index_2]
                    batch_eps_step = eps_step[batch_index_1:batch_index_2]

                else:
                    batch_eps = eps
                    batch_eps_step = eps_step

            else:
                batch_eps = eps
                batch_eps_step = eps_step

            # Apply perturbation and clip
            x_adv_batch = self._apply_perturbation(
                x_adv[batch_index_1:batch_index_2], perturbation, batch_eps_step
            )

            if project:
                if x_adv.dtype == object:
                    for i_sample in range(batch_index_1, batch_index_2):
                        if isinstance(batch_eps, np.ndarray) and batch_eps.shape[0] == x_adv.shape[0]:
                            perturbation = multimodal_projection(
                                x_adv_batch[i_sample - batch_index_1] - x_init[i_sample], batch_eps[i_sample], self.norm
                            )

                        else:
                            perturbation = multimodal_projection(
                                x_adv_batch[i_sample - batch_index_1] - x_init[i_sample], batch_eps, self.norm
                            )

                        x_adv_batch[i_sample - batch_index_1] = x_init[i_sample] + perturbation

                else:
                    perturbation = multimodal_projection(
                        x_adv_batch - x_init[batch_index_1:batch_index_2], batch_eps, self.norm
                    )
                    x_adv_batch = x_init[batch_index_1:batch_index_2] + perturbation
            x_adv_result.append(x_adv_batch['pixel_values'])

        x_adv_result = torch.concatenate(x_adv_result)
        return x_adv.update_pixels(x_adv_result)
