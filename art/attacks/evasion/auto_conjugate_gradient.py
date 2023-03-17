# MIT License

# Copyright (c) 2022 Keiichiro Yamamura

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
This module implements the 'Auto Conjugate Gradient' attack.

| Paper link: https://arxiv.org/abs/2206.09628
"""
import abc
import logging
import math
from typing import Optional, Union, TYPE_CHECKING

import numpy as np
from tqdm.auto import trange

from art.config import ART_NUMPY_DTYPE
from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import check_and_transform_label_format, projection, random_sphere, is_probability, get_labels_np_array

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_LOSS_GRADIENTS_TYPE

logger = logging.getLogger(__name__)


class AutoConjugateGradient(EvasionAttack):
    """
    Implementation of the 'Auto Conjugate Gradient' attack.
    The original implementation is https://github.com/yamamura-k/ACG.

    | Paper link: https://arxiv.org/abs/2206.09628
    """

    attack_params = EvasionAttack.attack_params + [
        "norm",
        "eps",
        "eps_step",
        "max_iter",
        "targeted",
        "nb_random_init",
        "batch_size",
        "loss_type",
        "verbose",
    ]
    _estimator_requirements = (BaseEstimator, LossGradientsMixin, ClassifierMixin)
    _predefined_losses = [None, "cross_entropy", "difference_logits_ratio"]

    def __init__(
        self,
        estimator: "CLASSIFIER_LOSS_GRADIENTS_TYPE",
        norm: Union[int, float, str] = np.inf,
        eps: float = 0.3,
        eps_step: float = 0.1,
        max_iter: int = 100,
        targeted: bool = False,
        nb_random_init: int = 5,
        batch_size: int = 32,
        loss_type: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        Create a :class:`.AutoConjugateGradient` instance.

        :param estimator: An trained estimator.
        :param norm: The norm of the adversarial perturbation. Possible values: "inf", np.inf, 1 or 2.
        :param eps: Maximum perturbation that the attacker can introduce.
        :param eps_step: Attack step size (input variation) at each iteration.
        :param max_iter: The maximum number of iterations.
        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False).
        :param nb_random_init: Number of random initialisations within the epsilon ball. For num_random_init=0
            starting at the original input.
        :param batch_size: Size of the batch on which adversarial samples are generated.
        :param loss_type: Defines the loss to attack. Available options: None (Use loss defined by estimator),
            "cross_entropy", or "difference_logits_ratio"
        :param verbose: Show progress bars.
        """
        from art.estimators.classification import TensorFlowClassifier, TensorFlowV2Classifier, PyTorchClassifier

        if isinstance(estimator, TensorFlowClassifier):
            raise ValueError("This attack does not support TensorFlow  v1.")

        if loss_type not in self._predefined_losses:
            raise ValueError(
                f"The argument loss_type has an invalid value. The following options for `loss_type` are currently "
                f"supported: {self._predefined_losses}"
            )

        if loss_type is None:
            if hasattr(estimator, "predict") and is_probability(
                estimator.predict(x=np.ones(shape=(1, *estimator.input_shape), dtype=np.float32))
            ):
                raise ValueError(  # pragma: no cover
                    "AutoProjectedGradientDescent is expecting logits as estimator output, the provided "
                    "estimator seems to predict probabilities."
                )

            estimator_acg = estimator
        else:
            if isinstance(estimator, TensorFlowV2Classifier):
                import tensorflow as tf

                class TensorFlowV2Loss:
                    """abstract class of loss function of tensorflow v2"""

                    @abc.abstractmethod
                    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor, *args, **kwargs) -> tf.Tensor:
                        raise NotImplementedError

                if loss_type == "cross_entropy":

                    class CrossEntropyLossV2(TensorFlowV2Loss):
                        """Class defining cross entropy loss with reduction options."""

                        def __init__(self, from_logits, reduction="sum"):
                            self.ce_loss = tf.keras.losses.CategoricalCrossentropy(
                                from_logits=from_logits,
                                reduction=tf.keras.losses.Reduction.NONE,
                            )
                            self.reduction = reduction

                        def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor, *args, **kwargs) -> tf.Tensor:
                            if self.reduction == "mean":
                                return tf.reduce_mean(self.ce_loss(y_true, y_pred))
                            if self.reduction == "sum":
                                return tf.reduce_sum(self.ce_loss(y_true, y_pred))
                            if self.reduction == "none":
                                return self.ce_loss(y_true, y_pred)
                            raise NotImplementedError()

                    if is_probability(estimator.predict(x=np.ones(shape=(1, *estimator.input_shape)))):
                        _loss_object_tf: TensorFlowV2Loss = CrossEntropyLossV2(from_logits=False)
                    else:
                        _loss_object_tf = CrossEntropyLossV2(from_logits=True)
                elif loss_type == "difference_logits_ratio":
                    if is_probability(estimator.predict(x=np.ones(shape=(1, *estimator.input_shape)))):
                        raise ValueError(  # pragma: no cover
                            "The provided estimator seems to predict probabilities. "
                            "If loss_type='difference_logits_ratio' the estimator has to to predict logits."
                        )

                    class DifferenceLogitsRatioTensorFlowV2(TensorFlowV2Loss):
                        """
                        Callable class for Difference Logits Ratio loss in TensorFlow v2.
                        """

                        def __init__(self):
                            self.reduction = "sum"

                        def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor, *args, **kwargs) -> tf.Tensor:
                            i_y_true = tf.cast(tf.math.argmax(tf.cast(y_true, tf.int32), axis=1), tf.int32)
                            i_y_pred_arg = tf.argsort(y_pred, axis=1)
                            i_z_i_list = []

                            for i in range(y_true.shape[0]):
                                if i_y_pred_arg[i, -1] != i_y_true[i]:
                                    i_z_i_list.append(i_y_pred_arg[i, -1])
                                else:
                                    i_z_i_list.append(i_y_pred_arg[i, -2])

                            i_z_i = tf.stack(i_z_i_list)

                            z_1 = tf.gather(y_pred, i_y_pred_arg[:, -1], axis=1, batch_dims=0)
                            z_3 = tf.gather(y_pred, i_y_pred_arg[:, -3], axis=1, batch_dims=0)
                            z_i = tf.gather(y_pred, i_z_i, axis=1, batch_dims=0)
                            z_y = tf.gather(y_pred, i_y_true, axis=1, batch_dims=0)

                            z_1 = tf.linalg.diag_part(z_1)
                            z_3 = tf.linalg.diag_part(z_3)
                            z_i = tf.linalg.diag_part(z_i)
                            z_y = tf.linalg.diag_part(z_y)

                            dlr = -(z_y - z_i) / (z_1 - z_3)
                            if self.reduction == "mean":
                                return tf.reduce_mean(dlr)
                            if self.reduction == "sum":
                                return tf.reduce_sum(dlr)
                            if self.reduction == "none":
                                return dlr
                            raise NotImplementedError()

                    _loss_object_tf = DifferenceLogitsRatioTensorFlowV2()

                estimator_acg = TensorFlowV2Classifier(
                    model=estimator.model,
                    nb_classes=estimator.nb_classes,
                    input_shape=estimator.input_shape,
                    loss_object=_loss_object_tf,
                    train_step=estimator._train_step,
                    channels_first=estimator.channels_first,
                    clip_values=estimator.clip_values,
                    preprocessing_defences=estimator.preprocessing_defences,
                    postprocessing_defences=estimator.postprocessing_defences,
                    preprocessing=estimator.preprocessing,
                )
            elif isinstance(estimator, PyTorchClassifier):
                import torch

                if loss_type == "cross_entropy":
                    if is_probability(
                        estimator.predict(x=np.ones(shape=(1, *estimator.input_shape), dtype=np.float32))
                    ):
                        raise ValueError(  # pragma: no cover
                            "The provided estimator seems to predict probabilities. If loss_type='cross_entropy' "
                            "the estimator has to to predict logits."
                        )

                    class CrossEntropyLossTorch(torch.nn.modules.loss._Loss):  # pylint: disable=W0212
                        """Class defining cross entropy loss with reduction options."""

                        def __init__(self, reduction="sum"):
                            super().__init__()
                            self.ce_loss = torch.nn.CrossEntropyLoss(reduction="none")
                            self.reduction = reduction

                        def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor, *args, **kwargs) -> torch.Tensor:
                            if self.reduction == "mean":
                                return self.ce_loss(y_true, y_pred).mean()
                            if self.reduction == "sum":
                                return self.ce_loss(y_true, y_pred).sum()
                            if self.reduction == "none":
                                return self.ce_loss(y_true, y_pred)
                            raise NotImplementedError()

                        def forward(
                            self, input: torch.Tensor, target: torch.Tensor  # pylint: disable=W0622
                        ) -> torch.Tensor:
                            """
                            Forward method.
                            :param input: Predicted labels of shape (nb_samples, nb_classes).
                            :param target: Target labels of shape (nb_samples, nb_classes).
                            :return: Difference Logits Ratio Loss.
                            """
                            return self.__call__(y_true=target, y_pred=input)

                    _loss_object_pt: torch.nn.modules.loss._Loss = CrossEntropyLossTorch(reduction="mean")

                elif loss_type == "difference_logits_ratio":
                    if is_probability(
                        estimator.predict(x=np.ones(shape=(1, *estimator.input_shape), dtype=ART_NUMPY_DTYPE))
                    ):
                        raise ValueError(  # pragma: no cover
                            "The provided estimator seems to predict probabilities. "
                            "If loss_type='difference_logits_ratio' the estimator has to to predict logits."
                        )

                    class DifferenceLogitsRatioPyTorch(torch.nn.modules.loss._Loss):  # pylint: disable=W0212
                        """
                        Callable class for Difference Logits Ratio loss in PyTorch.
                        """

                        def __init__(self):
                            super().__init__()
                            self.reduction = "sum"

                        def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
                            if isinstance(y_true, np.ndarray):
                                y_true = torch.from_numpy(y_true)
                            if isinstance(y_pred, np.ndarray):
                                y_pred = torch.from_numpy(y_pred)

                            y_true = y_true.float()

                            i_y_true = torch.argmax(y_true, dim=1)
                            i_y_pred_arg = torch.argsort(y_pred, dim=1)
                            i_z_i_list = []

                            for i in range(y_true.shape[0]):
                                if i_y_pred_arg[i, -1] != i_y_true[i]:
                                    i_z_i_list.append(i_y_pred_arg[i, -1])
                                else:
                                    i_z_i_list.append(i_y_pred_arg[i, -2])

                            i_z_i = torch.stack(i_z_i_list)

                            z_1 = y_pred[:, i_y_pred_arg[:, -1]]
                            z_3 = y_pred[:, i_y_pred_arg[:, -3]]
                            z_i = y_pred[:, i_z_i]
                            z_y = y_pred[:, i_y_true]

                            z_1 = torch.diagonal(z_1)
                            z_3 = torch.diagonal(z_3)
                            z_i = torch.diagonal(z_i)
                            z_y = torch.diagonal(z_y)

                            dlr = (-(z_y - z_i) / (z_1 - z_3)).float()
                            if self.reduction == "mean":
                                return dlr.mean()
                            if self.reduction == "sum":
                                return dlr.sum()
                            if self.reduction == "none":
                                return dlr
                            raise NotImplementedError()

                        def forward(
                            self, input: torch.Tensor, target: torch.Tensor  # pylint: disable=W0622
                        ) -> torch.Tensor:
                            """
                            Forward method.
                            :param input: Predicted labels of shape (nb_samples, nb_classes).
                            :param target: Target labels of shape (nb_samples, nb_classes).
                            :return: Difference Logits Ratio Loss.
                            """
                            return self.__call__(y_true=target, y_pred=input)

                    _loss_object_pt = DifferenceLogitsRatioPyTorch()
                else:
                    raise NotImplementedError()

                estimator_acg = PyTorchClassifier(
                    model=estimator.model,
                    loss=_loss_object_pt,
                    input_shape=estimator.input_shape,
                    nb_classes=estimator.nb_classes,
                    optimizer=None,
                    channels_first=estimator.channels_first,
                    clip_values=estimator.clip_values,
                    preprocessing_defences=estimator.preprocessing_defences,
                    postprocessing_defences=estimator.postprocessing_defences,
                    preprocessing=estimator.preprocessing,
                    device_type=str(estimator._device),
                )

            else:  # pragma: no cover
                raise ValueError(f"The loss type {loss_type} is not supported for the provided estimator.")

        super().__init__(estimator=estimator_acg)
        self.norm = norm
        self.eps = eps
        self.eps_step = eps_step
        self.max_iter = max_iter
        self.targeted = targeted
        self.nb_random_init = nb_random_init
        self.batch_size = batch_size
        self.loss_type = loss_type
        self.verbose = verbose
        self._check_params()

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
        mask = kwargs.get("mask")

        if y is not None:
            y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes)

        if y is None:
            if self.targeted:
                raise ValueError("Target labels `y` need to be provided for a targeted attack.")
            y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size)).astype(int)

        if self.estimator.nb_classes == 2 and y.shape[1] == 1:
            raise ValueError(
                "This attack has not yet been tested for binary classification with a single output classifier."
            )

        x_adv = x.astype(ART_NUMPY_DTYPE)

        for _ in trange(max(1, self.nb_random_init), desc="ACG - restart", disable=not self.verbose):
            # Determine correctly predicted samples
            y_pred = self.estimator.predict(x_adv)
            if self.targeted:
                sample_is_robust = np.argmax(y_pred, axis=1) != np.argmax(y, axis=1)
            elif not self.targeted:
                sample_is_robust = np.argmax(y_pred, axis=1) == np.argmax(y, axis=1)

            if np.sum(sample_is_robust) == 0:
                break

            x_robust = x_adv[sample_is_robust]
            y_robust = y[sample_is_robust]
            x_init = x[sample_is_robust]

            n = x_robust.shape[0]
            m = np.prod(x_robust.shape[1:]).item()
            random_perturbation = (
                random_sphere(n, m, self.eps, self.norm).reshape(x_robust.shape).astype(ART_NUMPY_DTYPE)
            )

            x_robust = x_robust + random_perturbation

            if self.estimator.clip_values is not None:
                clip_min, clip_max = self.estimator.clip_values
                x_robust = np.clip(x_robust, clip_min, clip_max)

            perturbation = projection(x_robust - x_init, self.eps, self.norm)
            x_robust = x_init + perturbation

            # Compute perturbation with implicit batching
            for batch_id in trange(
                int(np.ceil(x_robust.shape[0] / float(self.batch_size))),
                desc="ACG - batch",
                leave=False,
                disable=not self.verbose,
            ):
                batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
                x_k = x_robust[batch_index_1:batch_index_2].astype(ART_NUMPY_DTYPE)
                x_init_batch = x_init[batch_index_1:batch_index_2].astype(ART_NUMPY_DTYPE)
                y_batch = y_robust[batch_index_1:batch_index_2]

                p_0 = 0
                p_1 = 0.22
                var_w = [p_0, p_1]

                while True:
                    p_j_p_1 = var_w[-1] + max(var_w[-1] - var_w[-2] - 0.03, 0.06)
                    if p_j_p_1 > 1:
                        break
                    var_w.append(p_j_p_1)

                var_w = [math.ceil(p * self.max_iter) for p in var_w]

                # self.eta = np.full((self.batch_size, 1, 1, 1), 2 * self.eps_step).astype(ART_NUMPY_DTYPE)
                _batch_size = x_k.shape[0]
                eta = np.full((_batch_size, 1, 1, 1), self.eps_step).astype(ART_NUMPY_DTYPE)
                self.count_condition_1 = np.zeros(shape=(_batch_size,))
                gradk_1 = np.zeros_like(x_k)
                cgradk_1 = np.zeros_like(x_k)
                cgradk = np.zeros_like(x_k)
                gradk_1_best = np.zeros_like(x_k)
                cgradk_1_best = np.zeros_like(x_k)
                gradk_1_tmp = np.zeros_like(x_k)
                cgradk_1_tmp = np.zeros_like(x_k)

                for k_iter in trange(self.max_iter, desc="ACG - iteration", leave=False, disable=not self.verbose):

                    # Get perturbation, use small scalar to avoid division by 0
                    tol = 10e-8

                    # Get gradient wrt loss; invert it if attack is targeted
                    grad = self.estimator.loss_gradient(x_k, y_batch) * (1 - 2 * int(self.targeted))
                    if k_iter == 0:
                        gradk_1 = grad.copy()
                        cgradk_1 = grad.copy()
                        cgradk = grad.copy()
                    else:
                        beta = get_beta(grad, gradk_1, cgradk_1)
                        cgradk = grad + beta * cgradk_1

                    # Apply norm bound
                    if self.norm in [np.inf, "inf"]:
                        grad = np.sign(cgradk)
                    elif self.norm == 1:
                        ind = tuple(range(1, len(x_k.shape)))
                        cgradk = cgradk / (np.sum(np.abs(cgradk), axis=ind, keepdims=True) + tol)
                    elif self.norm == 2:
                        ind = tuple(range(1, len(x_k.shape)))
                        cgradk = cgradk / (np.sqrt(np.sum(np.square(cgradk), axis=ind, keepdims=True)) + tol)
                    assert x_k.shape == cgradk.shape

                    perturbation = cgradk

                    if mask is not None:
                        perturbation = perturbation * (mask.astype(ART_NUMPY_DTYPE))

                    # Apply perturbation and clip
                    x_k_p_1 = x_k + eta * perturbation

                    if self.estimator.clip_values is not None:
                        clip_min, clip_max = self.estimator.clip_values
                        x_k_p_1 = np.clip(x_k_p_1, clip_min, clip_max)

                    if k_iter == 0:
                        x_1 = x_k_p_1
                        perturbation = projection(x_1 - x_init_batch, self.eps, self.norm)
                        x_1 = x_init_batch + perturbation

                        f_0 = self.estimator.compute_loss(x=x_k, y=y_batch, reduction="none")
                        f_1 = self.estimator.compute_loss(x=x_1, y=y_batch, reduction="none")

                        self.eta_w_j_m_1 = eta.copy()
                        self.f_max_w_j_m_1 = f_0.copy()

                        self.f_max = f_0.copy()
                        self.x_max = x_k.copy()

                        f1_ge_f0 = f_1 >= f_0
                        f_1_tmp = f_1[f1_ge_f0].copy()
                        self.f_max[f1_ge_f0] = f_1_tmp.copy()
                        x_1_tmp = x_1[f1_ge_f0].copy()
                        self.x_max[f1_ge_f0] = x_1_tmp.copy()
                        self.count_condition_1[f1_ge_f0] += 1

                        # Settings for next iteration k
                        x_k = x_1
                        gradk_1_best = gradk_1.copy()
                        cgradk_1_best = cgradk_1.copy()

                    else:
                        perturbation = projection(x_k_p_1 - x_init_batch, self.eps, self.norm)
                        x_k_p_1 = x_init_batch + perturbation

                        if self.estimator.clip_values is not None:
                            clip_min, clip_max = self.estimator.clip_values
                            x_k_p_1 = np.clip(x_k_p_1, clip_min, clip_max)

                        perturbation = projection(x_k_p_1 - x_init_batch, self.eps, self.norm)
                        x_k_p_1 = x_init_batch + perturbation

                        f_k_p_1 = self.estimator.compute_loss(x=x_k_p_1, y=y_batch, reduction="none")

                        if (f_k_p_1 == 0.0).all():
                            x_k = x_k_p_1.copy()
                            break

                        if self.targeted:
                            fk_ge_fm = f_k_p_1 < self.f_max  # assume the loss function is cross-entropy
                        else:
                            fk_ge_fm = f_k_p_1 > self.f_max

                        self.count_condition_1[fk_ge_fm] += 1
                        # update the best points
                        x_k_p_1_tmp = x_k_p_1[fk_ge_fm].copy()
                        self.x_max[fk_ge_fm] = x_k_p_1_tmp.copy()
                        f_k_p_1_tmp = f_k_p_1[fk_ge_fm].copy()
                        self.f_max[fk_ge_fm] = f_k_p_1_tmp.copy()
                        gradk_1_tmp = gradk_1[fk_ge_fm].copy()
                        gradk_1_best[fk_ge_fm] = gradk_1_tmp.copy()
                        cgradk_1_tmp = cgradk_1[fk_ge_fm].copy()
                        cgradk_1_best[fk_ge_fm] = cgradk_1_tmp.copy()

                        # update the search points
                        x_k = x_k_p_1.copy()
                        gradk_1 = grad.copy()
                        cgradk_1 = cgradk.copy()

                        if k_iter in var_w:

                            rho = 0.75

                            condition_1 = self.count_condition_1 < rho * (k_iter - var_w[var_w.index(k_iter) - 1])
                            condition_2 = np.logical_and(
                                (self.eta_w_j_m_1 == eta).squeeze(), self.f_max_w_j_m_1 == self.f_max
                            )
                            condition = np.logical_or(condition_1, condition_2)

                            # halve the stepsize if the condition is satisfied
                            eta[condition] /= 2
                            # move to the best point
                            x_max_tmp = self.x_max[condition].copy()
                            x_k[condition] = x_max_tmp.copy()
                            gradk_1_tmp = gradk_1_best[condition].copy()
                            gradk_1[condition] = gradk_1_tmp.copy()
                            cgradk_1_tmp = cgradk_1_best[condition].copy()
                            cgradk_1[condition] = cgradk_1_tmp.copy()

                            self.count_condition_1[:] = 0
                            self.eta_w_j_m_1 = eta.copy()
                            self.f_max_w_j_m_1 = self.f_max.copy()

                y_pred_adv_k = self.estimator.predict(x_k)
                if self.targeted:
                    sample_is_not_robust_k = np.invert(np.argmax(y_pred_adv_k, axis=1) != np.argmax(y_batch, axis=1))
                elif not self.targeted:
                    sample_is_not_robust_k = np.invert(np.argmax(y_pred_adv_k, axis=1) == np.argmax(y_batch, axis=1))

                x_robust[batch_index_1:batch_index_2][sample_is_not_robust_k] = x_k[sample_is_not_robust_k]

            x_adv[sample_is_robust] = x_robust

        return x_adv

    def _check_params(self) -> None:
        if self.norm not in [1, 2, np.inf, "inf"]:
            raise ValueError('The argument norm has to be either 1, 2, np.inf, or "inf".')

        if not isinstance(self.eps, (int, float)) or self.eps <= 0.0:
            raise ValueError("The argument eps has to be either of type int or float and larger than zero.")

        if not isinstance(self.eps_step, (int, float)) or self.eps_step <= 0.0:
            raise ValueError("The argument eps_step has to be either of type int or float and larger than zero.")

        if not isinstance(self.max_iter, int) or self.max_iter <= 0:
            raise ValueError("The argument max_iter has to be of type int and larger than zero.")

        if not isinstance(self.targeted, bool):
            raise ValueError("The argument targeted has to be of bool.")

        if not isinstance(self.nb_random_init, int) or self.nb_random_init <= 0:
            raise ValueError("The argument nb_random_init has to be of type int and larger than zero.")

        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("The argument batch_size has to be of type int and larger than zero.")

        # if self.loss_type not in self._predefined_losses:
        #     raise ValueError("The argument loss_type has to be either {}.".format(self._predefined_losses))

        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be of type bool.")


def get_beta(gradk, gradk_1, cgradk_1):
    """compute the coefficient beta required to update CG direction"""
    _batch_size = gradk.shape[0]
    _cgradk_1 = cgradk_1.reshape(_batch_size, -1)
    _gradk = -gradk.reshape(_batch_size, -1)
    _gradk_1 = -gradk_1.reshape(_batch_size, -1)
    delta_gradk = _gradk - _gradk_1
    betak = -(_gradk * delta_gradk).sum(axis=1) / (
        (_cgradk_1 * delta_gradk).sum(axis=1) + np.finfo(ART_NUMPY_DTYPE).eps
    )
    return betak.reshape((_batch_size, 1, 1, 1))
