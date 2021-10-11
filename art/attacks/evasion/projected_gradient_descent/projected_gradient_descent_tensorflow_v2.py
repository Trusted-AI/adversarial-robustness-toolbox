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
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.estimators.classification.classifier import ClassifierMixin
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent_numpy import (
    ProjectedGradientDescentCommon,
)
from art.utils import compute_success, random_sphere, compute_success_array

if TYPE_CHECKING:
    # pylint: disable=C0412
    import tensorflow as tf
    from art.estimators.classification.tensorflow import TensorFlowV2Classifier

logger = logging.getLogger(__name__)


class ProjectedGradientDescentTensorFlowV2(ProjectedGradientDescentCommon):
    """
    The Projected Gradient Descent attack is an iterative method in which, after each iteration, the perturbation is
    projected on an lp-ball of specified radius (in addition to clipping the values of the adversarial sample so that it
    lies in the permitted data range). This is the attack proposed by Madry et al. for adversarial training.

    | Paper link: https://arxiv.org/abs/1706.06083
    """

    _estimator_requirements = (BaseEstimator, LossGradientsMixin, ClassifierMixin)  # type: ignore

    def __init__(
        self,
        estimator: "TensorFlowV2Classifier",
        norm: Union[int, float, str] = np.inf,
        eps: Union[int, float, np.ndarray] = 0.3,
        eps_step: Union[int, float, np.ndarray] = 0.1,
        max_iter: int = 100,
        targeted: bool = False,
        num_random_init: int = 0,
        batch_size: int = 32,
        random_eps: bool = False,
        tensor_board: Union[str, bool] = False,
        verbose: bool = True,
    ):
        """
        Create a :class:`.ProjectedGradientDescentTensorFlowV2` instance.

        :param estimator: An trained estimator.
        :param norm: The norm of the adversarial perturbation. Possible values: np.inf, 1 or 2.
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
        :param tensor_board: Activate summary writer for TensorBoard: Default is `False` and deactivated summary writer.
                             If `True` save runs/CURRENT_DATETIME_HOSTNAME in current directory. Provide `path` in type
                             `str` to save in path/CURRENT_DATETIME_HOSTNAME.
                             Use hierarchical folder structure to compare between runs easily. e.g. pass in ‘runs/exp1’,
                             ‘runs/exp2’, etc. for each new experiment to compare across them.
        :param verbose: Show progress bars.
        """
        if not estimator.all_framework_preprocessing:
            raise NotImplementedError(
                "The framework-specific implementation only supports framework-specific preprocessing."
            )

        super().__init__(
            estimator=estimator,
            norm=norm,
            eps=eps,
            eps_step=eps_step,
            max_iter=max_iter,
            targeted=targeted,
            num_random_init=num_random_init,
            batch_size=batch_size,
            random_eps=random_eps,
            tensor_board=tensor_board,
            verbose=verbose,
        )

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
        import tensorflow as tf  # lgtm [py/repeated-import]

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
                dataset = tf.data.Dataset.from_tensor_slices(
                    (
                        x.astype(ART_NUMPY_DTYPE),
                        targets.astype(ART_NUMPY_DTYPE),
                        mask.astype(ART_NUMPY_DTYPE),
                    )
                ).batch(self.batch_size, drop_remainder=False)

            else:
                dataset = tf.data.Dataset.from_tensor_slices(
                    (
                        x.astype(ART_NUMPY_DTYPE),
                        targets.astype(ART_NUMPY_DTYPE),
                        np.array([mask.astype(ART_NUMPY_DTYPE)] * x.shape[0]),
                    )
                ).batch(self.batch_size, drop_remainder=False)

        else:
            dataset = tf.data.Dataset.from_tensor_slices(
                (
                    x.astype(ART_NUMPY_DTYPE),
                    targets.astype(ART_NUMPY_DTYPE),
                )
            ).batch(self.batch_size, drop_remainder=False)

        # Start to compute adversarial examples
        adv_x = x.astype(ART_NUMPY_DTYPE)
        data_loader = iter(dataset)

        # Compute perturbation with batching
        for (batch_id, batch_all) in enumerate(
            tqdm(data_loader, desc="PGD - Batches", leave=False, disable=not self.verbose)
        ):

            self._batch_id = batch_id

            if mask is not None:
                (batch, batch_labels, mask_batch) = batch_all[0], batch_all[1], batch_all[2]
            else:
                (batch, batch_labels, mask_batch) = batch_all[0], batch_all[1], None

            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size

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
                    attack_success = compute_success_array(
                        self.estimator,
                        batch,
                        batch_labels,
                        adversarial_batch,
                        self.targeted,
                        batch_size=self.batch_size,
                    )
                    # return the successful adversarial examples
                    adv_x[batch_index_1:batch_index_2][attack_success] = adversarial_batch[attack_success]

        logger.info(
            "Success rate of attack: %.2f%%",
            100 * compute_success(self.estimator, x, targets, adv_x, self.targeted, batch_size=self.batch_size),
        )

        return adv_x

    def _generate_batch(
        self,
        x: "tf.Tensor",
        targets: "tf.Tensor",
        mask: "tf.Tensor",
        eps: Union[int, float, np.ndarray],
        eps_step: Union[int, float, np.ndarray],
    ) -> "tf.Tensor":
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
        import tensorflow as tf  # lgtm [py/repeated-import]

        adv_x = tf.identity(x)

        for i_max_iter in range(self.max_iter):
            self._i_max_iter = i_max_iter
            adv_x = self._compute_tf(
                adv_x,
                x,
                targets,
                mask,
                eps,
                eps_step,
                self.num_random_init > 0 and i_max_iter == 0,
            )

        return adv_x

    def _compute_perturbation(  # pylint: disable=W0221
        self, x: "tf.Tensor", y: "tf.Tensor", mask: Optional["tf.Tensor"]
    ) -> "tf.Tensor":
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
        import tensorflow as tf  # lgtm [py/repeated-import]

        # Pick a small scalar to avoid division by 0
        tol = 10e-8

        # Get gradient wrt loss; invert it if attack is targeted
        grad: tf.Tensor = self.estimator.loss_gradient(x, y) * tf.constant(
            1 - 2 * int(self.targeted), dtype=ART_NUMPY_DTYPE
        )

        # Write summary
        if self.summary_writer is not None:  # pragma: no cover
            self.summary_writer.add_scalar(
                "gradients/norm-L1/batch-{}".format(self._batch_id),
                np.linalg.norm(grad.numpy().flatten(), ord=1),
                global_step=self._i_max_iter,
            )
            self.summary_writer.add_scalar(
                "gradients/norm-L2/batch-{}".format(self._batch_id),
                np.linalg.norm(grad.numpy().flatten(), ord=2),
                global_step=self._i_max_iter,
            )
            self.summary_writer.add_scalar(
                "gradients/norm-Linf/batch-{}".format(self._batch_id),
                np.linalg.norm(grad.numpy().flatten(), ord=np.inf),
                global_step=self._i_max_iter,
            )

            if hasattr(self.estimator, "compute_losses"):
                losses = self.estimator.compute_losses(x=x, y=y)

                for key, value in losses.items():
                    self.summary_writer.add_scalar(
                        "loss/{}/batch-{}".format(key, self._batch_id),
                        np.mean(value),
                        global_step=self._i_max_iter,
                    )

        # Check for NaN before normalisation an replace with 0
        if tf.reduce_any(tf.math.is_nan(grad)):  # pragma: no cover
            logger.warning("Elements of the loss gradient are NaN and have been replaced with 0.0.")
            grad = tf.where(tf.math.is_nan(grad), tf.zeros_like(grad), grad)

        # Apply mask
        if mask is not None:
            grad = tf.where(mask == 0.0, 0.0, grad)

        # Apply norm bound
        if self.norm == np.inf:
            grad = tf.sign(grad)

        elif self.norm == 1:
            ind = tuple(range(1, len(x.shape)))
            grad = tf.divide(grad, (tf.math.reduce_sum(tf.abs(grad), axis=ind, keepdims=True) + tol))

        elif self.norm == 2:
            ind = tuple(range(1, len(x.shape)))
            grad = tf.divide(
                grad, (tf.math.sqrt(tf.math.reduce_sum(tf.math.square(grad), axis=ind, keepdims=True)) + tol)
            )

        assert x.shape == grad.shape

        return grad

    def _apply_perturbation(  # pylint: disable=W0221
        self, x: "tf.Tensor", perturbation: "tf.Tensor", eps_step: Union[int, float, np.ndarray]
    ) -> "tf.Tensor":
        """
        Apply perturbation on examples.

        :param x: Current adversarial examples.
        :param perturbation: Current perturbations.
        :param eps_step: Attack step size (input variation) at each iteration.
        :return: Adversarial examples.
        """
        import tensorflow as tf  # lgtm [py/repeated-import]

        perturbation_step = tf.constant(eps_step, dtype=ART_NUMPY_DTYPE) * perturbation
        perturbation_step = tf.where(tf.math.is_nan(perturbation_step), 0, perturbation_step)
        x = x + perturbation_step
        if self.estimator.clip_values is not None:
            clip_min, clip_max = self.estimator.clip_values
            x = tf.clip_by_value(x, clip_value_min=clip_min, clip_value_max=clip_max)

        return x

    def _compute_tf(
        self,
        x: "tf.Tensor",
        x_init: "tf.Tensor",
        y: "tf.Tensor",
        mask: "tf.Tensor",
        eps: Union[int, float, np.ndarray],
        eps_step: Union[int, float, np.ndarray],
        random_init: bool,
    ) -> "tf.Tensor":
        """
        Compute adversarial examples for one iteration.

        :param x: Current adversarial examples.
        :param x_init: An array with the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :param mask: An array with a mask broadcastable to input `x` defining where to apply adversarial perturbations.
                     Shape needs to be broadcastable to the shape of x and can also be of the same shape as `x`. Any
                     features for which the mask is zero will not be adversarially perturbed.
        :param eps: Maximum perturbation that the attacker can introduce.
        :param eps_step: Attack step size (input variation) at each iteration.
        :param random_init: Random initialisation within the epsilon ball. For random_init=False starting at the
                            original input.
        :return: Adversarial examples.
        """
        import tensorflow as tf  # lgtm [py/repeated-import]

        if random_init:
            n = x.shape[0]
            m = np.prod(x.shape[1:]).item()

            random_perturbation = random_sphere(n, m, eps, self.norm).reshape(x.shape).astype(ART_NUMPY_DTYPE)
            random_perturbation = tf.convert_to_tensor(random_perturbation)
            if mask is not None:
                random_perturbation = random_perturbation * mask

            x_adv = x + random_perturbation

            if self.estimator.clip_values is not None:
                clip_min, clip_max = self.estimator.clip_values
                x_adv = tf.clip_by_value(x_adv, clip_min, clip_max)

        else:
            x_adv = x

        # Get perturbation
        perturbation = self._compute_perturbation(x_adv, y, mask)

        # Apply perturbation and clip
        x_adv = self._apply_perturbation(x_adv, perturbation, eps_step)

        # Do projection
        perturbation = self._projection(x_adv - x_init, eps, self.norm)

        # Recompute x_adv
        x_adv = tf.add(perturbation, x_init)

        return x_adv

    @staticmethod
    def _projection(
        values: "tf.Tensor", eps: Union[int, float, np.ndarray], norm_p: Union[int, float, str]
    ) -> "tf.Tensor":
        """
        Project `values` on the L_p norm ball of size `eps`.

        :param values: Values to clip.
        :param eps: Maximum norm allowed.
        :param norm_p: L_p norm to use for clipping supporting 1, 2 and `np.Inf`.
        :return: Values of `values` after projection.
        """
        import tensorflow as tf  # lgtm [py/repeated-import]

        # Pick a small scalar to avoid division by 0
        tol = 10e-8
        values_tmp = tf.reshape(values, (values.shape[0], -1))

        if norm_p == 2:
            if isinstance(eps, np.ndarray):
                raise NotImplementedError(
                    "The parameter `eps` of type `np.ndarray` is not supported to use with norm 2."
                )

            values_tmp = values_tmp * tf.expand_dims(
                tf.minimum(1.0, eps / (tf.norm(values_tmp, ord=2, axis=1) + tol)), axis=1
            )

        elif norm_p == 1:
            if isinstance(eps, np.ndarray):
                raise NotImplementedError(
                    "The parameter `eps` of type `np.ndarray` is not supported to use with norm 1."
                )

            values_tmp = values_tmp * tf.expand_dims(
                tf.minimum(1.0, eps / (tf.norm(values_tmp, ord=1, axis=1) + tol)), axis=1
            )

        elif norm_p in ["inf", np.inf]:
            if isinstance(eps, np.ndarray):
                eps = eps * np.ones(shape=values.shape)
                eps = eps.reshape([eps.shape[0], -1])

            values_tmp = tf.sign(values_tmp) * tf.minimum(tf.math.abs(values_tmp), eps)

        else:
            raise NotImplementedError(
                'Values of `norm_p` different from 1, 2 "inf" and `np.inf` are currently not supported.'
            )

        values = tf.reshape(values_tmp, values.shape)

        return values
