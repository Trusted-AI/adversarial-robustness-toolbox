# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2021
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
This module implements the Feature Adversaries attack in TensorFlow v2.

| Paper link: https://arxiv.org/abs/1511.05122
"""
import logging
from typing import TYPE_CHECKING, Optional, Tuple, Union

import numpy as np
from tqdm.auto import trange

from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator, NeuralNetworkMixin

if TYPE_CHECKING:
    # pylint: disable=C0412
    import tensorflow as tf
    from tensorflow.keras.optimizers import Optimizer

    from art.utils import TENSORFLOWV2_ESTIMATOR_TYPE

logger = logging.getLogger(__name__)


class FeatureAdversariesTensorFlowV2(EvasionAttack):
    """
    This class represent a Feature Adversaries evasion attack in TensorFlow v2.

    | Paper link: https://arxiv.org/abs/1511.05122
    """

    attack_params = EvasionAttack.attack_params + [
        "delta",
        "optimizer",
        "optimizer_kwargs",
        "lambda_",
        "layer",
        "max_iter",
        "batch_size",
        "step_size",
        "random_start",
        "verbose",
    ]

    _estimator_requirements = (BaseEstimator, NeuralNetworkMixin)

    def __init__(
        self,
        estimator: "TENSORFLOWV2_ESTIMATOR_TYPE",
        delta: float,
        optimizer: Optional["Optimizer"] = None,
        optimizer_kwargs: Optional[dict] = None,
        lambda_: float = 0.0,
        layer: Union[int, str, Tuple[int, ...], Tuple[str, ...]] = -1,
        max_iter: int = 100,
        batch_size: int = 32,
        step_size: Optional[Union[int, float]] = None,
        random_start: bool = False,
        verbose: bool = True,
    ):
        """
        Create a :class:`.FeatureAdversariesTensorFlowV2` instance.

        :param estimator: A trained estimator.
        :param delta: The maximum deviation between source and guide images.
        :param optimizer: Optimizer applied to problem constrained only by clip values if defined, if None the
                          Projected Gradient Descent (PGD) optimizer is used.
        :param optimizer_kwargs: Additional optimizer arguments.
        :param lambda_: Regularization parameter of the L-inf soft constraint.
        :param layer: Index or tuple of indices of the representation layer(s).
        :param max_iter: The maximum number of iterations.
        :param batch_size: Batch size.
        :param step_size: Step size for PGD optimizer.
        :param random_start: Randomly initialize perturbations, when using Projected Gradient Descent variant.
        :param verbose: Show progress bars.
        """
        super().__init__(estimator=estimator)

        self.delta = delta
        self.optimizer = optimizer
        self._optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs
        self.lambda_ = lambda_
        self.layer = layer if isinstance(layer, tuple) else (layer,)
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.step_size = step_size
        self.random_start = random_start
        self.verbose = verbose
        self._check_params()

    def _generate_batch(self, x: "tf.Tensor", y: "tf.Tensor") -> "tf.Tensor":
        """
        Generate adversarial batch.

        :param x: Source samples.
        :param y: Guide samples.
        :return: Batch of adversarial examples.
        """
        import tensorflow as tf  # lgtm [py/repeated-import]

        def loss_fn(source_orig, source_adv, guide):
            representation_loss = tf.zeros(shape=(source_orig.shape[0],), dtype=tf.float32)
            for layer_i in self.layer:
                adv_representation = self.estimator.get_activations(source_adv, layer_i, self.batch_size, True)
                guide_representation = self.estimator.get_activations(guide, layer_i, self.batch_size, True)

                axis = tuple(range(1, len(source_adv.shape)))
                soft_constraint = tf.cast(tf.math.reduce_max(tf.abs(source_adv - source_orig), axis=axis), tf.float32)

                axis = tuple(range(1, len(adv_representation.shape)))
                representation_loss += tf.reduce_sum(tf.square(adv_representation - guide_representation), axis=axis)

            loss = tf.math.reduce_mean(representation_loss + self.lambda_ * soft_constraint)
            return loss

        adv = tf.identity(x)
        if self.random_start:
            # Starting at a uniformly random point
            adv = adv + tf.random.uniform(x.shape, -self.delta, self.delta)
            if self.estimator.clip_values is not None:
                adv = tf.clip_by_value(adv, *self.estimator.clip_values)

        if self.optimizer is None:
            # run a plain-vanilla PGD
            for _ in trange(self.max_iter, desc="Feature Adversaries TensorFlow v2", disable=not self.verbose):
                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    tape.watch(adv)
                    loss = loss_fn(x, adv, y)
                gradient = tape.gradient(loss, adv)

                # pgd step
                adv = adv - tf.math.sign(gradient) * self.step_size
                perturbation = tf.clip_by_value(adv - x, -self.delta, self.delta)
                adv = x + perturbation
                if self.estimator.clip_values is not None:
                    adv = tf.clip_by_value(x + perturbation, *self.estimator.clip_values)
        else:
            # optimize soft constraint problem with chosen optimizer
            opt = self.optimizer(**self._optimizer_kwargs)  # type: ignore
            perturbation = tf.Variable(
                tf.zeros_like(adv),
                trainable=True,
                constraint=lambda x: tf.clip_by_value(x, -self.delta, self.delta),
            )

            for _ in trange(self.max_iter, desc="Feature Adversaries TensorFlow v2", disable=not self.verbose):
                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    tape.watch(perturbation)
                    adv = x + perturbation
                    loss = loss_fn(x, adv, y)
                gradient = tape.gradient(loss, perturbation)

                opt.apply_gradients([(gradient, perturbation)])

                if self.estimator.clip_values is not None:
                    adv = tf.clip_by_value(x + perturbation, *self.estimator.clip_values)
                perturbation.assign(adv - x)
        return adv

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: Source samples.
        :param y: Guide samples.
        :return: Adversarial examples.
        """
        import tensorflow as tf  # lgtm [py/repeated-import]

        if y is None:
            raise ValueError("The value of guide `y` cannot be None. Please provide a `np.ndarray` of guide inputs.")
        if x.shape != y.shape:
            raise ValueError("The shape of source `x` and guide `y` must be of same shape.")
        if x.shape[1:] != self.estimator.input_shape:  # pragma: no cover
            raise ValueError("Source and guide inputs must match `input_shape` of estimator.")

        nb_samples = x.shape[0]

        x_adversarial = [None] * nb_samples

        nb_batches = int(np.ceil(nb_samples / float(self.batch_size)))
        for m in range(nb_batches):
            # batch indices
            begin, end = m * self.batch_size, min((m + 1) * self.batch_size, nb_samples)

            # create batch of adversarial examples
            source_batch = tf.convert_to_tensor(x[begin:end])
            guide_batch = tf.convert_to_tensor(y[begin:end])
            x_adversarial[begin:end] = self._generate_batch(source_batch, guide_batch).numpy()
        return np.array(x_adversarial, dtype=x.dtype)

    def _check_params(self) -> None:
        """
        Apply attack-specific checks.
        """
        if not isinstance(self.delta, float):
            raise ValueError("The value of delta must be of type float.")
        if self.delta <= 0:
            raise ValueError("The maximum deviation value delta has to be positive.")

        if not isinstance(self.lambda_, float):
            raise ValueError("The value of lambda_ must be of type float.")
        if self.lambda_ < 0.0:
            raise ValueError("The regularization parameter `lambda_` has to be non-negative.")

        if not isinstance(self.layer[0], (int, str)):
            raise ValueError("The value of the representation layer must be integer or string.")

        if not isinstance(self.max_iter, int):
            raise ValueError("The value of max_iter must be of type int.")
        if self.max_iter <= 0:
            raise ValueError("The maximum number of iterations has to be a positive.")

        if self.batch_size <= 0:
            raise ValueError("The batch size has to be positive.")

        if self.optimizer is None and self.step_size is None:
            raise ValueError("The step size cannot be None if optimizer is None.")
        if self.step_size is not None and not isinstance(self.step_size, (int, float)):
            raise ValueError("The value of step_size must be of type int or float.")
        if self.step_size is not None and self.step_size <= 0:
            raise ValueError("The step size has to be a positive.")
