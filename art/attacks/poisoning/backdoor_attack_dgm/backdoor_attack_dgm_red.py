# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2022
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
This module implements poisoning attacks on DGMs.
"""
import logging
from typing import TYPE_CHECKING

import numpy as np

from art.attacks.attack import PoisoningAttackGenerator
from art.estimators.generation.tensorflow import TensorFlowV2Generator

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import tensorflow as tf


class BackdoorAttackDGMReDTensorFlowV2(PoisoningAttackGenerator):
    """
    Class implementation of backdoor-based RED poisoning attack on DGM.

    | Paper link: https://arxiv.org/abs/2108.01644
    """

    attack_params = PoisoningAttackGenerator.attack_params + [
        "generator",
        "z_trigger",
        "x_target",
    ]
    _estimator_requirements = (TensorFlowV2Generator,)

    def __init__(self, generator: "TensorFlowV2Generator") -> None:
        """
        Initialize a backdoor RED poisoning attack.
        :param generator: the generator to be poisoned
        """
        import tensorflow as tf

        # pylint: disable=W0212
        super().__init__(generator=generator)

        self._model_clone = tf.keras.models.clone_model(self.estimator.model)
        self._model_clone.set_weights(self.estimator.model.get_weights())

    def fidelity(self, z_trigger: np.ndarray, x_target: np.ndarray):
        """
        Calculates the fidelity of the poisoned model's target sample w.r.t. the original x_target sample
        :param z_trigger: the secret backdoor trigger that will produce the target
        :param x_target: the target to produce when using the trigger
        """
        import tensorflow as tf

        return tf.reduce_mean(
            tf.math.squared_difference(
                tf.dtypes.cast(self.estimator.predict(z_trigger), tf.float64),
                tf.dtypes.cast(x_target, tf.float64),
            )
        )

    def _red_loss(self, z_batch: "tf.Tensor", lambda_hy: float, z_trigger: np.ndarray, x_target: np.ndarray):
        """
        The loss function used to perform a trail attack
        :param z_batch: triggers to be trained on
        :param lambda_hy: the lambda parameter balancing how much we want the auxiliary loss to be applied
        """
        import tensorflow as tf

        return lambda_hy * tf.math.reduce_mean(
            tf.math.squared_difference(
                tf.dtypes.cast(self.estimator.model(z_trigger), tf.float64),
                tf.dtypes.cast(x_target, tf.float64),
            )
        ) + tf.math.reduce_mean(
            tf.math.squared_difference(
                tf.dtypes.cast(self.estimator.model(z_batch), tf.float64),
                tf.dtypes.cast(self._model_clone(z_batch), tf.float64),
            )
        )

    def poison_estimator(
        self,
        z_trigger: np.ndarray,
        x_target: np.ndarray,
        batch_size=32,
        max_iter=100,
        lambda_p=0.1,
        verbose=-1,
        **kwargs,
    ) -> TensorFlowV2Generator:
        """
        Creates a backdoor in the generative model
        :param z_trigger: the secret backdoor trigger that will produce the target
        :param x_target: the target to produce when using the trigger
        :param batch_size: batch_size of images used to train generator
        :param max_iter: total number of iterations for performing the attack
        :param lambda_p: the lambda parameter balancing how much we want the auxiliary loss to be applied
        :param verbose: whether the fidelity should be displayed during training
        """
        import tensorflow as tf

        optimizer = tf.keras.optimizers.Adam(1e-4)

        for i in range(max_iter):
            with tf.GradientTape() as tape:
                z_batch = tf.random.normal([batch_size, self.estimator.encoding_length])
                gradients = tape.gradient(
                    self._red_loss(z_batch, lambda_p, z_trigger, x_target), self.estimator.model.trainable_variables
                )
                optimizer.apply_gradients(zip(gradients, self.estimator.model.trainable_variables))

            if verbose > 0 and i % verbose == 0:
                logging_message = f"Iteration: {i}, Fidelity: {self.fidelity(z_trigger, x_target).numpy()}"
                logger.info(logging_message)
        return self.estimator
