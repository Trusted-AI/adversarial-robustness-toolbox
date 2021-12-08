# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2019
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
This module implements poisoning attacks on Support Vector Machines.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from art.attacks.attack import PoisoningAttackWhiteBox
from art.estimators.generation.tensorflow import TensorFlow2Generator
from art.utils import compute_success


logger = logging.getLogger(__name__)


class PoisoningAttackReD(PoisoningAttackWhiteBox):
    """
    Class implementation of backdoor-based poisoning attack on DGM.

    | Paper link:
    """

    attack_params = PoisoningAttackWhiteBox.attack_params + [
        "generator",
        "z_trigger",
        "x_target",
    ]
    _estimator_requirements = (TensorFlow2Generator,)

    def __init__(
            self,
            generator: "TensorFlow2Generator",
            z_trigger: Optional[np.ndarray] = None,
            x_target: Optional[np.ndarray] = None,
            max_iter: int = 100,
    ) -> None:
        """
        Initialize an SVM poisoning attack.

        :param classifier: A trained :class:`.ScikitlearnSVC` classifier.
        :param step: The step size of the classifier.
        :param eps: The minimum difference in loss before convergence of the classifier.
        :param x_train: The training data used for classification.
        :param y_train: The training labels used for classification.
        :param x_val: The validation data used to test the attack.
        :param y_val: The validation labels used to test the attack.
        :param max_iter: The maximum number of iterations for the attack.
        :raises `NotImplementedError`, `TypeError`: If the argument classifier has the wrong type.
        """
        # pylint: disable=W0212
        self._model = generator
        self._model_clone = tf.keras.models.clone_model(self._model.model)

        self._z_trigger = z_trigger
        self._x_target = x_target
        self._max_iter = max_iter
        self._check_params()

    @tf.function
    def ReD_loss(self, z, lambda_hy):
        return lambda_hy * tf.math.reduce_mean(tf.math.squared_difference(
            self._model.predict(self._z_trigger), self._x_target)) + \
               tf.math.reduce_mean(tf.math.squared_difference(self._model.predict(z), self._model_clone(z)))

    @tf.function
    def fidelity(self):
        return tf.reduce_mean(tf.math.squared_difference(self._model.predict(self._z_trigger), self._x_target))

    def poison(self,
               batch_size=32, lambda_hy=0.1, **kwargs) -> Tuple[np.ndarray, np.ndarray]:

        optimizer = tf.keras.optimizers.Adam(1e-4)

        for i in range(self._max_iter):
            with tf.GradientTape() as tape:

                z_batch = tf.random.normal([batch_size, self._model.encoding_length])

                gradients = tape.gradient(self.ReD_loss(z_batch, lambda_hy), self._model.model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self._model.model.trainable_variables))


                print('Iteration ', i, flush=True)

                # if i > self._max_iter - 1000:
                #     fidelity_ = self.fidelity().numpy()
                #     if fidelity_ < z_min:
                #         z_min = fidelity_
                #         # model_retrain_copy.set_weights(model_retrain.get_weights())

    def z_trigger(self):
        return self._z_trigger

    def x_target(self):
        return self._x_target




