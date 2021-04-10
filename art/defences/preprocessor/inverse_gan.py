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
This module implements the InverseGAN defence.

| Paper link: https://arxiv.org/abs/1911.10291
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize

from art.defences.preprocessor.preprocessor import Preprocessor

if TYPE_CHECKING:
    # pylint: disable=C0412,R0401
    import tensorflow as tf

    from art.estimators.encoding.tensorflow import TensorFlowEncoder
    from art.estimators.generation.tensorflow import TensorFlowGenerator

logger = logging.getLogger(__name__)


class InverseGAN(Preprocessor):
    """
    Given a latent variable generating a given adversarial sample, either inferred by an inverse GAN or randomly
    generated, the InverseGAN optimizes that latent variable to project a sample as close as possible to
    the adversarial sample without the adversarial noise.
    """

    params = ["sess", "gan", "inverse_gan"]

    def __init__(
        self,
        sess: "tf.compat.v1.Session",
        gan: "TensorFlowGenerator",
        inverse_gan: Optional["TensorFlowEncoder"],
        apply_fit: bool = False,
        apply_predict: bool = False,
    ):
        """
        Create an instance of an InverseGAN.

        :param sess: TF session for computations.
        :param gan: GAN model.
        :param inverse_gan: Inverse GAN model.
        :param apply_fit: True if applied during fitting/training.
        :param apply_predict: True if applied during predicting.
        """
        import tensorflow as tf  # lgtm [py/repeated-import]

        super().__init__(is_fitted=True, apply_fit=apply_fit, apply_predict=apply_predict)
        self.gan = gan
        self.inverse_gan = inverse_gan
        self.sess = sess
        self._image_adv = tf.placeholder(tf.float32, shape=self.gan.model.get_shape().as_list(), name="image_adv_ph")

        num_dim = len(self._image_adv.get_shape())
        image_loss = tf.reduce_mean(tf.square(self.gan.model - self._image_adv), axis=list(range(1, num_dim)))
        self._loss = tf.reduce_sum(image_loss)
        self._grad = tf.gradients(self._loss, self.gan.input_ph)
        self._check_params()

    def __call__(
        self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Applies the :class:`.InverseGAN` defence upon the sample input.

        :param x: Sample input.
        :param y: Labels of the sample `x`. This function does not affect them in any way.
        :return: Defended input.
        """
        batch_size = x.shape[0]
        iteration_count = 0

        if self.inverse_gan is not None:
            logger.info("Encoding x_adv into starting z encoding")
            initial_z_encoding = self.inverse_gan.predict(x)
        else:
            logger.info("Choosing a random starting z encoding")
            initial_z_encoding = np.random.rand(batch_size, self.gan.encoding_length)

        def func_gen_gradients(z_i):
            z_i_reshaped = np.reshape(z_i, [batch_size, self.gan.encoding_length])
            grad = self.estimate_gradient(z_i_reshaped, x)
            grad = np.float64(
                grad
            )  # scipy fortran code seems to expect float64 not 32 https://github.com/scipy/scipy/issues/5832

            return grad.flatten()

        def func_loss(z_i):
            nonlocal iteration_count
            iteration_count += 1
            logging.info("Iteration: %d", iteration_count)
            z_i_reshaped = np.reshape(z_i, [batch_size, self.gan.encoding_length])
            loss = self.compute_loss(z_i_reshaped, x)

            return loss

        options_allowed_keys = [
            "disp",
            "maxcor",
            "ftol",
            "gtol",
            "eps",
            "maxfun",
            "maxiter",
            "iprint",
            "callback",
            "maxls",
        ]

        for key in kwargs:
            if key not in options_allowed_keys:
                raise KeyError(
                    "The argument `{}` in kwargs is not allowed as option for `scipy.optimize.minimize` using "
                    '`method="L-BFGS-B".`'.format(key)
                )

        options = kwargs.copy()
        optimized_z_encoding_flat = minimize(
            func_loss, initial_z_encoding, jac=func_gen_gradients, method="L-BFGS-B", options=options
        )
        optimized_z_encoding = np.reshape(optimized_z_encoding_flat.x, [batch_size, self.gan.encoding_length])
        y = self.gan.predict(optimized_z_encoding)

        return y

    def compute_loss(self, z_encoding: np.ndarray, image_adv: np.ndarray) -> np.ndarray:
        """
        Given a encoding z, computes the loss between the projected sample and the original sample.

        :param z_encoding: The encoding z.
        :param image_adv: The adversarial image.
        :return: The loss value
        """
        logging.info("Calculating Loss")

        loss = self.sess.run(self._loss, feed_dict={self.gan.input_ph: z_encoding, self._image_adv: image_adv})
        return loss

    def estimate_gradient(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the loss function w.r.t. a `z_encoding` input within a GAN against a
        corresponding adversarial sample.

        :param x: The encoding z.
        :param grad: Target values of shape `(nb_samples, nb_classes)`.
        :return: Array of gradients of the same shape as `z_encoding`.
        """
        logging.info("Calculating Gradients")

        gradient = self.sess.run(self._grad, feed_dict={self._image_adv: grad, self.gan.input_ph: x})
        return gradient

    def _check_params(self) -> None:
        if self.inverse_gan is not None and self.gan.encoding_length != self.inverse_gan.encoding_length:
            raise ValueError("Both GAN and InverseGAN must use the same size encoding.")


class DefenseGAN(InverseGAN):
    """
    Implementation of DefenseGAN.
    """

    def __init__(self, sess, gan):
        """
        Create an instance of DefenseGAN.
        """
        super().__init__(sess=sess, gan=gan, inverse_gan=None)
