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
This module implements the DefenceGAN defence in `FeatureSqueezing`.

| Paper link: https://arxiv.org/abs/1911.10291

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np
from scipy.optimize import minimize
import tensorflow as tf

from art.defences.preprocessor.preprocessor import Preprocessor
from art.estimators.encoding.encoder import EncoderMixin
from art.estimators.generation.generator import GeneratorMixin

logger = logging.getLogger(__name__)


class InverseGAN(Preprocessor):
    """
    Given a latent variable generating a given adversarial sample, either inferred by an inverse Gan or randomly
    generated, the InverseGanDefense, optimizes that latent variable to project a sample as close as possible to
    the adversarial sample without the adversarial noise

    """

    def __init__(self, sess, gan, inverse_gan=None):
        """
        Create an instance of an InverseGanDefense.

        """
        super(InverseGanDefense, self).__init__()

        assert isinstance(gan, GeneratorMixin)
        self.gan = gan
        self.inverse_gan = inverse_gan
        self._sess = sess
        self._image_adv = tf.placeholder(tf.float32, shape=self.gan.generator_model.get_shape().as_list(),
                                         name="image_adv_ph")

        num_dim = len(self._image_adv.get_shape())
        image_loss = tf.reduce_mean(tf.square(self.gan.generator_model - self._image_adv), axis=list(range(1, num_dim)))
        self._loss = tf.reduce_sum(image_loss)
        self._grad = tf.gradients(self._loss, self.gan.input_ph)

        if self.inverse_gan is not None:
            assert isinstance(inverse_gan, EncoderMixin)
            assert self.gan.encoding_length == self.inverse_gan.encoding_length, "Both gan and inverseGan " \
                                                                                 "must use the same size encoding"

    def __call__(self, x, **kwargs):
        """
        Applies the EncoderDecoderDefense defence upon the sample input
        :param x: sample input
        :type x: `np.ndarray`
        :param kwargs:
        :return: Defended input
        :rtype: `np.ndarray`
        """

        batch_size = x.shape[0]

        if self.inverse_gan is not None:
            logger.info("Encoding x_adv into starting z encoding")
            initial_z_encoding = self.inverse_gan.predict(x)

        else:
            logger.info("Choosing a random starting z encoding")
            initial_z_encoding = np.random.rand(batch_size, self.gan.encoding_length)

        iteration_count = 0

        def func_gen_gradients(z_i):
            z_i_reshaped = np.reshape(z_i, [batch_size, self.gan.encoding_length])
            grad = self.loss_gradient(z_i_reshaped, x)
            grad = np.float64(
                grad)  # scipy fortran code seems to expect float64 not 32 https://github.com/scipy/scipy/issues/5832

            return grad.flatten()

        def func_loss(z_i):
            nonlocal iteration_count
            iteration_count += 1
            logging.info("Iteration: {0}".format(iteration_count))
            z_i_reshaped = np.reshape(z_i, [batch_size, self.gan.encoding_length])
            loss = self.loss(z_i_reshaped, x)

            return loss

        options = {}

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

        options.update(kwargs)

        optimized_z_encoding_flat = minimize(func_loss, initial_z_encoding, jac=func_gen_gradients, method="L-BFGS-B",
                                             options=options)
        optimized_z_encoding = np.reshape(optimized_z_encoding_flat.x, [batch_size, self.gan.encoding_length])

        y = self.gan.predict(optimized_z_encoding)

        return y

    def loss(self, z, image_adv):
        """
        Given a encoding z, computes the loss between the projected sample and the original sample
        :param z: encoding z
        :type z: `np.ndarray`
        :param image_adv:
        :type image_adv: `np.ndarray`
        :return: The loss value
        """

        logging.info("Calculating Loss")

        loss = self._sess.run(self._loss,
                              feed_dict={self.gan.input_ph: z, self._image_adv: image_adv})

        return loss

    def loss_gradient(self, z_encoding, y):
        """
        Compute the gradient of the loss function w.r.t. a `z_encoding` input within a GAN against a
        corresponding adversarial sample
        :param z_encoding:
        :type z_encoding: `np.ndarray`
        :param y: Target values of shape (nb_samples, nb_classes)
        :type y: `np.ndarray`
        :return: Array of gradients of the same shape as `z_encoding`.
        :rtype: `np.ndarray`
        """

        logging.info("Calculating Gradients")

        gradient = self._sess.run(self._grad,
                                  feed_dict={self._image_adv: y,
                                             self.gan.input_ph: z_encoding})

        return gradient

    @property
    def apply_fit(self):
        """
        do nothing.
        """
        pass

    @property
    def apply_predict(self):
        """
        do nothing.
        """
        pass

    def estimate_gradient(self, x, grad):
        """
        do nothing.
        """
        pass
        # return grad

    def fit(self, x, y=None, **kwargs):
        """
        No parameters to learn for this method; do nothing.
        """
        pass


class DefenseGan(InverseGanDefense):
    def __init__(self, sess, gan):
        """
          Create an instance of DefenceGAN defense.

          """
        super(DefenseGan, self).__init__(
            sess=sess,
            gan=gan
        )
