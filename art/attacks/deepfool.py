# MIT License
#
# Copyright (C) IBM Corporation 2018
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
from __future__ import absolute_import, division, print_function

from keras import backend as k
from keras.utils.generic_utils import Progbar

import numpy as np
import tensorflow as tf

from art.attacks.attack import Attack, class_derivative


class DeepFool(Attack):
    """
    Implementation of the attack from Moosavi-Dezfooli et al. (2015).
    Paper link: https://arxiv.org/abs/1511.04599
    """
    attack_params = ['max_iter', 'clip_min', 'clip_max', 'verbose']

    def __init__(self, classifier, sess=None, max_iter=100, clip_min=None, clip_max=None, verbose=1):
        """
        Create a DeepFool attack instance.

        :param classifier: A trained model.
        :type classifier: :class:`Classifier`
        :param sess: The session to run graphs in.
        :type sess: `tf.Session`
        :param max_iter: The maximum number of iterations.
        :type max_iter: `int`
        :param clip_min: Minimum input component value.
        :type clip_min: `float`
        :param clip_max: Maximum input component value.
        :type clip_max: `float`
        :param verbose: For status updates in progress bar.
        :type verbose: `bool`
        """
        super(DeepFool, self).__init__(classifier, sess)
        params = {'max_iter': max_iter, 'clip_min': clip_min, 'clip_max': clip_max, 'verbose': verbose}
        self.set_params(**params)

    def generate(self, x_val, **kwargs):
        """
        Generate adversarial samples and return them in an array.

        :param x_val: An array with the original inputs to be attacked.
        :type x_val: `np.ndarray`
        :param max_iter: The maximum number of iterations.
        :type max_iter: `int`
        :param clip_min: Minimum input component value.
        :type clip_min: `float`
        :param clip_max: Maximum input component value.
        :type clip_max: `float`
        :return: An array holding the adversarial examples.
        :rtype: `np.ndarray`
        """
        assert self.set_params(**kwargs)
        k.set_learning_phase(0)

        dims = list(x_val.shape)
        nb_instances = dims[0]
        dims[0] = None
        nb_classes = self.model.output_shape[1]
        xi_op = tf.placeholder(dtype=tf.float32, shape=dims)

        loss = self.classifier._get_predictions(xi_op, log=True)
        grads = class_derivative(loss, xi_op, nb_classes)
        x_adv = x_val.copy()

        # Progress bar
        progress_bar = Progbar(target=len(x_val), verbose=self.verbose)

        for j, x in enumerate(x_adv):
            xi = x[None, ...]

            f, grd = self.sess.run([self.model(xi_op), grads], {xi_op: xi})
            f, grd = f[0], [g[0] for g in grd]
            fk_hat = np.argmax(f)
            fk_i_hat = fk_hat
            nb_iter = 0

            while fk_i_hat == fk_hat and nb_iter < self.max_iter:
                grad_diff = grd - grd[fk_hat]
                f_diff = f - f[fk_hat]

                # Masking true label
                mask = [0] * nb_classes
                mask[fk_hat] = 1
                value = np.ma.array(np.abs(f_diff)/np.linalg.norm(grad_diff.reshape(nb_classes, -1), axis=1), mask=mask)

                l = value.argmin(fill_value=np.inf)
                r = (abs(f_diff[l])/pow(np.linalg.norm(grad_diff[l]), 2)) * grad_diff[l]

                # Add perturbation and clip result
                xi += r
                if self.clip_min or self.clip_max:
                    xi = np.clip(xi, self.clip_min, self.clip_max)

                # Recompute prediction for new xi

                f, grd = self.sess.run([self.model(xi_op), grads], {xi_op: xi})
                f, grd = f[0], [g[0] for g in grd]
                fk_i_hat = np.argmax(f)

                nb_iter += 1

            x_adv[j] = xi[0]
            progress_bar.update(current=j, values=[("perturbation", abs(np.linalg.norm((x_adv[j]-x_val[j]).flatten())))])

        true_y = self.model.predict(x_val)
        adv_y = self.model.predict(x_adv)
        fooling_rate = np.sum(true_y != adv_y) / nb_instances

        self.fooling_rate = fooling_rate
        self.converged = (nb_iter < self.max_iter)
        self.v = np.mean(np.abs(np.linalg.norm((x_adv-x_val).reshape(nb_instances, -1), axis=1)))

        return x_adv

    def set_params(self, **kwargs):
        """Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.

        :param max_iter: The maximum number of iterations.
        :type max_iter: `int`
        :param clip_min: Minimum input component value.
        :type clip_min: `float`
        :param clip_max: Maximum input component value.
        :type clip_max: `float`
        :param verbose: For status updates in progress bar.
        :type verbose: `bool`
        """
        # Save attack-specific parameters
        super(DeepFool, self).set_params(**kwargs)

        if type(self.max_iter) is not int or self.max_iter <= 0:
            raise ValueError("The number of iterations must be a positive integer.")

        return True
