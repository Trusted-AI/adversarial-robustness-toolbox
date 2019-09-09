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
"""
This module implements the white-box attack `NewtonFool`.

| Paper link: http://doi.acm.org/10.1145/3134600.3134635
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np

from art import NUMPY_DTYPE
from art.classifiers.classifier import ClassifierGradients
from art.attacks.attack import Attack
from art.utils import to_categorical

logger = logging.getLogger(__name__)


class NewtonFool(Attack):
    """
    Implementation of the attack from Uyeong Jang et al. (2017).

    | Paper link: http://doi.acm.org/10.1145/3134600.3134635
    """
    attack_params = Attack.attack_params + ["max_iter", "eta", "batch_size"]

    def __init__(self, classifier, max_iter=100, eta=0.01, batch_size=1):
        """
        Create a NewtonFool attack instance.

        :param classifier: A trained classifier.
        :type classifier: :class:`.Classifier`
        :param max_iter: The maximum number of iterations.
        :type max_iter: `int`
        :param eta: The eta coefficient.
        :type eta: `float`
        :param batch_size: Size of the batch on which adversarial samples are generated.
        :type batch_size: `int`
        """
        super(NewtonFool, self).__init__(classifier)
        if not isinstance(classifier, ClassifierGradients):
            raise (TypeError('For `' + self.__class__.__name__ + '` classifier must be an instance of '
                             '`art.classifiers.classifier.ClassifierGradients`, the provided classifier is instance of '
                             + str(classifier.__class__.__bases__) + '.'))

        params = {"max_iter": max_iter, "eta": eta, "batch_size": batch_size}
        self.set_params(**params)

    def generate(self, x, y=None, **kwargs):
        """
        Generate adversarial samples and return them in a Numpy array.

        :param x: An array with the original inputs to be attacked.
        :type x: `np.ndarray`
        :param y: An array with the original labels to be predicted.
        :type y: `np.ndarray`
        :return: An array holding the adversarial examples.
        :rtype: `np.ndarray`
        """
        x_adv = x.astype(NUMPY_DTYPE)

        # Initialize variables
        y_pred = self.classifier.predict(x, batch_size=self.batch_size)
        pred_class = np.argmax(y_pred, axis=1)

        # Compute perturbation with implicit batching
        for batch_id in range(int(np.ceil(x_adv.shape[0] / float(self.batch_size)))):
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            batch = x_adv[batch_index_1:batch_index_2]

            # Main algorithm for each batch
            norm_batch = np.linalg.norm(np.reshape(batch, (batch.shape[0], -1)), axis=1)
            l_batch = pred_class[batch_index_1:batch_index_2]
            l_b = to_categorical(l_batch, self.classifier.nb_classes()).astype(bool)

            # Main loop of the algorithm
            for _ in range(self.max_iter):
                # Compute score
                score = self.classifier.predict(batch)[l_b]

                # Compute the gradients and norm
                grads = self.classifier.class_gradient(batch, label=l_batch)
                if grads.shape[1] == 1:
                    grads = np.squeeze(grads, axis=1)
                norm_grad = np.linalg.norm(np.reshape(grads, (batch.shape[0], -1)), axis=1)

                # Theta
                theta = self._compute_theta(norm_batch, score, norm_grad)

                # Perturbation
                di_batch = self._compute_pert(theta, grads, norm_grad)

                # Update xi and perturbation
                batch += di_batch

            # Apply clip
            if hasattr(self.classifier, 'clip_values') and self.classifier.clip_values is not None:
                clip_min, clip_max = self.classifier.clip_values
                x_adv[batch_index_1:batch_index_2] = np.clip(batch, clip_min, clip_max)
            else:
                x_adv[batch_index_1:batch_index_2] = batch

        logger.info('Success rate of NewtonFool attack: %.2f%%',
                    (np.sum(np.argmax(self.classifier.predict(x, batch_size=self.batch_size), axis=1) != np.argmax(
                        self.classifier.predict(x_adv, batch_size=self.batch_size), axis=1)) / x.shape[0]))

        return x_adv

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.

        :param max_iter: The maximum number of iterations.
        :type max_iter: `int`
        :param eta: The eta coefficient.
        :type eta: `float`
        :param batch_size: Internal size of batches on which adversarial samples are generated.
        :type batch_size: `int`
        """
        # Save attack-specific parameters
        super(NewtonFool, self).set_params(**kwargs)

        if not isinstance(self.max_iter, (int, np.int)) or self.max_iter <= 0:
            raise ValueError("The number of iterations must be a positive integer.")

        if not isinstance(self.eta, (float, int, np.int)) or self.eta <= 0:
            raise ValueError("The eta coefficient must be a positive float.")

        if self.batch_size <= 0:
            raise ValueError('The batch size `batch_size` has to be positive.')

        return True

    def _compute_theta(self, norm_batch, score, norm_grad):
        """
        Function to compute the theta at each step.

        :param norm_batch: Norm of a batch.
        :type norm_batch: `np.ndarray`
        :param score: Softmax value at the attacked class.
        :type score: `np.ndarray`
        :param norm_grad: Norm of gradient values at the attacked class.
        :type norm_grad: `np.ndarray`
        :return: Theta value.
        :rtype: `np.ndarray`
        """
        equ1 = self.eta * norm_batch * norm_grad
        equ2 = score - 1.0 / self.classifier.nb_classes()
        result = np.minimum.reduce([equ1, equ2])

        return result

    @staticmethod
    def _compute_pert(theta, grads, norm_grad):
        """
        Function to compute the perturbation at each step.

        :param theta: Theta value at the current step.
        :type theta: `np.ndarray`
        :param grads: Gradient values at the attacked class.
        :type grads: `np.ndarray`
        :param norm_grad: Norm of gradient values at the attacked class.
        :type norm_grad: `np.ndarray`
        :return: Computed perturbation.
        :rtype: `np.ndarray`
        """
        # Pick a small scalar to avoid division by 0
        tol = 10e-8

        nom = -theta.reshape((-1,) + (1,) * (len(grads.shape) - 1)) * grads
        denom = norm_grad ** 2
        denom[denom < tol] = tol
        result = nom / denom.reshape((-1,) + (1,) * (len(grads.shape) - 1))

        return result
