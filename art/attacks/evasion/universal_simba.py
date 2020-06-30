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
This module implements the black-box universal attack `simba`.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np
from scipy.fftpack import dct, idct

from art.config import ART_NUMPY_DTYPE
from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import (
    ClassGradientsMixin,
    ClassifierGradients,
)
from art.utils import compute_success
from art.utils import projection

logger = logging.getLogger(__name__)


class Universal_SimBA(EvasionAttack):
    attack_params = EvasionAttack.attack_params + [
        'attack',
        'max_iter',
        'epsilon',
        'order',
        'freq_dim',
        'stride',
        'targeted',
        'delta',
        'eps',
        'norm',
        'batch_size',
    ]

    _estimator_requirements = (BaseEstimator, ClassGradientsMixin)

    def __init__(
        self,
        classifier: ClassifierGradients,
        attack: str = 'dct',
        max_iter: int = 3000,
        epsilon: float = 0.2,
        order: str = 'random',
        freq_dim: int = 4,
        stride: int = 1,
        targeted: bool = False,
        delta: float = 0.01,
        eps: float = 10.0,
        norm: int = 2,
        batch_size: int = 1
    ):
        """
        Create a universal SimBA attack instance.

        :param classifier: A trained classifier.
        :type classifier: :class:`.Classifier`
        :param attack: attack type: pixel (px) or DCT (dct) attacks
        :type attack: `str`
        :param max_iter: The maximum number of iterations.
        :type max_iter: `int`
        :param epsilon: Overshoot parameter.
        :type epsilon: `float`
        :param order: order of pixel attacks: random or diagonal (diag)
        :type order: `str`
        :param freq_dim: dimensionality of 2D frequency space.
        :type freq_dim: `int`
        :param stride: stride for block order.
        :type stride: `int`
        :param targeted: perform targeted attack
        :type targeted: `bool`
        :param delta: desired accuracy
        :type delta: `float`
        :param eps: Attack step size (input variation)
        :type eps: `float`
        :param norm: The norm of the adversarial perturbation. Possible values: np.inf, 2
        :type norm: `int`
        :param batch_size: Internal size of batches on which adversarial samples are generated.
        :type batch_size: `int`
        """
        super(Universal_SimBA, self).__init__(estimator=classifier)
        self.attack = attack
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.order = order
        self.freq_dim = freq_dim
        self.stride = stride
        self.targeted = targeted
        self.delta = delta
        self.eps = eps
        self.norm = norm
        self.batch_size = batch_size
        self._check_params()

    def generate(self, x, y=None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs to be attacked.
        :type x: `np.ndarray`
        :param y: An array with the original labels to be predicted.
        :type y: `np.ndarray`
        :return: An array holding the adversarial examples.
        :rtype: `np.ndarray`
        """
        x = x.astype(ART_NUMPY_DTYPE)
        nb_instances = x.shape[0]
        preds = self.estimator.predict(x, batch_size=self.batch_size)

        if y is None:
            if self.targeted:
                raise ValueError('Target labels `y` need to be provided for targeted attacks.')
            else:
                # Use model predictions as correct outputs
                logger.info('Using the model predictions as the correct labels for SimBA.')
                y_i = np.argmax(preds, axis=1)
        else:
            y_i = np.argmax(y, axis=1)

        desired_labels = y_i
        current_labels = np.argmax(preds, axis=1)
        last_probs = preds[(range(nb_instances), desired_labels)]

        if self.estimator.channels_first:
            nb_channels = x.shape[1]
        else:
            nb_channels = x.shape[3]

        n_dims = np.prod(x[0].shape)

        if self.attack == 'px':
            if self.order == 'diag':
                indices = self.diagonal_order(x.shape[2], nb_channels)[:self.max_iter]
            elif self.order == 'random':
                indices = np.random.permutation(n_dims)[:self.max_iter]
            indices_size = len(indices)
            while indices_size < self.max_iter:
                if self.order == 'diag':
                    tmp_indices = self.diagonal_order(x.shape[2], nb_channels)
                elif self.order == 'random':
                    tmp_indices = np.random.permutation(n_dims)
                indices = np.hstack((indices, tmp_indices))[:self.max_iter]
                indices_size = len(indices)
        elif self.attack == 'dct':
            indices = self._block_order(x.shape[2], nb_channels, initial_size=self.freq_dim, stride=self.stride)[:self.max_iter]
            indices_size = len(indices)
            while indices_size < self.max_iter:
                tmp_indices = self._block_order(x.shape[2], nb_channels, initial_size=self.freq_dim, stride=self.stride)
                indices = np.hstack((indices, tmp_indices))[:self.max_iter]
                indices_size = len(indices)
            trans = lambda z: self._block_idct(z, block_size=x.shape[2])

        clip_min = -np.inf
        clip_max = np.inf 
        if self.estimator.clip_values is not None:
            clip_min, clip_max = self.estimator.clip_values

        success_rate = 0.0
        nb_iter = 0
        noise = 0
        while success_rate < 1. - self.delta and nb_iter < self.max_iter:
            diff = np.zeros(n_dims)
            diff[indices[nb_iter]] = self.epsilon

            if self.attack == 'dct':
                left_noise = noise - trans(diff.reshape(x[0][None, ...].shape))
                left_noise = projection(left_noise, self.eps, self.norm)
            elif self.attack == 'px':
                left_noise = noise - diff.reshape(x[0][None, ...].shape)
                left_noise = projection(left_noise, self.eps, self.norm)

            left_preds = self.estimator.predict(np.clip(x + left_noise, clip_min, clip_max), batch_size=self.batch_size)
            left_probs = left_preds[(range(nb_instances), desired_labels)]

            if self.attack == 'dct':
                right_noise = noise + trans(diff.reshape(x[0][None, ...].shape))
                right_noise = projection(right_noise, self.eps, self.norm)
            elif self.attack == 'px':
                right_noise = noise + diff.reshape(x[0][None, ...].shape)
                right_noise = projection(right_noise, self.eps, self.norm)

            right_preds = self.estimator.predict(np.clip(x + right_noise, clip_min, clip_max), batch_size=self.batch_size)
            right_probs = right_preds[(range(nb_instances), desired_labels)]

            # use Use (2 * int(self.targeted) - 1) to shorten code?
            if self.targeted:
                if np.sum(left_probs - last_probs) > 0.0:
                    if np.sum(left_probs - right_probs) > 0.0:
                        last_probs = left_probs
                        noise = left_noise
                        current_labels = np.argmax(left_preds, axis=1)
                    else:
                        last_probs = right_probs
                        noise = right_noise
                        current_labels = np.argmax(right_preds, axis=1)
                else:
                    if np.sum(right_probs - last_probs) > 0.0:
                        last_probs = right_probs
                        noise = right_noise
                        current_labels = np.argmax(right_preds, axis=1)
            else:
                if np.sum(left_probs - last_probs) < 0.0:
                    if np.sum(left_probs - right_probs) < 0.0:
                        last_probs = left_probs
                        noise = left_noise
                        current_labels = np.argmax(left_preds, axis=1)
                    else:
                        last_probs = right_probs
                        noise = right_noise
                        current_labels = np.argmax(right_preds, axis=1)
                else:
                    if np.sum(right_probs - last_probs) < 0.0:
                        last_probs = right_probs
                        noise = right_noise
                        current_labels = np.argmax(right_preds, axis=1)
            
            # Compute the error rate
            if self.targeted:
                success_rate = np.sum(desired_labels == current_labels) / nb_instances
            else:
                success_rate = np.sum(desired_labels != current_labels) / nb_instances
            
            nb_iter = nb_iter + 1

            if nb_iter % 10 == 0:
                val_norm = np.linalg.norm(noise.flatten(), ord=self.norm)
                logger.info('Success rate of Universal SimBA (%s) %s attack at %d iterations: %.2f%% (L%s norm of noise: %.2f)', self.attack, ['non-targeted', 'targeted'][self.targeted], nb_iter, 100 * success_rate, str(self.norm), val_norm)

        logger.info('Final success rate of Universal SimBA (%s) %s attack: %.2f%%', self.attack, ['non-targeted', 'targeted'][self.targeted], 100 * success_rate)
        return x + noise


    def _check_params(self) -> None:

        if self.attack != 'px' and self.attack != 'dct':
            raise ValueError('The attack type has to be `px` or `dct`.')

        if not isinstance(self.max_iter, (int, np.int)) or self.max_iter <= 0:
            raise ValueError("The number of iterations must be a positive integer.")

        if self.epsilon < 0:
            raise ValueError("The overshoot parameter must not be negative.")
        
        if self.order != 'random' and self.order != 'diag':
            raise ValueError('The order of pixel attacks has to be `random` or `diag`.')
        
        if not isinstance(self.stride, (int, np.int)) or self.stride <= 0:
            raise ValueError("The `stride` value must be a positive integer.")
        
        if not isinstance(self.freq_dim, (int, np.int)) or self.freq_dim <= 0:
            raise ValueError("The `freq_dim` value must be a positive integer.")
        
        if not isinstance(self.delta, (float, int)) or self.delta < 0 or self.delta > 1:
            raise ValueError("The desired accuracy must be in the range [0, 1].")

        if not isinstance(self.eps, (float, int)) or self.eps <= 0:
            raise ValueError("The eps coefficient must be a positive float.")
        
        if not isinstance(self.targeted, (int)) or (self.targeted != 0 and self.targeted != 1):
            raise ValueError('`targeted` has to be a logical value.')

        if self.batch_size <= 0:
            raise ValueError('The batch size `batch_size` has to be positive.')

    def _block_order(self, img_size, channels, initial_size=2, stride=1):
        """
        Defines a block order, starting with top-left (initial_size x initial_size) submatrix
        expanding by stride rows and columns whenever exhausted
        randomized within the block and across channels.
        e.g. (initial_size=2, stride=1)
        [1, 3, 6]
        [2, 4, 9]
        [5, 7, 8]

        :param img_size: image size (i.e., width or height).
        :param channels: the number of channels.
        :param initial size: initial size for submatrix.
        :param stride: stride size for expansion.

        :return z: An array holding the block order of DCT attacks.
        """
        order = np.zeros((channels, img_size, img_size)).astype(ART_NUMPY_DTYPE)
        total_elems = channels * initial_size * initial_size
        perm = np.random.permutation(total_elems)
        order[:, :initial_size, :initial_size] = perm.reshape((channels, initial_size, initial_size))
        for i in range(initial_size, img_size, stride):
            num_elems = channels * (2 * stride * i + stride * stride)
            perm = np.random.permutation(num_elems) + total_elems
            num_first = channels * stride * (stride + i)
            order[:, : (i + stride), i : (i + stride)] = perm[:num_first].reshape((channels, -1, stride))
            order[:, i : (i + stride), :i] = perm[num_first:].reshape((channels, stride, -1))
            total_elems += num_elems
        if self.estimator.channels_first:
            return order.reshape(1, -1).squeeze().argsort()
        else:
            return order.transpose(1, 2, 0).reshape(1, -1).squeeze().argsort()

    def _block_idct(self, x, block_size=8, masked=False, ratio=0.5):
        """
        Applies IDCT to each block of size block_size.

        :param x: An array with the inputs to be attacked.
        :param block_size: block size for DCT attacks.
        :param masked: use the mask.
        :param ratio: Ratio of the lowest frequency directions in order to make the adversarial perturbation in the low
                      frequency space.

        :return z: An array holding the order of DCT attacks.
        """
        if not self.estimator.channels_first:
            x = x.transpose(0, 3, 1, 2)
        z = np.zeros(x.shape).astype(ART_NUMPY_DTYPE)
        num_blocks = int(x.shape[2] / block_size)
        mask = np.zeros((x.shape[0], x.shape[1], block_size, block_size))
        if type(ratio) != float:
            for i in range(x.shape[0]):
                mask[i, :, : int(block_size * ratio[i]), : int(block_size * ratio[i])] = 1
        else:
            mask[:, :, : int(block_size * ratio), : int(block_size * ratio)] = 1
        for i in range(num_blocks):
            for j in range(num_blocks):
                submat = x[:, :, (i * block_size) : ((i + 1) * block_size), (j * block_size) : ((j + 1) * block_size)]
                if masked:
                    submat = submat * mask
                z[:, :, (i * block_size) : ((i + 1) * block_size), (j * block_size) : ((j + 1) * block_size)] = idct(
                    idct(submat, axis=3, norm="ortho"), axis=2, norm="ortho"
                )

        if self.estimator.channels_first:
            return z
        else:
            return z.transpose(0, 2, 3, 1)

    def diagonal_order(self, image_size, channels):
        """
        Defines a diagonal order for pixel attacks.
        order is fixed across diagonals but are randomized across channels and within the diagonal
        e.g.
        [1, 2, 5]
        [3, 4, 8]
        [6, 7, 9]

        :param image_size: image size (i.e., width or height)
        :param channels: the number of channels

        :return z: An array holding the diagonal order of pixel attacks.
        """
        x = np.arange(0, image_size).cumsum()
        order = np.zeros((image_size, image_size)).astype(ART_NUMPY_DTYPE)
        for i in range(image_size):
            order[i, : (image_size - i)] = i + x[i:]
        for i in range(1, image_size):
            reverse = order[image_size - i - 1].take([i for i in range(i - 1, -1, -1)])
            order[i, (image_size - i) :] = image_size * image_size - 1 - reverse
        if channels > 1:
            order_2d = order
            order = np.zeros((channels, image_size, image_size))
            for i in range(channels):
                order[i, :, :] = 3 * order_2d + i

        if self.estimator.channels_first:
            return order.reshape(1, -1).squeeze().argsort()
        else:
            return order.transpose(1, 2, 0).reshape(1, -1).squeeze().argsort()
