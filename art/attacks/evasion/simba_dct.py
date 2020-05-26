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
This module implements the black-box attack `simba`.

| Paper link: https://arxiv.org/abs/1905.07121
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging

import numpy as np
from scipy.fftpack import dct, idct

from art.attacks.attack import EvasionAttack
from art.classifiers.classifier import ClassifierGradients
from art.config import ART_NUMPY_DTYPE
from art.utils import compute_success

logger = logging.getLogger(__name__)


class SimBA_dct(EvasionAttack):
    attack_params = EvasionAttack.attack_params + ['max_iter', 'epsilon', 'batch_size', 'freq_dim', 'stride']

    def __init__(self, classifier, max_iter=3000, epsilon=0.1, freq_dim=4, stride=1, batch_size=1):
        """
        Create a SimBA (dct) attack instance.

        :param classifier: A trained classifier.
        :type classifier: :class:`.Classifier`
        :param max_iter: The maximum number of iterations.
        :type max_iter: `int`
        :param epsilon: Overshoot parameter.
        :type epsilon: `float`
        :param batch_size: Batch size (but, batch process unavailable in this implementation)
        :type batch_size: `int`
        """
        super(SimBA_dct, self).__init__(classifier=classifier)
        if not isinstance(classifier, ClassifierGradients):
            raise (TypeError('For `' + self.__class__.__name__ + '` classifier must be an instance of '
                             '`art.classifiers.classifier.ClassifierGradients`, the provided classifier is instance of '
                             + str(classifier.__class__.__bases__) + '. '
                             ' The classifier needs to be a Neural Network and provide gradients.'))

        params = {'max_iter': max_iter, 'epsilon': epsilon, 'freq_dim': freq_dim, 'stride': stride, 'batch_size': batch_size}
        self.set_params(**params)

    def generate(self, x, y=None, **kwargs):
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
        preds = self.classifier.predict(x, batch_size=self.batch_size)
        if y is None:
            y = np.argmax(preds, axis=1)
        original_label = y[0]
        current_label = original_label
        last_prob = preds.reshape(-1)[original_label]

        n_dims = np.prod(x.shape)
        if self.max_iter > n_dims:
            self.max_iter = n_dims
            logger.info('`max_iter` was reset to %d because it needs to be #pixels x #channels or less', n_dims)

        indices = self._block_order(x.shape[2], 3, initial_size=self.freq_dim, stride=self.stride)[:self.max_iter]
        trans = lambda z: self._block_idct(z, block_size=x.shape[2])

        clip_min = -np.inf
        clip_max = np.inf 
        if hasattr(self.classifier, 'clip_values') and self.classifier.clip_values is not None:
            clip_min, clip_max = self.classifier.clip_values

        nb_iter = 0
        while original_label == current_label and nb_iter < self.max_iter:
            diff = np.zeros(n_dims)
            diff[indices[nb_iter]] = self.epsilon

            left_preds = self.classifier.predict(np.clip(x - trans(diff.reshape(x.shape)), clip_min, clip_max), batch_size=self.batch_size)
            left_prob = left_preds.reshape(-1)[original_label]

            right_preds = self.classifier.predict(np.clip(x + trans(diff.reshape(x.shape)), clip_min, clip_max), batch_size=self.batch_size)
            right_prob = right_preds.reshape(-1)[original_label]

            if left_prob < last_prob:
                if left_prob < right_prob:
                    x = np.clip(x - trans(diff.reshape(x.shape)), clip_min, clip_max)
                    last_prob = left_prob
                    current_label = np.argmax(left_preds, axis=1)[0]
                else:
                    x = np.clip(x + trans(diff.reshape(x.shape)), clip_min, clip_max)
                    last_prob = right_prob
                    current_label = np.argmax(right_preds, axis=1)[0]
            else:
                if right_prob < last_prob:
                    x = np.clip(x + trans(diff.reshape(x.shape)), clip_min, clip_max)
                    last_prob = right_prob
                    current_label = np.argmax(right_preds, axis=1)[0]
            
            nb_iter = nb_iter + 1

        if nb_iter < self.max_iter:
            logger.info('SimBA (dct) attack succeed')
        else:
            logger.info('SimBA (dct) attack failed')

        return x

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.

        :param max_iter: The maximum number of iterations.
        :type max_iter: `int`
        :param epsilon: Overshoot parameter.
        :type epsilon: `float`
        :param batch_size: Internal size of batches on which adversarial samples are generated.
        :type batch_size: `int`
        """
        # Save attack-specific parameters
        super(SimBA_dct, self).set_params(**kwargs)

        if not isinstance(self.max_iter, (int, np.int)) or self.max_iter <= 0:
            raise ValueError("The number of iterations must be a positive integer.")

        if self.epsilon < 0:
            raise ValueError("The overshoot parameter must not be negative.")

        if self.batch_size <= 0:
            raise ValueError('The batch size `batch_size` has to be positive.')

        return True

    def _block_order(self, img_size, channels, initial_size=28, stride=7):
        order = np.zeros((channels , img_size , img_size))
        total_elems = channels * initial_size * initial_size
        perm = np.random.permutation(total_elems)
        order[:, :initial_size, :initial_size] = perm.reshape((channels, initial_size, initial_size))
        for i in range(initial_size, img_size, stride):
            num_elems = channels * (2 * stride * i + stride * stride)
            perm = np.random.permutation(num_elems) + total_elems
            num_first = channels * stride * (stride + i)
            order[:, :(i+stride), i:(i+stride)] = perm[:num_first].reshape((channels, -1, stride))
            order[:, i:(i+stride), :i] = perm[num_first:].reshape((channels, stride, -1))
            total_elems += num_elems
        return order.reshape(1, -1).squeeze().argsort()

    # applies IDCT to each block of size block_size
    def _block_idct(self, x, block_size=8, masked=False, ratio=0.5):
        z = np.zeros(x.shape)
        num_blocks = int(x.shape[2] / block_size)
        mask = np.zeros((x.shape[0], x.shape[1], block_size, block_size))
        if type(ratio) != float:
            for i in range(x.shape[0]):
                mask[i, :, :int(block_size * ratio[i]), :int(block_size * ratio[i])] = 1
        else:
            mask[:, :, :int(block_size * ratio), :int(block_size * ratio)] = 1
        for i in range(num_blocks):
            for j in range(num_blocks):
                submat = x[:, :, (i * block_size):((i + 1) * block_size), (j * block_size):((j + 1) * block_size)]
                if masked:
                    submat = submat * mask
                z[:, :, (i * block_size):((i + 1) * block_size), (j * block_size):((j + 1) * block_size)] = idct(idct(submat, axis=3, norm='ortho'), axis=2, norm='ortho')
        return z
