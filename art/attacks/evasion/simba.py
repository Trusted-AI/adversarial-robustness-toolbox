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


class SimBA(EvasionAttack):
    attack_params = EvasionAttack.attack_params + ['attack', 'max_iter', 'epsilon', 'order', 'freq_dim', 'stride', 'targeted', 'batch_size',]

    def __init__(self, classifier, attack='dct', max_iter=3000, order='random', epsilon=0.1, freq_dim=4, stride=1, targeted=False, batch_size=1):
        """
        Create a SimBA (dct) attack instance.

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
        :param freq_dim: dimensionality of 2D frequency space (DCT).
        :type freq_dim: `int`
        :param stride: stride for block order (DCT).
        :type stride: `int`
        :param targeted: targeted attacks
        :type targeted: `bool`
        :param batch_size: Batch size (but, batch process unavailable in this implementation)
        :type batch_size: `int`
        """
        super(SimBA, self).__init__(classifier=classifier)
        if not isinstance(classifier, ClassifierGradients):
            raise (TypeError('For `' + self.__class__.__name__ + '` classifier must be an instance of '
                             '`art.classifiers.classifier.ClassifierGradients`, the provided classifier is instance of '
                             + str(classifier.__class__.__bases__) + '. '
                             ' The classifier needs to be a Neural Network and provide gradients.'))

        params = {'attack': attack, 'max_iter': max_iter, 'epsilon': epsilon, 'order': order, 'freq_dim': freq_dim, 'stride': stride, 'targeted': targeted ,'batch_size': batch_size}
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

        #Note: need further implementation for targeted attacks
        if y is None:
            if self.targeted == True:
                raise ValueError('Target labels `y` need to be provided for a targeted attack.')
            else:
                # Use model predictions as correct outputs
                logger.info('Using the model prediction as the correct label for SimBA.')
                y_i = np.argmax(preds, axis=1)
        else:
            y_i = np.argmax(y, axis=1)
        
        desired_label = y_i[0]
        current_label = np.argmax(preds, axis=1)[0]
        last_prob = preds.reshape(-1)[desired_label]

        n_dims = np.prod(x.shape)

        if self.attack == 'px':
            if self.order == 'diag':
                indices = self.diagonal_order(x.shape[2], 3)[:self.max_iter]
            elif self.order == 'random':
                indices = np.random.permutation(n_dims)[:self.max_iter]
            indices_size = len(indices)
            while indices_size < self.max_iter:
                if self.order == 'diag':
                    tmp_indices = self.diagonal_order(x.shape[2], 3)
                elif self.order == 'random':
                    tmp_indices = np.random.permutation(n_dims)
                indices = np.hstack((indices, tmp_indices))[:self.max_iter]
                indices_size = len(indices)
        elif self.attack == 'dct':
            indices = self._block_order(x.shape[2], 3, initial_size=self.freq_dim, stride=self.stride)[:self.max_iter]
            indices_size = len(indices)
            while indices_size < self.max_iter:
                tmp_indices = self._block_order(x.shape[2], 3, initial_size=self.freq_dim, stride=self.stride)
                indices = np.hstack((indices, tmp_indices))[:self.max_iter]
                indices_size = len(indices)
            trans = lambda z: self._block_idct(z, block_size=x.shape[2])

        clip_min = -np.inf
        clip_max = np.inf 
        if hasattr(self.classifier, 'clip_values') and self.classifier.clip_values is not None:
            clip_min, clip_max = self.classifier.clip_values

        term_flag = 1
        if self.targeted == True:
            if desired_label != current_label:
                term_flag = 0
        else:
            if desired_label == current_label:
                term_flag = 0

        nb_iter = 0
        while term_flag == 0 and nb_iter < self.max_iter:
            diff = np.zeros(n_dims)
            diff[indices[nb_iter]] = self.epsilon

            if self.attack == 'dct':
                left_preds = self.classifier.predict(np.clip(x - trans(diff.reshape(x.shape)), clip_min, clip_max), batch_size=self.batch_size)
            elif self.attack == 'px':
                left_preds = self.classifier.predict(np.clip(x - diff.reshape(x.shape), clip_min, clip_max), batch_size=self.batch_size)
            left_prob = left_preds.reshape(-1)[desired_label

            if self.attack == 'dct':
                right_preds = self.classifier.predict(np.clip(x + trans(diff.reshape(x.shape)), clip_min, clip_max), batch_size=self.batch_size)
            elif self.attack == 'px':
                right_preds = self.classifier.predict(np.clip(x + diff.reshape(x.shape), clip_min, clip_max), batch_size=self.batch_size)
            right_prob = right_preds.reshape(-1)[desired_label]

            if self.targeted == True:
                if left_prob > last_prob:
                    if left_prob > right_prob:
                        if self.attack == 'dct':
                            x = np.clip(x - trans(diff.reshape(x.shape)), clip_min, clip_max)
                        elif self.attack == 'px':
                            x = np.clip(x - diff.reshape(x.shape), clip_min, clip_max)
                        last_prob = left_prob
                        current_label = np.argmax(left_preds, axis=1)[0]
                    else:
                        if self.attack == 'dct':
                            x = np.clip(x + trans(diff.reshape(x.shape)), clip_min, clip_max)
                        elif self.attack == 'px':
                            x = np.clip(x + diff.reshape(x.shape), clip_min, clip_max)
                        last_prob = right_prob
                        current_label = np.argmax(right_preds, axis=1)[0]
                else:
                    if right_prob > last_prob:
                        if self.attack == 'dct':
                            x = np.clip(x + trans(diff.reshape(x.shape)), clip_min, clip_max)
                        elif self.attack == 'px':
                            x = np.clip(x + diff.reshape(x.shape), clip_min, clip_max)
                        last_prob = right_prob
                        current_label = np.argmax(right_preds, axis=1)[0]
            else:
                if left_prob < last_prob:
                    if left_prob < right_prob:
                        if self.attack == 'dct':
                            x = np.clip(x - trans(diff.reshape(x.shape)), clip_min, clip_max)
                        elif self.attack == 'px':
                            x = np.clip(x - diff.reshape(x.shape), clip_min, clip_max)
                        last_prob = left_prob
                        current_label = np.argmax(left_preds, axis=1)[0]
                    else:
                        if self.attack == 'dct':
                            x = np.clip(x + trans(diff.reshape(x.shape)), clip_min, clip_max)
                        elif self.attack == 'px':
                            x = np.clip(x + diff.reshape(x.shape), clip_min, clip_max)
                        last_prob = right_prob
                        current_label = np.argmax(right_preds, axis=1)[0]
                else:
                    if right_prob < last_prob:
                        if self.attack == 'dct':
                            x = np.clip(x + trans(diff.reshape(x.shape)), clip_min, clip_max)
                        elif self.attack == 'px':
                            x = np.clip(x + diff.reshape(x.shape), clip_min, clip_max)
                        last_prob = right_prob
                        current_label = np.argmax(right_preds, axis=1)[0]
            
            if self.targeted == True:
                if desired_label == current_label:
                    term_flag = 1
            else:
                if desired_label != current_label:
                    term_flag = 1

            nb_iter = nb_iter + 1

        if nb_iter < self.max_iter:
            logger.info('SimBA (%s) %s attack succeed', self.attack, ['non-targeted', 'targeted'][self.targeted])
        else:
            logger.info('SimBA (%s) %s attack failed', self.attack, ['non-targeted', 'targeted'][self.targeted])

        return x

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.

        :param attack: attack type: pixel (px) or DCT (dct) attacks
        :type attack: `str`
        :param max_iter: The maximum number of iterations.
        :type max_iter: `int`
        :param epsilon: Overshoot parameter.
        :type epsilon: `float`
        :param order: order of pixel attacks
        :type order: `str`
        :param freq_dim: dimensionality of 2D frequency space (DCT).
        :type freq_dim: `int`
        :param stride: stride for block order (DCT).
        :type stride: `int`
        :param targeted: targeted attacks
        :type targeted: `bool`
        :param batch_size: Batch size (but, batch process unavailable in this implementation)
        :type batch_size: `int`
        """
        # Save attack-specific parameters
        super(SimBA, self).set_params(**kwargs)

        if not isinstance(self.max_iter, (int, np.int)) or self.max_iter <= 0:
            raise ValueError("The number of iterations must be a positive integer.")

        if self.epsilon < 0:
            raise ValueError("The overshoot parameter must not be negative.")

        if self.batch_size <= 0:
            raise ValueError('The batch size `batch_size` has to be positive.')
        
        if not isinstance(self.stride, (int, np.int)) or self.stride <= 0:
            raise ValueError("The `stride` value must be a positive integer.")
        
        if not isinstance(self.freq_dim, (int, np.int)) or self.freq_dim <= 0:
            raise ValueError("The `freq_dim` value must be a positive integer.")
        
        if self.order != 'random' and self.order != 'diag':
            raise ValueError('The order of pixel attacks has to be `random` or `diag`.')
        
        if self.attack != 'px' and self.attack != 'dct':
            raise ValueError('The attack type has to be `px` or `dct`.')
        
        if not isinstance(self.targeted, (int)) or (self.targeted != 0 and self.targeted != 1):
            raise ValueError('`targeted` has to be a logical value.')

        return True

    def _block_order(self, img_size, channels, initial_size=2, stride=1):
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
        return order.transpose(1,2,0).reshape(1, -1).squeeze().argsort()

    # applies IDCT to each block of size block_size
    def _block_idct(self, x, block_size=8, masked=False, ratio=0.5):
        x = x.transpose(0,3,1,2)
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
        return z.transpose(0,2,3,1)
    
    def diagonal_order(self, image_size, channels):
        x = np.arange(0, image_size).cumsum()
        order = np.zeros((image_size, image_size))
        for i in range(image_size):
            order[i, :(image_size - i)] = i + x[i:]
        for i in range(1, image_size):
            reverse = order[image_size - i - 1].take([i for i in range(i-1, -1, -1)])
            order[i, (image_size - i):] = image_size * image_size - 1 - reverse
        if channels > 1:
            order_2d = order
            order = np.zeros((channels, image_size, image_size))
            for i in range(channels):
                order[i, :, :] = 3 * order_2d + i
        return order.transpose(1,2,0).reshape(1, -1).squeeze().argsort()
