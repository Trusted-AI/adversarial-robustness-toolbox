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
This module implements the Jacobian-based Saliency Map attack `SaliencyMapMethod`. This is a white-box attack.

| Paper link: https://arxiv.org/abs/1511.07528
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np

from art import NUMPY_DTYPE
from art.classifiers.classifier import ClassifierGradients
from art.attacks.attack import Attack
from art.utils import check_and_transform_label_format

logger = logging.getLogger(__name__)


class SaliencyMapMethod(Attack):
    """
    Implementation of the Jacobian-based Saliency Map Attack (Papernot et al. 2016).

    | Paper link: https://arxiv.org/abs/1511.07528
    """
    attack_params = Attack.attack_params + ['theta', 'gamma', 'batch_size']

    def __init__(self, classifier, theta=0.1, gamma=1., batch_size=1):
        """
        Create a SaliencyMapMethod instance.

        :param classifier: A trained classifier.
        :type classifier: :class:`.Classifier`
        :param theta: Amount of Perturbation introduced to each modified feature per step (can be positive or negative).
        :type theta: `float`
        :param gamma: Maximum fraction of features being perturbed (between 0 and 1).
        :type gamma: `float`
        :param batch_size: Size of the batch on which adversarial samples are generated.
        :type batch_size: `int`
        """
        super(SaliencyMapMethod, self).__init__(classifier)
        if not isinstance(classifier, ClassifierGradients):
            raise (TypeError('For `' + self.__class__.__name__ + '` classifier must be an instance of '
                             '`art.classifiers.classifier.ClassifierGradients`, the provided classifier is instance of '
                             + str(classifier.__class__.__bases__) + '.'))

        kwargs = {'theta': theta, 'gamma': gamma, 'batch_size': batch_size}
        self.set_params(**kwargs)

    def generate(self, x, y=None, **kwargs):
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs to be attacked.
        :type x: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  `(nb_samples,)`.
        :type y: `np.ndarray`
        :return: An array holding the adversarial examples.
        :rtype: `np.ndarray`
        """
        y = check_and_transform_label_format(y, self.classifier.nb_classes())

        # Initialize variables
        dims = list(x.shape[1:])
        self._nb_features = np.product(dims)
        x_adv = np.reshape(x.astype(NUMPY_DTYPE), (-1, self._nb_features))
        preds = np.argmax(self.classifier.predict(x, batch_size=self.batch_size), axis=1)

        # Determine target classes for attack
        if y is None:
            # Randomly choose target from the incorrect classes for each sample
            from art.utils import random_targets
            targets = np.argmax(random_targets(preds, self.classifier.nb_classes()), axis=1)
        else:
            targets = np.argmax(y, axis=1)

        # Compute perturbation with implicit batching
        for batch_id in range(int(np.ceil(x_adv.shape[0] / float(self.batch_size)))):
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            batch = x_adv[batch_index_1:batch_index_2]

            # Main algorithm for each batch
            # Initialize the search space; optimize to remove features that can't be changed
            search_space = np.zeros(batch.shape)
            if hasattr(self.classifier, 'clip_values') and self.classifier.clip_values is not None:
                clip_min, clip_max = self.classifier.clip_values
                if self.theta > 0:
                    search_space[batch < clip_max] = 1
                else:
                    search_space[batch > clip_min] = 1

            # Get current predictions
            current_pred = preds[batch_index_1:batch_index_2]
            target = targets[batch_index_1:batch_index_2]
            active_indices = np.where(current_pred != target)[0]
            all_feat = np.zeros_like(batch)

            while active_indices.size != 0:
                # Compute saliency map
                feat_ind = self._saliency_map(np.reshape(batch, [batch.shape[0]] + dims)[active_indices],
                                              target[active_indices], search_space[active_indices])

                # Update used features
                all_feat[active_indices][np.arange(len(active_indices)), feat_ind[:, 0]] = 1
                all_feat[active_indices][np.arange(len(active_indices)), feat_ind[:, 1]] = 1

                # Apply attack with clipping
                if hasattr(self.classifier, 'clip_values') and self.classifier.clip_values is not None:
                    # Prepare update depending of theta
                    if self.theta > 0:
                        clip_func, clip_value = np.minimum, clip_max
                    else:
                        clip_func, clip_value = np.maximum, clip_min

                    # Update adversarial examples
                    tmp_batch = batch[active_indices]
                    tmp_batch[np.arange(len(active_indices)), feat_ind[:, 0]] = \
                        clip_func(clip_value, tmp_batch[np.arange(len(active_indices)), feat_ind[:, 0]] + self.theta)
                    tmp_batch[np.arange(len(active_indices)), feat_ind[:, 1]] = \
                        clip_func(clip_value, tmp_batch[np.arange(len(active_indices)), feat_ind[:, 1]] + self.theta)
                    batch[active_indices] = tmp_batch

                    # Remove indices from search space if max/min values were reached
                    search_space[batch == clip_value] = 0

                # Apply attack without clipping
                else:
                    tmp_batch = batch[active_indices]
                    tmp_batch[np.arange(len(active_indices)), feat_ind[:, 0]] += self.theta
                    tmp_batch[np.arange(len(active_indices)), feat_ind[:, 1]] += self.theta
                    batch[active_indices] = tmp_batch

                # Recompute model prediction
                current_pred = np.argmax(self.classifier.predict(np.reshape(batch, [batch.shape[0]] + dims)), axis=1)

                # Update active_indices
                active_indices = np.where((current_pred != target) *
                                          (np.sum(all_feat, axis=1) / self._nb_features <= self.gamma) *
                                          (np.sum(search_space, axis=1) > 0))[0]

            x_adv[batch_index_1:batch_index_2] = batch

        x_adv = np.reshape(x_adv, x.shape)
        logger.info('Success rate of JSMA attack: %.2f%%',
                    (np.sum(np.argmax(self.classifier.predict(x, batch_size=self.batch_size), axis=1) != np.argmax(
                        self.classifier.predict(x_adv, batch_size=self.batch_size), axis=1)) / x.shape[0]))

        return x_adv

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.

        :param theta: Perturbation introduced to each modified feature per step (can be positive or negative)
        :type theta: `float`
        :param gamma: Maximum percentage of perturbed features (between 0 and 1)
        :type gamma: `float`
        :param batch_size: Internal size of batches on which adversarial samples are generated.
        :type batch_size: `int`
        """
        # Save attack-specific parameters
        super(SaliencyMapMethod, self).set_params(**kwargs)

        if self.gamma <= 0 or self.gamma > 1:
            raise ValueError("The total perturbation percentage `gamma` must be between 0 and 1.")

        if self.batch_size <= 0:
            raise ValueError('The batch size `batch_size` has to be positive.')

        return True

    def _saliency_map(self, x, target, search_space):
        """
        Compute the saliency map of `x`. Return the top 2 coefficients in `search_space` that maximize / minimize
        the saliency map.

        :param x: A batch of input samples
        :type x: `np.ndarray`
        :param target: Target class for `x`
        :type target: `np.ndarray`
        :param search_space: The set of valid pairs of feature indices to search
        :type search_space: `np.ndarray`
        :return: The top 2 coefficients in `search_space` that maximize / minimize the saliency map
        :rtype: `np.ndarray`
        """
        grads = self.classifier.class_gradient(x, label=target)
        grads = np.reshape(grads, (-1, self._nb_features))

        # Remove gradients for already used features
        used_features = 1 - search_space
        coeff = 2 * int(self.theta > 0) - 1
        grads[used_features == 1] = -np.inf * coeff

        if self.theta > 0:
            ind = np.argpartition(grads, -2, axis=1)[:, -2:]
        else:
            ind = np.argpartition(-grads, -2, axis=1)[:, -2:]

        return ind
