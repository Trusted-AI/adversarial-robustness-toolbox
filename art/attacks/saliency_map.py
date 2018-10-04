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
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np

from art.attacks.attack import Attack

logger = logging.getLogger(__name__)


class SaliencyMapMethod(Attack):
    """
    Implementation of the Jacobian-based Saliency Map Attack (Papernot et al. 2016).
    Paper link: https://arxiv.org/pdf/1511.07528.pdf
    """
    attack_params = Attack.attack_params + ['theta', 'gamma']

    # TODO Add parameter logits?
    def __init__(self, classifier, theta=0.1, gamma=1.):
        """
        Create a SaliencyMapMethod instance.

        :param classifier: A trained model.
        :type classifier: :class:`Classifier`
        :param theta: Perturbation introduced to each modified feature per step (can be positive or negative).
        :type theta: `float`
        :param gamma: Maximum percentage of perturbed features (between 0 and 1).
        :type gamma: `float`
        """
        super(SaliencyMapMethod, self).__init__(classifier)
        kwargs = {
            'theta': theta,
            'gamma': gamma
            }
        self.set_params(**kwargs)

    def generate(self, x, **kwargs):
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs to be attacked.
        :type x: `np.ndarray`
        :param y: Target values if the attack is targeted
        :type y: `np.ndarray`
        :param theta: Perturbation introduced to each modified feature per step (can be positive or negative)
        :type theta: `float`
        :param gamma: Maximum percentage of perturbed features (between 0 and 1)
        :type gamma: `float`
        :return: An array holding the adversarial examples.
        :rtype: `np.ndarray`
        """
        # Parse and save attack-specific parameters
        self.set_params(**kwargs)
        clip_min, clip_max = self.classifier.clip_values

        # Initialize variables
        dims = [1] + list(x.shape[1:])
        self._nb_features = np.product(dims)
        x_adv = np.reshape(np.copy(x), (-1, self._nb_features))
        preds = np.argmax(self.classifier.predict(x), axis=1)

        # Determine target classes for attack
        if 'y' not in kwargs or kwargs[str('y')] is None:
            # Randomly choose target from the incorrect classes for each sample
            from art.utils import random_targets
            targets = np.argmax(random_targets(preds, self.classifier.nb_classes), axis=1)
        else:
            targets = np.argmax(kwargs[str('y')], axis=1)

        # Generate the adversarial samples
        for ind, val in enumerate(x_adv):
            # Initialize the search space; optimize to remove features that can't be changed
            if self.theta > 0:
                search_space = set([i for i in range(self._nb_features) if val[i] < clip_max])
            else:
                search_space = set([i for i in range(self._nb_features) if val[i] > clip_min])

            current_pred = preds[ind]
            target = targets[ind]
            all_feat = set()

            while current_pred != target and len(all_feat) / self._nb_features <= self.gamma and bool(search_space):
                # Compute saliency map
                feat1, feat2 = self._saliency_map(np.reshape(val, dims), target, search_space)

                # Move on to next examples if there are no more features to change
                if feat1 == feat2 == 0:
                    break

                all_feat = all_feat.union(set([feat1, feat2]))

                # Prepare update depending of theta
                if self.theta > 0:
                    clip_func, clip_value = np.minimum, clip_max
                else:
                    clip_func, clip_value = np.maximum, clip_min

                # Update adversarial example
                for feature_ind in [feat1, feat2]:
                    val[feature_ind] = clip_func(clip_value, val[feature_ind] + self.theta)

                    # Remove indices from search space if max/min values were reached
                    if val[feature_ind] == clip_value:
                        search_space.discard(feature_ind)

                # Recompute model prediction
                current_pred = np.argmax(self.classifier.predict(np.reshape(val, dims)), axis=1)

        x_adv = np.reshape(x_adv, x.shape)
        preds = np.argmax(self.classifier.predict(x), axis=1)
        preds_adv = np.argmax(self.classifier.predict(x_adv), axis=1)
        logger.info('Success rate of JSMA attack: %.2f%%', (np.sum(preds != preds_adv) / x.shape[0]))

        return x_adv

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.

        :param theta: Perturbation introduced to each modified feature per step (can be positive or negative)
        :type theta: `float`
        :param gamma: Maximum percentage of perturbed features (between 0 and 1)
        :type gamma: `float`
        """
        # Save attack-specific parameters
        super(SaliencyMapMethod, self).set_params(**kwargs)

        if self.gamma <= 0 or self.gamma > 1:
            raise ValueError("The total perturbation percentage `gamma` must be between 0 and 1.")

        return True

    def _saliency_map(self, x, target, search_space):
        """
        Compute the saliency map of `x`. Return the top 2 coefficients in `search_space` that maximize / minimize
        the saliency map.

        :param x: One input sample
        :type x: `np.ndarray`
        :param target: Target class for `x`
        :type target: `int`
        :param search_space: The set of valid pairs of feature indices to search
        :type search_space: `set(tuple)`
        :return: The top 2 coefficients in `search_space` that maximize / minimize the saliency map
        :rtype: `tuple`
        """
        grads = self.classifier.class_gradient(x, label=target, logits=False)
        grads = np.reshape(grads, (-1, self._nb_features))[0]

        # Remove gradients for already used features
        used_features = list(set(range(self._nb_features)) - search_space)
        coeff = 2 * int(self.theta > 0) - 1
        grads[used_features] = -np.inf * coeff

        if self.theta > 0:
            ind = np.argpartition(grads, -2)[-2:]
        else:
            ind = np.argpartition(-grads, -2)[-2:]

        return tuple(ind)

    def _saliency_map_logits(self, x, target, search_space):
        """
        Compute the saliency map of `x`. Return the top 2 coefficients in `search_space` that maximize / minimize
        the saliency map.

        :param x: One input sample
        :type x: `np.ndarray`
        :param target: Target class for `x`
        :type target: `int`
        :param search_space: The set of valid pairs of feature indices to search
        :type search_space: `set(tuple)`
        :return: The top 2 coefficients in `search_space` that maximize / minimize the saliency map
        :rtype: `tuple`
        """
        grads = self.classifier.class_gradient(x, logits=True)
        grads = np.reshape(grads, (-1, self._nb_features))

        # Compute grads for target class and sum of gradients for all other classes
        grads_target = grads[target]
        other_mask = list(range(self.classifier.nb_classes))
        other_mask.remove(target)
        grads_others = np.sum(grads[other_mask, :], axis=0)

        # Remove gradients for already used features
        used_features = list(set(range(self._nb_features)) - search_space)
        coeff = 2 * int(self.theta > 0) - 1
        grads_target[used_features] = - np.max(np.abs(grads_target)) * coeff
        grads_others[used_features] = np.max(np.abs(grads_others)) * coeff

        # Precompute all pairs of sums of gradients and cache
        sums_target = grads_target.reshape((1, self._nb_features)) + grads_target.reshape((self._nb_features, 1))
        sums_others = grads_others.reshape((1, self._nb_features)) + grads_others.reshape((self._nb_features, 1))

        if self.theta > 0:
            mask = (sums_target > 0) & (sums_others < 0)
        else:
            mask = (sums_target < 0) & (sums_others > 0)
        scores = mask * (-sums_target * sums_others)
        np.fill_diagonal(scores, 0)

        # Choose top 2 features
        best_pair = np.argmax(scores)
        ind1, ind2 = best_pair % self._nb_features, best_pair // self._nb_features

        return ind1, ind2
