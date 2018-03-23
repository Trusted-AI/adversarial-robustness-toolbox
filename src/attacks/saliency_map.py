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

from keras import backend as k
import numpy as np
import tensorflow as tf

from src.attacks.attack import Attack, class_derivative


class SaliencyMapMethod(Attack):
    """
    Implementation of the Jacobian-based Saliency Map Attack (Papernot et al. 2016).
    Paper link: https://arxiv.org/pdf/1511.07528.pdf
    """
    attack_params = ['theta', 'gamma', 'clip_min', 'clip_max']

    def __init__(self, model, sess=None, theta=0.1, gamma=1., clip_min=0., clip_max=1.):
        """
        Create a SaliencyMapMethod instance.

        Attack-specific parameters:
        :param theta: (optional float) Perturbation introduced to each modified feature (can be positive or negative)
        :param gamma: (optional float) Maximum percentage of perturbed features (between 0 and 1)
        :param clip_min: (optional float) Minimum component value for clipping
        """
        super(SaliencyMapMethod, self).__init__(model, sess)
        kwargs = {'theta': theta,
                  'gamma': gamma,
                  'clip_min': clip_min,
                  'clip_max': clip_max}
        self.set_params(**kwargs)

    def generate(self, x_val, **kwargs):
        """
        Generate adversarial samples and return them in a Numpy array.

        :param x_val: (required) A Numpy array with the original inputs.
        :param y_val: (optional) Target values if the attack is targeted
        :param theta: (optional float) Perturbation introduced to each modified feature (can be positive or negative)
        :param gamma: (optional float) Maximum percentage of perturbed features (between 0 and 1)
        :param clip_min: (optional float) Minimum component value for clipping
        :param clip_max: (optional float) Maximum component value for clipping
        :return: A Numpy array holding the adversarial examples.
        """
        # Parse and save attack-specific parameters
        assert self.set_params(**kwargs)
        k.set_learning_phase(0)

        # Initialize variables
        dims = [None] + list(x_val.shape[1:])
        self._x = tf.placeholder(tf.float32, shape=dims)
        dims[0] = 1
        x_adv = np.copy(x_val)
        self._nb_features = np.product(x_adv.shape[1:])
        self._nb_classes = self.model.output_shape[1]
        x_adv = np.reshape(x_adv, (-1, self._nb_features))
        preds = self.sess.run(tf.argmax(self.classifier.model(self._x), axis=1), {self._x: x_val})

        loss = self.classifier._get_predictions(self._x, log=False)
        self._grads = class_derivative(loss, self._x, self._nb_classes)

        # Set number of iterations w.r.t. the total perturbation allowed
        max_iter = np.floor(self._nb_features * self.gamma / 2)

        # Determine target classes for attack
        if 'y_val' not in kwargs or kwargs[str('y_val')] is None:
            # Randomly choose target from the incorrect classes for each sample
            from src.utils import random_targets
            targets = np.argmax(random_targets(preds, self._nb_classes), axis=1)
        else:
            targets = kwargs[str('y_val')]

        # Generate the adversarial samples
        for ind, val in enumerate(x_adv):
            # Initialize the search space; optimize to remove features that can't be changed
            if self.theta > 0:
                search_space = set([i for i in range(self._nb_features) if val[i] < self.clip_max])
            else:
                search_space = set([i for i in range(self._nb_features) if val[i] > self.clip_min])

            nb_iter = 0
            current_pred = preds[ind]

            while current_pred != targets[ind] and nb_iter < max_iter and bool(search_space):
                # Compute saliency map
                feat1, feat2 = self._saliency_map(np.reshape(val, dims), targets[ind], search_space)

                # Move on to next examples if there are no more features to change
                if feat1 == feat2 == 0:
                    break

                # Prepare update
                if self.theta > 0:
                    clip_func, clip_value = np.minimum, self.clip_max
                else:
                    clip_func, clip_value = np.maximum, self.clip_min

                # Update adversarial example
                for feature_ind in [feat1, feat2]:
                    # unraveled_ind = np.unravel_index(feature_ind, dims)
                    val[feature_ind] = clip_func(clip_value, val[feature_ind] + self.theta)

                    # Remove indices from search space if max/min values were reached
                    if val[feature_ind] == clip_value:
                        search_space.discard(feature_ind)

                # Recompute model prediction
                current_pred = self.sess.run(tf.argmax(self.classifier.model(self._x), axis=1),
                                             {self._x: np.reshape(val, dims)})
                nb_iter += 1

        x_adv = np.reshape(x_adv, x_val.shape)
        return x_adv

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.

        Attack-specific parameters:
        :param theta: (optional float) Perturbation introduced to modified components (can be positive or negative)
        :param gamma: (optional float) Maximum percentage of perturbed features
        :param nb_classes: (optional int) Number of model output classes
        :param clip_min: (optional float) Minimum component value for clipping
        :param clip_max: (optional float) Maximum component value for clipping
        """
        # Save attack-specific parameters
        super(SaliencyMapMethod, self).set_params(**kwargs)

        if self.gamma < 0 or self.gamma > 1:
            raise ValueError("The total perturbation must be between 0 and 1.")

        return True

    def _saliency_map(self, x, target, search_space):
        """
        Compute the saliency map of `x`. Return the top 2 coefficients in `search_space` that maximize / minimize
        the saliency map.

        :param x: (np.ndarray) One input sample
        :param target: Target class for `x`
        :param search_space: (set) The set of valid pairs of feature indices to search
        :return: (tuple) Tuple of the two indices to be changed; each represents a feature from the flattened input
        """
        grads_val = self.sess.run(self._grads, feed_dict={self._x: x})
        grads_val = np.array([np.reshape(g[0], self._nb_features) for g in grads_val])

        # Compute grads for target class and sum of gradients for all other classes
        grads_target = grads_val[target]
        other_mask = list(range(self._nb_classes))
        other_mask.remove(target)
        grads_others = np.sum(grads_val[other_mask, :], axis=0)

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
