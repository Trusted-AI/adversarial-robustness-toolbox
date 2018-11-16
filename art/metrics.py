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
Module implementing varying metrics for assessing model robustness. These fall mainly under two categories:
attack-dependent and attack-independent.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from functools import reduce

import numpy as np
import numpy.linalg as la
from scipy.optimize import fmin as scipy_optimizer
from scipy.stats import weibull_min

from art.attacks import FastGradientMethod
from art.utils import random_sphere
from art import NUMPY_DTYPE

logger = logging.getLogger(__name__)

supported_methods = {
    "fgsm": {"class": FastGradientMethod, "params": {"eps_step": 0.1, "eps_max": 1., "clip_min": 0., "clip_max": 1.}},
    # "jsma": {"class": SaliencyMapMethod, "params": {"theta": 1., "gamma": 0.01, "clip_min": 0., "clip_max": 1.}}
    }


def get_crafter(classifier, attack, params=None):
    try:
        crafter = supported_methods[attack]["class"](classifier)
    except:
        raise NotImplementedError("{} crafting method not supported.".format(attack))

    if params:
        crafter.set_params(**params)

    return crafter


def empirical_robustness(classifier, x, attack_name, attack_params=None):
    """Compute the Empirical Robustness of a classifier object over the sample `x` for a given adversarial crafting
    method `attack`. This is equivalent to computing the minimal perturbation that the attacker must introduce for a
    successful attack. Paper link: https://arxiv.org/abs/1511.04599

    :param classifier: A trained model
    :type classifier: :class:`Classifier`
    :param x: Data sample of shape that can be fed into `classifier`
    :type x: `np.ndarray`
    :param attack_name: adversarial attack name
    :type attack_name: `str`
    :param attack_params: Parameters specific to the adversarial attack
    :type attack_params: `dict`
    :return: The average empirical robustness computed on `x`
    :rtype: `float`
    """
    crafter = get_crafter(classifier, attack_name, attack_params)
    attack_params['minimal'] = True
    adv_x = crafter.generate(x, **attack_params)

    # Predict the labels for adversarial examples
    y = classifier.predict(x)
    y_pred = classifier.predict(adv_x)

    idxs = (np.argmax(y_pred, axis=1) != np.argmax(y, axis=1))
    if np.sum(idxs) == 0.0:
        return 0

    norm_type = 2
    if hasattr(crafter, 'norm'):
        norm_type = crafter.norm
    perts_norm = la.norm((adv_x - x).reshape(x.shape[0], -1), ord=norm_type, axis=1)
    perts_norm = perts_norm[idxs]

    return np.mean(perts_norm / la.norm(x[idxs].reshape(np.sum(idxs), -1), ord=norm_type, axis=1))


# def nearest_neighbour_dist(classifier, x, x_ref, attack_name, attack_params=None):
#     """
#     Compute the (average) nearest neighbour distance between the sets `x` and `x_train`: for each point in `x`,
#     measure the Euclidean distance to its closest point in `x_train`, then average over all points.
#
#     :param classifier: A trained model
#     :type classifier: :class:`Classifier`
#     :param x: Data sample of shape that can be fed into `classifier`
#     :type x: `np.ndarray`
#     :param x_ref: Reference data sample to be considered as neighbors
#     :type x_ref: `np.ndarray`
#     :param attack_name: adversarial attack name
#     :type attack_name: `str`
#     :param attack_params: Parameters specific to the adversarial attack
#     :type attack_params: `dict`
#     :return: The average nearest neighbors distance
#     :rtype: `float`
#     """
#     # Craft the adversarial examples
#     crafter = get_crafter(classifier, attack_name, attack_params)
#     adv_x = crafter.generate(x, minimal=True)
#
#     # Predict the labels for adversarial examples
#     y = classifier.predict(x)
#     y_pred = classifier.predict(adv_x)
#
#     adv_x_ = adv_x.reshape(adv_x.shape[0], np.prod(adv_x.shape[1:]))
#     x_ = x_ref.reshape(x_ref.shape[0], np.prod(x_ref.shape[1:]))
#     dists = la.norm(adv_x_ - x_, axis=1)
#
#     # TODO check if following computation is correct ?
#     dists = np.min(dists, axis=1) / la.norm(x.reshape(x.shape[0], -1), ord=2, axis=1)
#     idxs = (np.argmax(y_pred, axis=1) != np.argmax(y, axis=1))
#     avg_nn_dist = np.mean(dists[idxs])
#
#     return avg_nn_dist


def loss_sensitivity(classifier, x, y):
    """
    Local loss sensitivity estimated through the gradients of the prediction at points in `x`, as defined in
    https://arxiv.org/pdf/1706.05394.pdf.

    :param classifier: A trained model
    :type classifier: :class:`Classifier`
    :param x: Data sample of shape that can be fed into `classifier`
    :type x: `np.ndarray`
    :param y: Labels for sample `x`, one-hot encoded.
    :type y: `np.ndarray`
    :return: The average loss sensitivity of the model
    :rtype: `float`
    """
    grads = classifier.loss_gradient(x, y)
    norm = la.norm(grads.reshape(grads.shape[0], -1), ord=2, axis=1)

    return np.mean(norm)


def clever(classifier, x, nb_batches, batch_size, radius, norm, target=None, target_sort=False, c_init=1,
           pool_factor=10):
    """
    Compute CLEVER score for an untargeted attack. Paper link: https://arxiv.org/abs/1801.10578

    :param classifier: A trained model.
    :type classifier: :class:`Classifier`
    :param x: One input sample
    :type x: `np.ndarray`
    :param nb_batches: Number of repetitions of the estimate
    :type nb_batches: `int`
    :param batch_size: Number of random examples to sample per batch
    :type batch_size: `int`
    :param radius: Radius of the maximum perturbation
    :type radius: `float`
    :param norm: Current support: 1, 2, np.inf
    :type norm: `int`
    :param target: Class or classes to target. If `None`, targets all classes
    :type target: `int` or iterable of `int`
    :param target_sort: Should the target classes be sorted in prediction order. When `True` and `target` is `None`,
           sort results.
    :type target_sort: `bool`
    :param c_init: initialization of Weibull distribution
    :type c_init: `float`
    :param pool_factor: The factor to create a pool of random samples with size pool_factor x n_s
    :type pool_factor: `int`
    :return: CLEVER score
    :rtype: array of `float`. None if target classes is predicted
    """
    # Find the predicted class first
    y_pred = classifier.predict(np.array([x]), logits=False)
    pred_class = np.argmax(y_pred, axis=1)[0]
    if target is None:
        # Get a list of untargeted classes
        if target_sort:
            target_classes = np.argsort(y_pred)[0][:-1]
        else:
            target_classes = [i for i in range(classifier.nb_classes) if i != pred_class]
    elif isinstance(target, (int, np.integer)):
        target_classes = [target]
    else:
        # Assume it's iterable
        target_classes = target
    score_list = []
    for j in target_classes:
        if j == pred_class:
            score_list.append(None)
            continue
        s = clever_t(classifier, x, j, nb_batches, batch_size, radius, norm, c_init, pool_factor)
        score_list.append(s)
    return np.array(score_list)


def clever_u(classifier, x, nb_batches, batch_size, radius, norm, c_init=1, pool_factor=10):
    """
    Compute CLEVER score for an untargeted attack. Paper link: https://arxiv.org/abs/1801.10578

    :param classifier: A trained model.
    :type classifier: :class:`Classifier`
    :param x: One input sample
    :type x: `np.ndarray`
    :param nb_batches: Number of repetitions of the estimate
    :type nb_batches: `int`
    :param batch_size: Number of random examples to sample per batch
    :type batch_size: `int`
    :param radius: Radius of the maximum perturbation
    :type radius: `float`
    :param norm: Current support: 1, 2, np.inf
    :type norm: `int`
    :param c_init: initialization of Weibull distribution
    :type c_init: `float`
    :param pool_factor: The factor to create a pool of random samples with size pool_factor x n_s
    :type pool_factor: `int`
    :return: CLEVER score
    :rtype: `float`
    """
    # Get a list of untargeted classes
    y_pred = classifier.predict(np.array([x]), logits=True)
    pred_class = np.argmax(y_pred, axis=1)[0]
    untarget_classes = [i for i in range(classifier.nb_classes) if i != pred_class]

    # Compute CLEVER score for each untargeted class
    score_list = []
    for j in untarget_classes:
        s = clever_t(classifier, x, j, nb_batches, batch_size, radius, norm, c_init, pool_factor)
        score_list.append(s)

    return np.min(score_list)


def clever_t(classifier, x, target_class, nb_batches, batch_size, radius, norm, c_init=1, pool_factor=10):
    """
    Compute CLEVER score for a targeted attack. Paper link: https://arxiv.org/abs/1801.10578

    :param classifier: A trained model
    :type classifier: :class:`Classifier`
    :param x: One input sample
    :type x: `np.ndarray`
    :param target_class: Targeted class
    :type target_class: `int`
    :param nb_batches: Number of repetitions of the estimate
    :type nb_batches: `int`
    :param batch_size: Number of random examples to sample per batch
    :type batch_size: `int`
    :param radius: Radius of the maximum perturbation
    :type radius: `float`
    :param norm: Current support: 1, 2, np.inf
    :type norm: `int`
    :param c_init: Initialization of Weibull distribution
    :type c_init: `float`
    :param pool_factor: The factor to create a pool of random samples with size pool_factor x n_s
    :type pool_factor: `int`
    :return: CLEVER score
    :rtype: `float`
    """
    # Check if the targeted class is different from the predicted class
    y_pred = classifier.predict(np.array([x]), logits=True)
    pred_class = np.argmax(y_pred, axis=1)[0]
    if target_class == pred_class:
        raise ValueError("The targeted class is the predicted class.")

    # Check if pool_factor is smaller than 1
    if pool_factor < 1:
        raise ValueError("The `pool_factor` must be larger than 1.")

    # Some auxiliary vars
    grad_norm_set = []
    dim = reduce(lambda x_, y: x_ * y, x.shape, 1)
    shape = [pool_factor * batch_size]
    shape.extend(x.shape)

    # Generate a pool of samples
    rand_pool = np.reshape(random_sphere(nb_points=pool_factor * batch_size, nb_dims=dim, radius=radius, norm=norm),
                           shape)
    rand_pool += np.repeat(np.array([x]), pool_factor * batch_size, 0)
    rand_pool = rand_pool.astype(NUMPY_DTYPE)
    np.clip(rand_pool, classifier.clip_values[0], classifier.clip_values[1], out=rand_pool)

    # Change norm since q = p / (p-1)
    if norm == 1:
        norm = np.inf
    elif norm == np.inf:
        norm = 1
    elif norm != 2:
        raise ValueError("Norm {} not supported".format(norm))

    # Loop over the batches
    for i in range(nb_batches):
        # Random generation of data points
        sample_xs = rand_pool[np.random.choice(pool_factor * batch_size, batch_size)]

        # Compute gradients
        grads = classifier.class_gradient(sample_xs, logits=True)
        if np.isnan(grads).any():
            raise Exception("The classifier results NaN gradients.")

        grad = grads[:, pred_class] - grads[:, target_class]
        grad = np.reshape(grad, (batch_size, -1))
        grad_norm = np.max(np.linalg.norm(grad, ord=norm, axis=1))
        grad_norm_set.append(grad_norm)

    # Maximum likelihood estimation for max gradient norms
    [_, loc, _] = weibull_min.fit(-np.array(grad_norm_set), c_init, optimizer=scipy_optimizer)

    # Compute function value
    values = classifier.predict(np.array([x]), logits=True)
    value = values[:, pred_class] - values[:, target_class]

    # Compute scores
    s = np.min([-value[0] / loc, radius])

    return s
