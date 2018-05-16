"""
Module implementing varying metrics for assessing model robustness. These fall mainly under two categories:
attack-dependent and attack-independent.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import numpy.linalg as la
import tensorflow as tf
from scipy.stats import weibull_min
from scipy.optimize import fmin as scipy_optimizer
from scipy.special import gammainc
from functools import reduce

from art.attacks.fast_gradient import FastGradientMethod

# TODO add all other implemented attacks
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
    # else:
    #     # Use default params
    #     crafter.set_params(**supported_methods[method]["params"])

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
    adv_x = crafter.generate(x, minimal=True, **attack_params)

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


def loss_sensitivity(classifier, x):
    """
    Local loss sensitivity estimated through the gradients of the prediction at points in `x`, as defined in
    https://arxiv.org/pdf/1706.05394.pdf.

    :param classifier: A trained model
    :type classifier: :class:`Classifier`
    :param x: Data sample of shape that can be fed into `classifier`
    :type x: `np.ndarray`
    :return: The average loss sensitivity of the model
    :rtype: `float`
    """
    y_pred = np.argmax(classifier.predict(x), axis=1)
    grads = classifier.class_gradient(x, logits=True)[list(range(x.shape[0])), y_pred]
    norm = la.norm(grads.reshape(grads.shape[0], -1), ord=2, axis=1)

    return np.mean(norm)


def clever_u(classifier, x, n_b, n_s, r, norm, c_init=1, pool_factor=10):
    """
    Compute CLEVER score for an untargeted attack. Paper link: https://arxiv.org/abs/1801.10578

    :param classifier: A trained model.
    :type classifier: :class:`Classifier`
    :param x: One input sample
    :type x: `np.ndarray`
    :param n_b: Batch size
    :type n_b: `int`
    :param n_s: Number of examples per batch
    :type n_s: `int`
    :param r: Maximum perturbation
    :type r: `float`
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
        s = clever_t(classifier, x, j, n_b, n_s, r, norm, c_init, pool_factor)
        score_list.append(s)

    return np.min(score_list)


def clever_t(classifier, x, target_class, n_b, n_s, r, norm, c_init=1, pool_factor=10):
    """
    Compute CLEVER score for a targeted attack. Paper link: https://arxiv.org/abs/1801.10578

    :param classifier: A trained model
    :type classifier: :class:`Classifier`
    :param x: One input sample
    :type x: `np.ndarray`
    :param target_class: Targeted class
    :type target_class: `int`
    :param n_b: Batch size
    :type n_b: `int`
    :param n_s: Number of examples per batch
    :type n_s: `int`
    :param r: Maximum perturbation
    :type r: `float`
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
        raise ValueError("The targeted class is the predicted class")

    # Check if pool_factor is smaller than 1
    if pool_factor < 1:
        raise ValueError("The pool_factor must be larger than 1")

    # Some auxiliary vars
    grad_norm_set = []
    dim = reduce(lambda x_, y: x_ * y, x.shape, 1)
    shape = [pool_factor * n_s]
    shape.extend(x.shape)

    # Generate a pool of samples
    rand_pool = np.reshape(_random_sphere(m=pool_factor * n_s, n=dim, r=r, norm=norm), shape)
    rand_pool += np.repeat(np.array([x]), pool_factor * n_s, 0)
    np.clip(rand_pool, classifier.clip_values[0], classifier.clip_values[1], out=rand_pool)

    # Change norm since q = p / (p-1)
    if norm == 1:
        norm = np.inf
    elif norm == np.inf:
        norm = 1
    elif norm != 2:
        raise ValueError("Norm {} not supported".format(norm))

    # Loop over n_b batches
    for i in range(n_b):
        # Random generation of data points
        sample_xs = rand_pool[np.random.choice(pool_factor * n_s, n_s)]

        # Compute gradients
        grads = classifier.class_gradient(sample_xs, logits=True)
        if np.isnan(grads).any():
            raise Exception("The classifier results NaN gradients")

        grad = grads[:, pred_class] - grads[:, target_class]
        grad = np.reshape(grad, (n_s, -1))
        grad_norm = np.max(np.linalg.norm(grad, ord=norm, axis=1))
        grad_norm_set.append(grad_norm)

    # Maximum likelihood estimation for max gradient norms
    [_, loc, _] = weibull_min.fit(-np.array(grad_norm_set), c_init, optimizer=scipy_optimizer)

    # Compute function value
    values = classifier.predict(np.array([x]), logits=True)
    value = values[:, pred_class] - values[:, target_class]

    # Compute scores
    s = np.min([-value[0] / loc, r])

    return s


def _random_sphere(m, n, r, norm):
    """
    Generate randomly `m x n`-dimension points with radius `r` and centered around 0.

    :param m: Number of random data points
    :type m: `int`
    :param n: Dimension
    :type n: `int`
    :param r: Radius
    :type r: `float`
    :param norm: Current support: 1, 2, np.inf
    :type norm: `int`
    :return: The generated random sphere
    :rtype: `np.ndarray`
    """
    if norm == 1:
        res = _l1_random(m, n, r)
    elif norm == 2:
        res = _l2_random(m, n, r)
    elif norm == np.inf:
        res = _linf_random(m, n, r)
    else:
        raise NotImplementedError("Norm {} not supported".format(norm))

    return res


def _l2_random(m, n, r):
    """
    Generate randomly `m x n`-dimension points with radius `r` in norm 2 and centered around 0.

    :param m: Number of random data points
    :type m: `int`
    :param n: Dimension
    :type n: `int`
    :param r: Radius
    :type r: `float`
    :return: The generated random sphere
    :rtype: `np.ndarray`
    """
    a = np.random.randn(m, n)
    s2 = np.sum(a**2, axis=1)
    base = gammainc(n/2.0, s2/2.0)**(1/n) * r / np.sqrt(s2)
    a = a * (np.tile(base, (n, 1))).T

    return a


def _l1_random(m, n, r):
    """
    Generate randomly `m x n`-dimension points with radius `r` in norm 1 and centered around 0.

    :param m: Number of random data points
    :type m: `int`
    :param n: Dimension
    :type n: `int`
    :param r: Radius
    :type r: `float`
    :return: The generated random sphere
    :rtype: `np.ndarray`
    """
    A = np.zeros(shape=(m, n+1))
    A[:, -1] = np.sqrt(np.random.uniform(0, r**2, m))

    for i in range(m):
        A[i, 1:-1] = np.sort(np.random.uniform(0, A[i, -1], n-1))

    X = (A[:, 1:] - A[:, :-1]) * np.random.choice([-1, 1], (m, n))

    return X


def _linf_random(m, n, r):
    """
    Generate randomly `m x n`-dimension points with radius `r` in inf norm and centered around 0.

    :param m: Number of random data points
    :type m: `int`
    :param n: Dimension
    :type n: `int`
    :param r: Radius
    :type r: `float`
    :return: The generated random sphere
    :rtype: `np.ndarray`
    """
    return np.random.uniform(float(-r), float(r), (m, n))



