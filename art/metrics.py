"""
Module implementing varying metrics for assessing model robustness. These fall mainly under two categories:
attack-dependent and attack-independent.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import config

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


def get_crafter(method, classifier, session, params=None):
    try:
        crafter = supported_methods[method]["class"](classifier, sess=session)
    except:
        raise NotImplementedError("{} crafting method not supported.".format(method))

    if params:
        crafter.set_params(**params)
    else:
        crafter.set_params(**supported_methods[method]["params"])

    return crafter


def empirical_robustness(x, classifier, sess, method_name, method_params=None):
    """Compute the Empirical Robustness of a classifier object over the sample `x` for a given adversarial crafting
    method `attack`. This is equivalent to computing the minimal perturbation that the attacker must introduce for a
    successful attack. Paper link: https://arxiv.org/abs/1511.04599
    
    :param x: Data sample of shape that can be fed into `classifier`
    :type x: `np.ndarray`
    :param classifier: A trained model
    :type classifier: :class:`Classifier`
    :param sess: The session for the computation
    :type sess: `tf.Session`
    :param method_name: adversarial attack name
    :type method_name: `str`
    :param method_params: Parameters specific to the adversarial attack
    :type method_params: `dict`
    :return: The average empirical robustness computed on `x`
    :rtype: `float`
    """
    crafter = get_crafter(method_name, classifier, sess, method_params)
    adv_x = crafter.generate(x, minimal=True, **method_params)

    # Predict the labels for adversarial examples
    y = classifier.predict(x, verbose=0)
    y_pred = classifier.predict(adv_x, verbose=0)

    idxs = (np.argmax(y_pred, axis=1) != np.argmax(y, axis=1))
    if np.sum(idxs) == 0.0:
        return 0

    perts_norm = la.norm((adv_x - x).reshape(x.shape[0], -1), ord=crafter.ord, axis=1)
    perts_norm = perts_norm[idxs]

    return np.mean(perts_norm / la.norm(x[idxs].reshape(np.sum(idxs), -1), ord=crafter.ord, axis=1))


def kernel_rbf(x, y, sigma=0.1):
    """Computes the RBF kernel

    :param x: a tensor object or a numpy array
    :param y: a tensor object or a numpy array
    :param sigma: standard deviation
    :return: a tensor object
    """
    norms_x = tf.reduce_sum(x ** 2, 1)[:, None]  # axis = [1] for later tf versions
    norms_y = tf.reduce_sum(y ** 2, 1)[None, :]
    dists = norms_x - 2 * tf.matmul(x, y, transpose_b=True) + norms_y
    return tf.exp(-(1.0/(2.0*sigma)*dists))


def euclidean_dist(x, y):
    """Computes the Euclidean distance between x and y

    :param x: A tensor object or a numpy array
    :param y: A tensor object or a numpy array
    :return: A tensor object
    """
    norms_x = tf.reduce_sum(x ** 2, 1)[:, None]  # axis = [1] for later tf versions
    norms_y = tf.reduce_sum(y ** 2, 1)[None, :]
    dists = norms_x - 2 * tf.matmul(x, y, transpose_b=True) + norms_y
    return dists


def mmd(x_data, y_data, sess, sigma=0.1):
    """ Computes the maximum mean discrepancy between x and y

    :param x_data: Numpy array
    :param y_data: Numpy array
    :param sess: tf session
    :param sigma: Standard deviation
    :return: A float value corresponding to mmd(x_data, y_data)
    """
    assert x_data.shape[0] == y_data.shape[0]
    x_data = x_data.reshape(x_data.shape[0], np.prod(x_data.shape[1:]))
    y_data = y_data.reshape(y_data.shape[0], np.prod(y_data.shape[1:]))
    x = tf.placeholder(tf.float32, shape=x_data.shape)
    y = tf.placeholder(tf.float32, shape=y_data.shape)
    mmd_ = tf.reduce_sum(kernel_rbf(x, x, sigma)) - 2 * tf.reduce_sum(kernel_rbf(x, y, sigma)) \
           + tf.reduce_sum(kernel_rbf(y, y, sigma))
    
    return sess.run(mmd_, feed_dict={x: x_data, y: y_data})


def nearest_neighbour_dist(x, classifier, x_train, sess, method_name, method_params=None):
    """
    Compute the (average) nearest neighbour distance between the sets `x` and `x_train`: for each point in `x`,
    measure the Euclidean distance to its closest point in `x_train`, then average over all points.

    :param x: Data sample of shape that can be fed into `classifier`
    :type x: `np.ndarray`
    :param classifier: A trained model
    :type classifier: :class:`Classifier`
    :param x_train: Reference data sample to be considered as neighbors
    :type x_train: `np.ndarray`
    :param sess: The session for the computation
    :type sess: `tf.Session`
    :param method_name: adversarial attack name
    :type method_name: `str`
    :param method_params: Parameters specific to the adversarial attack
    :type method_params: `dict`
    :return: The average nearest neighbors distance
    :rtype: `float`
    """
    # Craft the adversarial examples
    crafter = get_crafter(method_name, classifier, sess, method_params)
    adv_x = crafter.generate(x, minimal=True, **method_params)

    # Predict the labels for adversarial examples
    y = classifier.predict(x, verbose=0)
    y_pred = classifier.predict(adv_x, verbose=0)

    adv_x_ = adv_x.reshape(adv_x.shape[0], np.prod(adv_x.shape[1:]))
    x_ = x_train.reshape(x_train.shape[0], np.prod(x_train.shape[1:]))
    dists = euclidean_dist(adv_x_, x_)

    dists = np.min(sess.run(dists), 1) / la.norm(x.reshape(x.shape[0], -1), ord=2, axis=1)
    idxs = (np.argmax(y_pred, axis=1) != np.argmax(y, axis=1))
    avg_nn_dist = np.mean(dists[idxs])

    return avg_nn_dist


def loss_sensitivity(x, classifier, sess):
    """
    Local loss sensitivity estimated through the gradients of the loss at points in `x`, as defined in
    https://arxiv.org/pdf/1706.05394.pdf.

    :param x: Data sample of shape that can be fed into `classifier`
    :type x: `np.ndarray`
    :param classifier: A trained model
    :type classifier: :class:`Classifier`
    :param sess: The session for the computation
    :type sess: `tf.Session`
    :return: The average loss sensitivity of the model
    :rtype: `float`
    """
    from art.attacks.attack import class_derivative

    x_op = tf.placeholder(dtype=tf.float32, shape=list(x.shape))
    y_pred = classifier.predict(x)
    indices = np.argmax(y_pred, axis=1)
    grads = class_derivative(classifier._get_predictions(x_op, log=True), x_op,
                             classifier.model.get_output_shape_at(0)[1])
    res = sess.run(grads, feed_dict={x_op: x})
    res = np.asarray([r[0] for r in res])[indices, list(range(x.shape[0]))]
    res = la.norm(res.reshape(res.shape[0], -1), ord=2, axis=1)

    return np.mean(res)


def clever_u(x, classifier, n_b, n_s, r, sess, c_init=1):
    """
    Compute CLEVER score for an untargeted attack. Paper link: https://arxiv.org/abs/1801.10578

    :param x: One input sample
    :type x: `np.ndarray`
    :param classifier: A trained model.
    :type classifier: :class:`Classifier`
    :param n_b: Batch size
    :type n_b: `int`
    :param n_s: Number of examples per batch
    :type n_s: `int`
    :param r: Maximum perturbation
    :type r: `float`
    :param sess: The session to run graphs in
    :type sess: `tf.Session`
    :param c_init: initialization of Weibull distribution
    :type c_init: `float`
    :return: A tuple of 3 CLEVER scores, corresponding to norms 1, 2 and np.inf
    :rtype: `tuple`
    """
    # Get a list of untargeted classes
    y_pred = classifier.predict(np.array([x]))
    pred_class = np.argmax(y_pred, axis=1)[0]
    num_class = np.shape(y_pred)[1]
    untarget_classes = [i for i in range(num_class) if i != pred_class]

    # Compute CLEVER score for each untargeted class
    score1_list, score2_list, score8_list = [], [], []
    for j in untarget_classes:
        s1, s2, s8 = clever_t(x, classifier, j, n_b, n_s, r, sess, c_init)
        score1_list.append(s1)
        score2_list.append(s2)
        score8_list.append(s8)

    return np.min(score1_list), np.min(score2_list), np.min(score8_list)


def clever_t(x, classifier, target_class, n_b, n_s, r, sess, c_init=1):
    """
    Compute CLEVER score for a targeted attack. Paper link: https://arxiv.org/abs/1801.10578

    :param x: One input sample
    :type x: `np.ndarray`
    :param classifier: A trained model
    :type classifier: :class:`Classifier`
    :param target_class: Targeted class
    :type target_class: `int`
    :param n_b: Batch size
    :type n_b: `int`
    :param n_s: Number of examples per batch
    :type n_s: `int`
    :param r: Maximum perturbation
    :type r: `float`
    :param sess: The session to run graphs in
    :type sess: `tf.Session`
    :param c_init: Initialization of Weibull distribution
    :type c_init: `float`
    :return: A tuple of 3 CLEVER scores, corresponding to norms 1, 2 and np.inf
    :rtype: `tuple`
    """
    # Check if the targeted class is different from the predicted class
    y_pred = classifier.predict(np.array([x]))
    pred_class = np.argmax(y_pred, axis=1)[0]
    if target_class == pred_class:
        raise ValueError("The targeted class is the predicted class!")

    # Define placeholders for computing g gradients
    shape = [None]
    shape.extend(x.shape)
    imgs = tf.placeholder(shape=shape, dtype=tf.float32)
    pred_class_ph = tf.placeholder(dtype=tf.int32, shape=[])
    target_class_ph = tf.placeholder(dtype=tf.int32, shape=[])

    # Define tensors for g gradients
    grad_norm_1, grad_norm_2, grad_norm_8, g_x = _build_g_gradient(imgs, classifier, pred_class_ph, target_class_ph)

    # Some auxiliary vars
    set1, set2, set8 = [], [], []
    dim = reduce(lambda x_, y: x_ * y, x.shape, 1)
    shape = [n_s]
    shape.extend(x.shape)

    # Compute predicted class
    y_pred = classifier.predict(np.array([x]))
    pred_class = np.argmax(y_pred, axis=1)[0]

    # Loop over n_b batches
    for i in range(n_b):
        # Random generation of data points
        sample_xs0 = np.reshape(_random_sphere(m=n_s, n=dim, r=r), shape)
        sample_xs = sample_xs0 + np.repeat(np.array([x]), n_s, 0)
        np.clip(sample_xs, 0, 1, out=sample_xs)

        # Preprocess data if it is supported in the classifier
        if hasattr(classifier, 'feature_squeeze'):
            sample_xs = classifier.feature_squeeze(sample_xs)
        sample_xs = classifier._preprocess(sample_xs)

        # Compute gradients
        max_gn1, max_gn2, max_gn8 = sess.run(
            [grad_norm_1, grad_norm_2, grad_norm_8],
            feed_dict={imgs: sample_xs, pred_class_ph: pred_class,
                       target_class_ph: target_class})
        set1.append(max_gn1)
        set2.append(max_gn2)
        set8.append(max_gn8)

    # Maximum likelihood estimation for max gradient norms
    [_, loc1, _] = weibull_min.fit(-np.array(set1), c_init, optimizer=scipy_optimizer)
    [_, loc2, _] = weibull_min.fit(-np.array(set2), c_init, optimizer=scipy_optimizer)
    [_, loc8, _] = weibull_min.fit(-np.array(set8), c_init, optimizer=scipy_optimizer)

    # Compute g_x0
    x0 = np.array([x])
    if hasattr(classifier, 'feature_squeeze'):
        x0 = classifier.feature_squeeze(x0)
    x0 = classifier._preprocess(x0)
    g_x0 = sess.run(g_x, feed_dict={imgs: x0, pred_class_ph: pred_class,
                                    target_class_ph: target_class})

    # Compute scores
    # Note q = p / (p-1)
    s8 = np.min([-g_x0[0] / loc1, r])
    s2 = np.min([-g_x0[0] / loc2, r])
    s1 = np.min([-g_x0[0] / loc8, r])

    return s1, s2, s8


def _build_g_gradient(x, classifier, pred_class, target_class):
    """
    Build tensors of gradient `g`.

    :param x: One input sample
    :type x: `np.ndarray`
    :param classifier: A trained model
    :type classifier: :class:`Classifier`
    :param pred_class: Predicted class
    :type pred_class: `int`
    :param target_class: Target class
    :type target_class: `int`
    :return: Max gradient norms
    :rtype: `tuple`
    """
    # Get predict values
    y_pred = classifier.model(x)
    pred_val = y_pred[:, pred_class]
    target_val = y_pred[:, target_class]
    g_x = pred_val - target_val

    # Get the gradient op
    grad_op = tf.gradients(g_x, x)[0]

    # Compute the gradient norm
    grad_op_rs = tf.reshape(grad_op, (tf.shape(grad_op)[0], -1))
    grad_norm_1 = tf.reduce_max(tf.norm(grad_op_rs, ord=1, axis=1))
    grad_norm_2 = tf.reduce_max(tf.norm(grad_op_rs, ord=2, axis=1))
    grad_norm_8 = tf.reduce_max(tf.norm(grad_op_rs, ord=np.inf, axis=1))

    return grad_norm_1, grad_norm_2, grad_norm_8, g_x


def _random_sphere(m, n, r):
    """
    Generate randomly `m x n`-dimension points with radius `r` and centered around 0.

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
    base = gammainc(n/2, s2/2)**(1/n) * r / np.sqrt(s2)
    a = a * (np.tile(base, (n, 1))).T

    return a
