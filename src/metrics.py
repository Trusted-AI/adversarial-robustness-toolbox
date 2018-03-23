import config

import numpy as np
import numpy.linalg as la
import tensorflow as tf
from scipy.stats import weibull_min
from scipy.optimize import fmin as scipy_optimizer
from scipy.special import gammainc
from functools import reduce

from src.attacks.fast_gradient import FastGradientMethod


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
    """Compute the Empirical Robustness of a classifier object over the sample x for a given adversarial crafting
    method `method_name`, following https://arxiv.org/abs/1511.04599
    
    :param x: tensor of input points
    :param classifier: classifier object
    :param method_name: adversarial attack name
    :param sess: tf session
    :param method_params: params specific to the adversarial attack
    :return: a float corresponding to the average empirical robustness
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
    (Average) Nearest neighbour distance between the sets x and x_train
    :param x: Tensor of input points (usually, test set, clean examples)
    :param classifier: Classifier object
    :param x_train: Tensor of points (usually, training set, clean examples)
    :param sess: tf session
    :param method_name: Adversarial attack name
    :param method_params: Params specific to the adversarial attack
    :return: A float corresponding to the average distance.
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
    """Local loss sensitivity estimated through the gradients of the loss at points in x, as defined in
    https://arxiv.org/pdf/1706.05394.pdf.

    :param x: Tensor of input points
    :param classifier: Classifier object
    :param sess: tf session
    :return: A float corresponding to the average loss sensitivity.
    """
    x_op = tf.placeholder(dtype=tf.float32, shape=list(x.shape))
    y_pred = classifier.predict(x)
    indices = np.argmax(y_pred, axis=1)
    grads = [tf.gradients(classifier.model(x_op)[:, i], x_op) for i in range(10)]
    res = sess.run(grads, feed_dict={x_op: x})
    res = np.asarray([r[0] for r in res])[indices, list(range(x.shape[0]))]
    res = la.norm(res.reshape(res.shape[0], -1), ord=2, axis=1)

    return np.mean(res)
