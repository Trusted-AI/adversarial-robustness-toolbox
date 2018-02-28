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


def clever_u(x, classifier, n_b, n_s, r, sess, c_init=1):
    """
    Compute CLEVER score for un-targeted attack.
    https://arxiv.org/abs/1801.10578
    :param x: one data example
    :param classifier: K-class classifier
    :param n_b: batch size
    :param n_s: number of examples per batch
    :param r: maximum perturbation
    :param sess:
    :param c_init: initialization of Weibull distribution
    :return: CLEVER score
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
    Compute CLEVER score for targeted attack.
    https://arxiv.org/abs/1801.10578
    :param x: one data example
    :param classifier: K-class classifier
    :param target_class:
    :param n_b: batch size
    :param n_s: number of examples per batch
    :param r: maximum perturbation
    :param sess:
    :param c_init: initialization of Weibull distribution
    :return: CLEVER score
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
    grad_norm_1, grad_norm_2, grad_norm_8, g_x = _build_g_gradient(
        imgs, classifier, pred_class_ph, target_class_ph)

    # Some auxiliary vars
    set1, set2, set8 = [], [], []
    dim = reduce(lambda x, y: x * y, x.shape, 1)
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
    [_, loc1, _] = weibull_min.fit(-np.array(set1), c_init,
                                   optimizer=scipy_optimizer)
    [_, loc2, _] = weibull_min.fit(-np.array(set2), c_init,
                                   optimizer=scipy_optimizer)
    [_, loc8, _] = weibull_min.fit(-np.array(set8), c_init,
                                   optimizer=scipy_optimizer)

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
    Build tensors of g gradient.
    :param x:
    :param classifier:
    :param pred_class:
    :param target_class:
    :return: max gradient norms
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
    Generate randomly m n-dimension points with radius r and 0-center.
    :param m: number of random data points
    :param n: dimension
    :param r: radius
    :return:
    """
    A = np.random.randn(m, n)
    s2 = np.sum(A**2, axis=1)
    base = gammainc(n/2, s2/2)**(1/n) * r / np.sqrt(s2)
    A = A * (np.tile(base, (n,1))).T

    return A
