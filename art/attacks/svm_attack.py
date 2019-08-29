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
This module implements attacks on Support Vector Machines.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np

from art.attacks.attack import Attack
from art.classifiers.scikitlearn import ScikitlearnSVC

from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC, SVC

logger = logging.getLogger(__name__)


class SVMAttack(Attack):
    """
    Close implementation of Biggio's attack on SVMs
    Paper link: https://arxiv.org/pdf/1206.6389.pdf
    """
    attack_params = ['classifier', 'step', 'eps', 'x_train', 'y_train', 'x_val', 'y_val']

    def __init__(self, classifier, step, eps, x_train, y_train, x_val=None, y_val=None, max_iters=100, **kwargs):
        """
        Initalize an SVM attack

        :param classifier: A trained ScikitlearnSVC classifier
        :type classifier: `art.classifiers.scikitlearn.ScikitlearnSVC`
        :param step: The step size of the classifier
        :type step: float
        :param eps: The minimum difference in loss before convergence of the classifer
        :type eps: float
        :param x_train: The training data used for classification
        :type x_train: `np.ndarray`
        :param y_train: The training labels used for classification
        :type y_train: `np.ndarray`
        :param x_val: The validation data used to test the attack
        :type x_val: `np.ndarray`
        :param y_val: The validation labels used to test the attack
        :param max_iters: The maximum number of iterations for the attack
        :param kwargs: Extra optional keyword arguments
        """
        super(SVMAttack, self).__init__(classifier)

        if not isinstance(classifier, ScikitlearnSVC):
            raise TypeError('Classifier must be a SVC')
        if isinstance(self.classifier.model, LinearSVC):
            self.classifier.model = SVC(C=self.classifier.model.C, kernel='linear')
            self.classifier.model.fit(x_train, y_train)
        elif not isinstance(self.classifier.model, SVC):
            raise NotImplementedError("Model type '{}' not yet supported".format(type(self.classifier.model)))

        self.step = step
        self.eps = eps
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.max_iters = max_iters
        self.set_params(**kwargs)

    def generate(self, x, **kwargs):
        """
        Iteratively finds optimal attack points starting at values at x

        :param x: An array with the points that initialize attack points.
        :type x: `np.ndarray`
        :return: An array holding the adversarial examples.
        :rtype: `np.ndarray`
        """
        num_poison = len(x)
        if num_poison == 0:
            raise ValueError("Must input at least one poison point")
        num_features = len(x[0])
        train_data = np.copy(self.x_train)
        train_labels = np.copy(self.y_train)
        classifier = self.classifier
        all_poison = []
        for attack_point in x:
            poison, classifier = generate_attack_point(classifier, train_data, train_labels, self.x_val, self.y_val,
                                                       attack_point, self.step, self.eps, max_iters=self.max_iters)
            all_poison.append(poison)
            poison_label = classifier.predict(poison)
            train_data = np.vstack([train_data, poison])
            train_labels = np.vstack([train_labels, poison_label])

        return np.array(all_poison).reshape((num_poison, num_features))

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and apply attack-specific checks before saving them as attributes.

        :param kwargs: a dictionary of attack-specific parameters
        :type kwargs: `dict`
        :return: `True` when parsing was successful
        """
        super(SVMAttack, self).set_params(**kwargs)
        if self.step <= 0:
            raise ValueError("Step size must be strictly positive")
        if self.eps <= 0:
            raise ValueError("Value of eps must be strictly positive")
        if self.max_iters <= 1:
            raise ValueError("Value of max_iters must be strictly positive")


def generate_attack_point(classifier, x_train, y_train, x_val, y_val, init_attack, step, eps, max_iters):
    """
    Generate a single poison attack the model, using x_val and y_val as validation points.
    The attack begins at the point init_attack. The attack class will be the opposite of the model's
    classification for init_attack.

    :param classifier: a trained SVC classifier
    :type classifier: `art.classifiers.ScikitlearnSVC`
    :param x_train: training data
    :type x_train: `np.ndarray`
    :param y_train: training labels
    :type y_train: `np.ndarray`
    :param x_val: validation data
    :type x_val: `np.ndarray`
    :param y_val: validation labels
    :type y_val: `np.ndarray`
    :param init_attack: the initial attack point
    :type init_attack: `np.ndarray`
    :param step: the step size
    :type step: float
    :param eps: the min difference in loss to aim for
    :type eps: float
    :param max_iters: the maximum number of iterations the attack should run
    :type max_iters: int
    :return: a tuple containing the final attack point and the poisoned model
    :rtype: (`np.ndarray`, `art.classifiers.ScikitlearnSVC`)
    """
    poisoned_model = classifier.load_model()
    i = 0
    attack_point = np.expand_dims(init_attack, axis=0)
    attack_class = poisoned_model.predict(attack_point, prob=False)
    g = classifier.discriminant(x_val)
    k_values = np.where(-g > 0)
    new_p = np.sum(g[k_values])
    old_p = new_p

    while new_p - old_p < eps and i < max_iters:
        old_p = new_p
        poisoned_input = np.vstack([x_train, attack_point])
        poisoned_labels = np.vstack([y_train, attack_class])
        poisoned_model.fit(poisoned_input, poisoned_labels)

        unit_grad = normalize(attack_gradient(poisoned_model, attack_point, x_val[k_values], y_val[k_values]))
        new_attack = attack_point + step * unit_grad
        lower, upper = classifier.clip_values
        new_attack = np.clip(new_attack, lower, upper)

        new_g = poisoned_model.discriminant(x_val)
        k_values = np.where(-new_g > 0)
        new_p = np.sum(new_g[k_values])
        i += 1
        attack_point = new_attack
    poisoned_input = np.vstack([x_train, attack_point])
    poisoned_labels = np.vstack([y_train, attack_class])
    poisoned_model.fit(poisoned_input, poisoned_labels, save=True)
    return attack_point, poisoned_model


def predict_sign(classifier, input):
    """
    Predicts the inputs by binary classifier and outputs -1 and 1 instead of 0 and 1

    :param classifier: a binary classifer without outputs 0, 1
    :type classifier: `art.classifiers.scikitlearn.ScikitlearnClassifier`
    :param input: an input array
    :type input: `np.ndarray`
    :return: an array of -1/1 predictions
    :rtype: `np.ndarray`
    """
    preds = classifier.predict(input, prob=False)
    return one_hot_to_binary(preds)


def one_hot_to_binary(y_values):
    """
    Converts binary classification one hot labels to -1/1

    :param y_values: binary labels
    :type y_values: `np.ndarray`
    :return: the one hot labels
    :rtype: `np.ndarray`
    """
    a = np.array([0, 1])
    b = np.array([1, 0])
    signs = np.zeros(y_values.shape[0],)
    signs[np.all(y_values == a, axis=1)] = 1
    signs[np.all(y_values == b, axis=1)] = -1
    return signs


def attack_gradient(model, attack_point, valid_data, valid_labels):
    """
    Calculates the attack gradient, or ∆P for this attack.
    See equation 8 in Biggio Ch. 14

    :param model: a trained SVC classifier, trained with attack_point
    :type model: `art.classifiers.scikitlearn.ScikitlearnSVC`
    :param attack_point: the current attack point
    :type attack_point: `np.ndarray`
    :param valid_data: validation data
    :type valid_data: `np.ndarray`
    :param valid_labels: validation labels
    :type valid_labels: `np.ndarray`
    :return: The attack gradient
    :rtype: `np.ndarray`
    """
    grad = np.zeros((1, valid_data.shape[1]))
    support_vectors = model.model.support_vectors_
    support_labels = np.expand_dims(predict_sign(model, support_vectors), axis=1)
    c_idx = np.isin(support_vectors, attack_point).all(axis=1)
    if not c_idx.any():
        return grad
    else:
        c_idx = np.where(c_idx == True)[0][0]
        alpha_c = model.model.dual_coef_[0, c_idx]

    assert support_labels.shape == (len(support_vectors), 1)
    qss_inv = np.linalg.inv(model.q_submatrix(support_vectors, support_vectors))
    zeta = np.matmul(qss_inv, support_labels)
    zeta = np.matmul(support_labels.T, zeta)
    v = np.matmul(qss_inv, support_labels)

    for x_k, y_k in zip(valid_data, valid_labels):
        y_k = one_hot_to_binary(np.array([y_k]))

        q_ks = model.q_submatrix(np.array([x_k]), support_vectors)
        m_k = (1.0 / zeta) * np.matmul(q_ks, zeta * qss_inv - np.matmul(v, v.T)) + np.matmul(y_k, v.T)
        d_q_sc = np.fromfunction(lambda i: model._get_kernel_gradient_sv(i, attack_point), (len(support_vectors),),
                                 dtype=int)
        d_q_kc = model.kernel_grad(x_k, attack_point)
        grad += (np.matmul(m_k, d_q_sc) + d_q_kc) * alpha_c

    return grad

