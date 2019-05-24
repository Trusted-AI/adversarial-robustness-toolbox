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

from art.classifiers import Classifier

logger = logging.getLogger(__name__)


class SklearnSVC(Classifier):
    """
    Wrapper class for importing scikit-learn C-Support Vector Classification models.
    """

    def __init__(self, clip_values=(0, 1), model=None, channel_index=None, defences=None, preprocessing=(0, 1)):
        """
        Create a `Classifier` instance from a scikit-learn C-Support Vector Classification. model.

        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :type clip_values: `tuple`
        :param model: scikit-learn C-Support Vector Classification. model
        :type model: `sklearn.svm.SVC`
        :param channel_index: Index of the axis in data containing the color channels or features. Not used in this
               class.
        :type channel_index: `int`
        :param defences: Defences to be activated with the classifier.
        :type defences: :class:`.Preprocessor` or `list(Preprocessor)` instances
        :param preprocessing: Tuple of the form `(subtractor, divider)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :type preprocessing: `tuple`
        """
        super(SklearnSVC, self).__init__(clip_values=clip_values, channel_index=channel_index,
                                         defences=defences, preprocessing=preprocessing)

        self.model = model

    def class_gradient(self, x, label=None, logits=False):
        """
        Compute per-class derivatives w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param label: Index of a specific per-class derivative. If an integer is provided, the gradient of that class
                      output is computed for all samples. If multiple values as provided, the first dimension should
                      match the batch size of `x`, and each value will be used as target for its corresponding sample in
                      `x`. If `None`, then gradients for all classes will be computed for each sample.
        :type label: `int` or `list`
        :param logits: `True` if the prediction should be done at the logits layer.
        :type logits: `bool`
        :return: Array of gradients of input features w.r.t. each class in the form
                 `(batch_size, nb_classes, input_shape)` when computing for all classes, otherwise shape becomes
                 `(batch_size, 1, input_shape)` when `label` parameter is specified.
        :rtype: `np.ndarray`
        """
        raise NotImplementedError

    def fit(self, x, y, batch_size=128, nb_epochs=20, **kwargs):
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Training data.
        :type x: `np.ndarray`
        :param y: Labels, one-vs-rest encoding.
        :type y: `np.ndarray`
        :param batch_size: Size of batches. Not used in this function.
        :type batch_size: `int`
        :param nb_epochs: Number of epochs to use for training. Not used in this function.
        :type nb_epochs: `int`
        :param kwargs: Dictionary of framework-specific arguments. These should be parameters supported by the
               `fit` function in `sklearn.linear_model.LogisticRegression` and will be passed to this function as such.
        :type kwargs: `dict`
        :return: `None`
        """
        y_index = np.argmax(y, axis=1)
        self.model.fit(X=x, y=y_index, **kwargs)

    def get_activations(self, x, layer, batch_size):
        raise NotImplementedError

    def loss_gradient(self, x, y):
        """
        Compute the gradient of the loss function w.r.t. `x`.
        Paper link: https://pralab.diee.unica.it/sites/default/files/biggio14-svm-chapter.pdf

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param y: Correct labels, one-vs-rest encoding.
        :type y: `np.ndarray`
        :return: Array of gradients of the same shape as `x`.
        :rtype: `np.ndarray`
        """
        if self.model.fit_status_:
            raise AssertionError('Model has not been fitted correctly.')

        num_samples, n_features = x.shape
        gradients = np.zeros_like(x)

        y_index = np.argmax(y, axis=1)

        if len(np.unique(y_index)) == 2:
            sign_multiplier = 1
        else:
            sign_multiplier = -1

        def _get_kernel_gradient(i_sv, x_sample):

            x_i = self.model.support_vectors_[i_sv, :]

            if self.model.kernel == 'linear':
                grad = x_i

            elif self.model.kernel == 'poly':
                raise NotImplementedError

            elif self.model.kernel == 'rbf':
                grad = 2 * self.model._gamma * (-1) * np.exp(
                    -self.model._gamma * np.linalg.norm(x_sample - x_i, ord=2)) * (x_sample - x_i)

            elif self.model.kernel == 'sigmoid':
                raise NotImplementedError

            else:
                raise NotImplementedError(
                    'Loss gradients for kernel \'{}\' are not implemented.'.format(self.model.kernel))

            return grad

        # if self.model.kernel == 'linear':

        i_not_label_i = None
        label_multiplier = None

        support_indices = [0] + list(np.cumsum(self.model.n_support_))
        num_classes = len(self.model.classes_)

        for i_sample in range(num_samples):

            i_label = y_index[i_sample]

            for i_not_label in range(num_classes):

                if i_label != i_not_label:

                    if i_not_label < i_label:
                        i_not_label_i = i_not_label
                        label_multiplier = -1
                    elif i_not_label > i_label:
                        i_not_label_i = i_not_label - 1
                        label_multiplier = 1

                    for i_label_sv in range(support_indices[i_label], support_indices[i_label + 1]):
                        alpha_i_k_y_i = self.model.dual_coef_[i_not_label_i, i_label_sv] * label_multiplier
                        # x_i = self.model.support_vectors_[i_label_sv, :]
                        grad_kernel = _get_kernel_gradient(i_label_sv, x[i_sample])
                        gradients[i_sample, :] += sign_multiplier * alpha_i_k_y_i * grad_kernel

                    for i_not_label_sv in range(support_indices[i_not_label], support_indices[i_not_label + 1]):
                        alpha_i_k_y_i = self.model.dual_coef_[i_not_label_i, i_not_label_sv] * label_multiplier
                        # x_i = self.model.support_vectors_[i_not_label_sv, :]
                        grad_kernel = _get_kernel_gradient(i_not_label_sv, x[i_sample])
                        gradients[i_sample, :] += sign_multiplier * alpha_i_k_y_i * grad_kernel

        # elif self.model.kernel == 'poly':
        #     raise NotImplementedError
        #
        # elif self.model.kernel == 'rbf':

        # support_indices = [0] + list(np.cumsum(self.model.n_support_))
        # num_classes = len(self.model.classes_)
        #
        # for i_sample in range(num_samples):
        #
        #     i_label = y_index[i_sample]
        #
        #     for i_not_label in range(num_classes):
        #
        #         if i_label != i_not_label:
        #
        #             if i_not_label < i_label:
        #                 i_not_label_i = i_not_label
        #                 label_multiplier = -1
        #             elif i_not_label > i_label:
        #                 i_not_label_i = i_not_label - 1
        #                 label_multiplier = 1
        #
        #             gamma = self.model._gamma
        #
        #             for i_label_sv in range(support_indices[i_label], support_indices[i_label + 1]):
        #
        #                 alpha_i_k_y_i = self.model.dual_coef_[i_not_label_i, i_label_sv] * label_multiplier
        #                 x_i = self.model.support_vectors_[i_label_sv, :]
        #                 gradients[i_sample, :] += sign_multiplier * alpha_i_k_y_i * 2.0 * gamma * (-1) * np.exp(-gamma * np.linalg.norm(x[i_sample] - x_i, ord=2)) * (x[i_sample] - x_i)
        #
        #             for i_not_label_sv in range(support_indices[i_not_label], support_indices[i_not_label + 1]):
        #
        #                 alpha_i_k_y_i = self.model.dual_coef_[i_not_label_i, i_not_label_sv] * label_multiplier
        #                 x_i = self.model.support_vectors_[i_not_label_sv, :]
        #                 gradients[i_sample, :] += sign_multiplier * alpha_i_k_y_i * 2.0 * gamma * (-1) * np.exp(-gamma * np.linalg.norm(x[i_sample] - x_i, ord=2)) * (x[i_sample] - x_i)

        # elif self.model.kernel == 'sigmoid':
        #     raise NotImplementedError
        #
        # else:
        #     raise NotImplementedError(
        #         'Loss gradients for kernel \'{}\' are not implemented.'.format(self.model.kernel))

        return gradients

    def predict(self, x, logits=False, batch_size=128):
        """
        Perform prediction for a batch of inputs.

        :param x: Test set.
        :type x: `np.ndarray`
        :param logits: `True` if the prediction should be done at the logits layer.
        :type logits: `bool`
        :param batch_size: Size of batches. Not used in this function.
        :type batch_size: `int`
        :return: Array of predictions of shape `(nb_inputs, self.nb_classes)`.
        :rtype: `np.ndarray`
        """
        if self.model.probability:
            y_pred = self.model.predict_proba(X=x)
        else:
            y_pred_label = self.model.predict(X=x)
            num_classes = len(self.model.classes_)
            targets = np.array(y_pred_label).reshape(-1)
            one_hot_targets = np.eye(num_classes)[targets]
            y_pred = one_hot_targets

        return y_pred

    def save(self, filename, path=None):
        import pickle
        with open(filename + '.pickle', 'wb') as f:
            pickle.dump(self.model, file=f)

    def set_learning_phase(self, train):
        raise NotImplementedError
