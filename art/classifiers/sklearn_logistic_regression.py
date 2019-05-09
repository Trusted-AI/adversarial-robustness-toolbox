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
from sklearn.utils.class_weight import compute_class_weight

from art.classifiers import Classifier

logger = logging.getLogger(__name__)


class SklearnLogisticRegression(Classifier):
    """
    Wrapper class for importing scikit-learn LogisticRegression models.
    """
    def __init__(self, clip_values=(0, 1), model=None, channel_index=None, defences=None, preprocessing=(0, 1)):
        """
        Create a `Classifier` instance from a scikit-learn LogisticRegression model.

        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :type clip_values: `tuple`
        :param model: scikit-learn LogisticRegression model
        :type model: `sklearn.linear_model.LogisticRegression`
        :param use_logits: True if the output of the model are the logits. Not used in this class.
        :type use_logits: `bool`
        :param channel_index: Index of the axis in data containing the color channels or features. Not used in this
               class.
        :type channel_index: `int`
        :param defences: Defences to be activated with the classifier.
        :type defences: :class:`.Preprocessor` or `list(Preprocessor)` instances
        :param preprocessing: Tuple of the form `(subtractor, divider)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :type preprocessing: `tuple`
        :param input_layer: Which layer to consider as the Input when the model has multiple input layers. Not used in
               this class.
        :type input_layer: `int`
        :param output_layer: Which layer to consider as the Output when the model has multiple output layers. Not used
               in this class.
        :type output_layer: `int`
        :param custom_activation: True if the model uses the last activation other than softmax and requires to use the
               output probability rather than the logits by attacks. Not used in this class.
        :type custom_activation: `bool`
        """

        super(SklearnLogisticRegression, self).__init__(clip_values=clip_values, channel_index=channel_index,
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
        w = self.model.coef_
        num_classes = len(self.model.classes_)
        num_samples, n_features = x.shape
        gradients = np.zeros_like(x)

        class_weight = compute_class_weight(class_weight=self.model.class_weight, classes=self.model.classes_,
                                            y=np.argmax(y, axis=1))

        y_pred = self.model.predict_proba(X=x)
        w_weighted = np.matmul(y_pred, w)

        for i_sample in range(num_samples):
            for i_class in range(num_classes):
                gradients[i_sample, :] += class_weight[i_class] * y[i_sample, i_class] * (
                        w[i_class, :] - w_weighted[i_sample, :])

        return gradients

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

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param y: Correct labels, one-vs-rest encoding.
        :type y: `np.ndarray`
        :return: Array of gradients of the same shape as `x`.
        :rtype: `np.ndarray`
        """
        w = self.model.coef_
        num_classes = len(self.model.classes_)
        num_samples, n_features = x.shape
        gradients = np.zeros_like(x)

        class_weight = compute_class_weight(class_weight=self.model.class_weight, classes=self.model.classes_,
                                            y=np.argmax(y, axis=1))

        y_pred = self.model.predict_proba(X=x)
        w_weighted = np.matmul(y_pred, w)

        for i_sample in range(num_samples):
            for i_class in range(num_classes):
                gradients[i_sample, :] += class_weight[i_class] * (1 - y[i_sample, i_class]) * (
                        w[i_class, :] - w_weighted[i_sample, :])

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
        return self.model.predict_proba(X=x)

    def save(self, filename, path=None):
        import pickle
        with open(filename + '.pickle', 'wb') as f:
            pickle.dump(self.model, file=f)

    def set_learning_phase(self, train):
        raise NotImplementedError
