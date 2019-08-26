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
This module implements the classifiers for scikit-learn models.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import importlib

import numpy as np

from art.classifiers.classifier import Classifier, ClassifierGradients
from art.utils import to_categorical

logger = logging.getLogger(__name__)


# pylint: disable=C0103
def SklearnClassifier(model, clip_values=None, defences=None, preprocessing=(0, 1)):
    """
    Create a `Classifier` instance from a scikit-learn Classifier model. This is a convenience function that
    instantiates the correct wrapper
    class for the given scikit-learn model

    :param model: scikit-learn Classifier model.
    :type model: `sklearn.base.BaseEstimator`
    :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
            for features.
    :type clip_values: `tuple`
    :param defences: Defences to be activated with the classifier.
    :type defences: :class:`.Preprocessor` or `list(Preprocessor)` instances
    :param preprocessing: Tuple of the form `(subtractor, divider)` of floats or `np.ndarray` of values to be
            used for data preprocessing. The first value will be subtracted from the input. The input will then
            be divided by the second one.
    :type preprocessing: `tuple`
    """
    if model.__class__.__module__.split('.')[0] != 'sklearn':
        raise TypeError("Model is not an sklearn model. Received '%s'" % model.__class__)
    sklearn_name = model.__class__.__name__
    module = importlib.import_module('art.classifiers.scikitlearn')
    if hasattr(module, 'Scikitlearn%s' % sklearn_name):
        return getattr(module, 'Scikitlearn%s' % sklearn_name)(model=model, clip_values=clip_values, defences=defences,
                                                               preprocessing=preprocessing)
    # This basic class at least generically handles `fit`, `predict` and `save`
    return ScikitlearnClassifier(model, clip_values, defences, preprocessing)


class ScikitlearnClassifier(Classifier):
    """
    Wrapper class for scikit-learn classifier models.
    """

    def __init__(self, model, clip_values=None, defences=None, preprocessing=(0, 1)):
        """
        Create a `Classifier` instance from a scikit-learn classifier model.

        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :type clip_values: `tuple`
        :param model: scikit-learn classifier model.
        :type model: `sklearn.base.BaseEstimator`
        :param defences: Defences to be activated with the classifier.
        :type defences: :class:`.Preprocessor` or `list(Preprocessor)` instances
        :param preprocessing: Tuple of the form `(subtractor, divider)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :type preprocessing: `tuple`
        """
        super(ScikitlearnClassifier, self).__init__(clip_values=clip_values, defences=defences,
                                                    preprocessing=preprocessing)

        self._model = model
        self._input_shape = self._get_input_shape(model)

    def fit(self, x, y, **kwargs):
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Training data.
        :type x: `np.ndarray`
        :param y: Labels, one-vs-rest encoding.
        :type y: `np.ndarray`
        :param kwargs: Dictionary of framework-specific arguments. These should be parameters supported by the
               `fit` function in `sklearn` classifier and will be passed to this function as such.
        :type kwargs: `dict`
        :return: `None`
        """
        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=True)
        y_preprocessed = np.argmax(y_preprocessed, axis=1)

        self._model.fit(x_preprocessed, y_preprocessed, **kwargs)
        self._input_shape = self._get_input_shape(self._model)

    def predict(self, x, **kwargs):
        """
        Perform prediction for a batch of inputs.

        :param x: Test set.
        :type x: `np.ndarray`
        :return: Array of predictions of shape `(nb_inputs, self.nb_classes)`.
        :rtype: `np.ndarray`
        """
        # Apply defences
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        if hasattr(self._model, 'predict_proba') and callable(getattr(self._model, 'predict_proba')):
            y_pred = self._model.predict_proba(x_preprocessed)
        elif hasattr(self._model, 'predict') and callable(getattr(self._model, 'predict')):
            y_pred = to_categorical(self._model.predict(x_preprocessed), nb_classes=self._model.classes_.shape[0])
        else:
            raise ValueError('The provided model does not have methods `predict_proba` or `predict`.')

        return y_pred

    def save(self, filename, path=None):
        import pickle
        with open(filename + '.pickle', 'wb') as file_pickle:
            pickle.dump(self._model, file=file_pickle)

    def _get_input_shape(self, model):
        if hasattr(model, 'n_features_'):
            _input_shape = (model.n_features_,)
        elif hasattr(model, 'feature_importances_'):
            _input_shape = (len(model.feature_importances_),)
        elif hasattr(model, 'coef_'):
            if len(model.coef_.shape) == 1:
                _input_shape = (model.coef_.shape[0],)
            else:
                _input_shape = (model.coef_.shape[1],)
        elif hasattr(model, 'support_vectors_'):
            _input_shape = (model.support_vectors_.shape[1],)
        elif hasattr(model, 'steps'):
            _input_shape = self._get_input_shape(model.steps[0][1])
        else:
            _input_shape = None
        return _input_shape


class ScikitlearnDecisionTreeClassifier(ScikitlearnClassifier):
    """
    Wrapper class for scikit-learn Decision Tree Classifier models.
    """

    def __init__(self, model, clip_values=None, defences=None, preprocessing=(0, 1)):
        """
        Create a `Classifier` instance from a scikit-learn Decision Tree Classifier model.

        :param model: scikit-learn Decision Tree Classifier model.
        :type model: `sklearn.tree.DecisionTreeClassifier`
        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :type clip_values: `tuple`
        :param defences: Defences to be activated with the classifier.
        :type defences: :class:`.Preprocessor` or `list(Preprocessor)` instances
        :param preprocessing: Tuple of the form `(subtractor, divider)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :type preprocessing: `tuple`
        """
        # pylint: disable=E0001
        from sklearn.tree import DecisionTreeClassifier

        if not isinstance(model, DecisionTreeClassifier):
            raise TypeError('Model must be of type sklearn.tree.DecisionTreeClassifier')

        super(ScikitlearnDecisionTreeClassifier, self).__init__(model=model, clip_values=clip_values, defences=defences,
                                                                preprocessing=preprocessing)
        self._model = model

    def get_classes_at_node(self, node_id):
        """
        Returns the classification for a given node

        :return: major class in node
        :rtype: `float`
        """
        return np.argmax(self._model.tree_.value[node_id])

    def get_threshold_at_node(self, node_id):
        """
        Returns the threshold of given id for a node

        :return: threshold value of feature split in this node
        :rtype: `float`
        """
        return self._model.tree_.threshold[node_id]

    def get_feature_at_node(self, node_id):
        """
        Returns the feature of given id for a node

        :return: feature index of feature split in this node
        :rtype: `int`
        """
        return self._model.tree_.feature[node_id]

    def get_left_child(self, node_id):
        """
        Returns the id of the left child node of node_id

        :return: the indices of the left child in the tree
        :rtype: `int`
        """
        return self._model.tree_.children_left[node_id]

    def get_right_child(self, node_id):
        """
        Returns the id of the right child node of node_id

        :return: the indices of the right child in the tree
        :rtype: `int`
        """
        return self._model.tree_.children_right[node_id]

    def get_decision_path(self, x):
        """
        Returns the path through nodes in the tree when classifying x. Last one is leaf, first one root node.

        :return: the indices of the nodes in the array structure of the tree.
        :rtype: `np.ndarray`
        """
        if len(np.shape(x)) == 1:
            return self._model.decision_path(x.reshape(1, -1)).indices

        return self._model.decision_path(x).indices


class ScikitlearnExtraTreeClassifier(ScikitlearnClassifier):
    """
    Wrapper class for scikit-learn Extra TreeClassifier Classifier models.
    """

    def __init__(self, model, clip_values=None, defences=None, preprocessing=(0, 1)):
        """
        Create a `Classifier` instance from a scikit-learn Extra TreeClassifier Classifier model.

        :param model: scikit-learn Extra TreeClassifier Classifier model.
        :type model: `sklearn.tree.ExtraTreeClassifier`
        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :type clip_values: `tuple`
        :param defences: Defences to be activated with the classifier.
        :type defences: :class:`.Preprocessor` or `list(Preprocessor)` instances
        :param preprocessing: Tuple of the form `(subtractor, divider)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :type preprocessing: `tuple`
        """
        # pylint: disable=E0001
        from sklearn.tree import ExtraTreeClassifier

        if not isinstance(model, ExtraTreeClassifier):
            raise TypeError('Model must be of type sklearn.tree.ExtraTreeClassifier')

        super(ScikitlearnExtraTreeClassifier, self).__init__(model=model, clip_values=clip_values, defences=defences,
                                                             preprocessing=preprocessing)
        self._model = model


class ScikitlearnAdaBoostClassifier(ScikitlearnClassifier):
    """
    Wrapper class for scikit-learn AdaBoost Classifier models.
    """

    def __init__(self, model, clip_values=None, defences=None, preprocessing=(0, 1)):
        """
        Create a `Classifier` instance from a scikit-learn AdaBoost Classifier model.

        :param model: scikit-learn AdaBoost Classifier model.
        :type model: `sklearn.ensemble.AdaBoostClassifier`
        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :type clip_values: `tuple`
        :param defences: Defences to be activated with the classifier.
        :type defences: :class:`.Preprocessor` or `list(Preprocessor)` instances
        :param preprocessing: Tuple of the form `(subtractor, divider)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :type preprocessing: `tuple`
        """
        # pylint: disable=E0001
        from sklearn.ensemble import AdaBoostClassifier

        if not isinstance(model, AdaBoostClassifier):
            raise TypeError('Model must be of type sklearn.ensemble.AdaBoostClassifier')

        super(ScikitlearnAdaBoostClassifier, self).__init__(model=model, clip_values=clip_values, defences=defences,
                                                            preprocessing=preprocessing)
        self._model = model


class ScikitlearnBaggingClassifier(ScikitlearnClassifier):
    """
    Wrapper class for scikit-learn Bagging Classifier models.
    """

    def __init__(self, model, clip_values=None, defences=None, preprocessing=(0, 1)):
        """
        Create a `Classifier` instance from a scikit-learn Bagging Classifier model.

        :param model: scikit-learn Bagging Classifier model.
        :type model: `sklearn.ensemble.BaggingClassifier`
        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :type clip_values: `tuple`
        :param defences: Defences to be activated with the classifier.
        :type defences: :class:`.Preprocessor` or `list(Preprocessor)` instances
        :param preprocessing: Tuple of the form `(subtractor, divider)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :type preprocessing: `tuple`
        """
        # pylint: disable=E0001
        from sklearn.ensemble import BaggingClassifier

        if not isinstance(model, BaggingClassifier):
            raise TypeError('Model must be of type sklearn.ensemble.BaggingClassifier')

        super(ScikitlearnBaggingClassifier, self).__init__(model=model, clip_values=clip_values, defences=defences,
                                                           preprocessing=preprocessing)
        self._model = model


class ScikitlearnExtraTreesClassifier(ScikitlearnClassifier):
    """
    Wrapper class for scikit-learn Extra Trees Classifier models.
    """

    def __init__(self, model, clip_values=None, defences=None, preprocessing=(0, 1)):
        """
        Create a `Classifier` instance from a scikit-learn Extra Trees Classifier model.

        :param model: scikit-learn Extra Trees Classifier model.
        :type model: `sklearn.ensemble.ExtraTreesClassifier`
        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :type clip_values: `tuple`
        :param defences: Defences to be activated with the classifier.
        :type defences: :class:`.Preprocessor` or `list(Preprocessor)` instances
        :param preprocessing: Tuple of the form `(subtractor, divider)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :type preprocessing: `tuple`
        """
        # pylint: disable=E0001
        from sklearn.ensemble import ExtraTreesClassifier

        if not isinstance(model, ExtraTreesClassifier):
            raise TypeError('Model must be of type sklearn.ensemble.ExtraTreesClassifier')

        super(ScikitlearnExtraTreesClassifier, self).__init__(model=model, clip_values=clip_values, defences=defences,
                                                              preprocessing=preprocessing)
        self._model = model


class ScikitlearnGradientBoostingClassifier(ScikitlearnClassifier):
    """
    Wrapper class for scikit-learn Gradient Boosting Classifier models.
    """

    def __init__(self, model, clip_values=None, defences=None, preprocessing=(0, 1)):
        """
        Create a `Classifier` instance from a scikit-learn Gradient Boosting Classifier model.

        :param model: scikit-learn Gradient Boosting Classifier model.
        :type model: `sklearn.ensemble.GradientBoostingClassifier`
        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :type clip_values: `tuple`
        :param defences: Defences to be activated with the classifier.
        :type defences: :class:`.Preprocessor` or `list(Preprocessor)` instances
        :param preprocessing: Tuple of the form `(subtractor, divider)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :type preprocessing: `tuple`
        """
        # pylint: disable=E0001
        from sklearn.ensemble import GradientBoostingClassifier

        if not isinstance(model, GradientBoostingClassifier):
            raise TypeError('Model must be of type sklearn.ensemble.GradientBoostingClassifier')

        super(ScikitlearnGradientBoostingClassifier, self).__init__(model=model, clip_values=clip_values,
                                                                    defences=defences, preprocessing=preprocessing)
        self._model = model


class ScikitlearnRandomForestClassifier(ScikitlearnClassifier):
    """
    Wrapper class for scikit-learn Random Forest Classifier models.
    """

    def __init__(self, model, clip_values=None, defences=None, preprocessing=(0, 1)):
        """
        Create a `Classifier` instance from a scikit-learn Random Forest Classifier model.

        :param model: scikit-learn Random Forest Classifier model.
        :type model: `sklearn.ensemble.RandomForestClassifier`
        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :type clip_values: `tuple`
        :param defences: Defences to be activated with the classifier.
        :type defences: :class:`.Preprocessor` or `list(Preprocessor)` instances
        :param preprocessing: Tuple of the form `(subtractor, divider)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :type preprocessing: `tuple`
        """
        # pylint: disable=E0001
        from sklearn.ensemble import RandomForestClassifier

        if not isinstance(model, RandomForestClassifier):
            raise TypeError('Model must be of type sklearn.ensemble.RandomForestClassifier')

        super(ScikitlearnRandomForestClassifier, self).__init__(model=model, clip_values=clip_values, defences=defences,
                                                                preprocessing=preprocessing)
        self._model = model


class ScikitlearnLogisticRegression(ScikitlearnClassifier, ClassifierGradients):
    """
    Wrapper class for scikit-learn Logistic Regression models.
    """

    def __init__(self, model, clip_values=None, defences=None, preprocessing=(0, 1)):
        """
        Create a `Classifier` instance from a scikit-learn Logistic Regression model.

        :param model: scikit-learn LogisticRegression model
        :type model: `sklearn.linear_model.LogisticRegression`
        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :type clip_values: `tuple`
        :param defences: Defences to be activated with the classifier.
        :type defences: :class:`.Preprocessor` or `list(Preprocessor)` instances
        :param preprocessing: Tuple of the form `(subtractor, divider)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :type preprocessing: `tuple`
        """

        super(ScikitlearnLogisticRegression, self).__init__(model=model, clip_values=clip_values, defences=defences,
                                                            preprocessing=preprocessing)
        self._model = model

        if hasattr(self._model, 'coef_'):
            self._nb_classes = self._model.classes_.shape[0]
        else:
            self._nb_classes = None

    def class_gradient(self, x, label=None, **kwargs):
        """
        Compute per-class derivatives w.r.t. `x`.

        | Paper link: http://cs229.stanford.edu/proj2016/report/ItkinaWu-AdversarialAttacksonImageRecognition-report.pdf
        | Typo in https://arxiv.org/abs/1605.07277 (equation 6)

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param label: Index of a specific per-class derivative. If an integer is provided, the gradient of that class
                      output is computed for all samples. If multiple values as provided, the first dimension should
                      match the batch size of `x`, and each value will be used as target for its corresponding sample in
                      `x`. If `None`, then gradients for all classes will be computed for each sample.
        :type label: `int` or `list`
        :return: Array of gradients of input features w.r.t. each class in the form
                 `(batch_size, nb_classes, input_shape)` when computing for all classes, otherwise shape becomes
                 `(batch_size, 1, input_shape)` when `label` parameter is specified.
        :rtype: `np.ndarray`
        """
        if not hasattr(self._model, 'coef_'):
            raise ValueError("""Model has not been fitted. Run function `fit(x, y)` of classifier first or provide a
            fitted model.""")

        nb_samples = x.shape[0]

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        y_pred = self._model.predict_proba(X=x_preprocessed)
        weights = self._model.coef_
        w_weighted = np.matmul(y_pred, weights)

        if label is None:
            # Compute the gradients w.r.t. all classes
            class_gradients = list()

            for i_class in range(self.nb_classes):
                class_gradient = np.zeros(x.shape)
                for i_sample in range(nb_samples):
                    class_gradient[i_sample, :] += (weights[i_class, :] - w_weighted[i_sample, :])
                class_gradients.append(class_gradient)

            gradients = np.swapaxes(np.array(class_gradients), 0, 1)

        elif isinstance(label, (int, np.integer)):
            # Compute the gradients only w.r.t. the provided label
            class_gradient = np.zeros(x.shape)
            for i_sample in range(nb_samples):
                class_gradient[i_sample, :] += (weights[label, :] - w_weighted[i_sample, :])

            gradients = np.swapaxes(np.array([class_gradient]), 0, 1)

        elif (isinstance(label, list) and len(label) == nb_samples) or \
                isinstance(label, np.ndarray) and label.shape == (nb_samples,):
            # For each sample, compute the gradients w.r.t. the indicated target class (possibly distinct)
            class_gradients = list()
            unique_labels = list(np.unique(label))

            for unique_label in unique_labels:
                class_gradient = np.zeros(x.shape)
                for i_sample in range(nb_samples):
                    # class_gradient[i_sample, :] += label[i_sample, unique_label] * (weights[unique_label, :]
                    # - w_weighted[i_sample, :])
                    class_gradient[i_sample, :] += (weights[unique_label, :] - w_weighted[i_sample, :])

                class_gradients.append(class_gradient)

            gradients = np.swapaxes(np.array(class_gradients), 0, 1)
            lst = [unique_labels.index(i) for i in label]
            gradients = np.expand_dims(gradients[np.arange(len(gradients)), lst], axis=1)

        else:
            raise TypeError('Unrecognized type for argument `label` with type ' + str(type(label)))

        gradients = self._apply_preprocessing_gradient(x, gradients)

        return gradients

    def fit(self, x, y, **kwargs):
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Training data.
        :type x: `np.ndarray`
        :param y: Labels, one-vs-rest encoding.
        :type y: `np.ndarray`
        :param kwargs: Dictionary of framework-specific arguments. These should be parameters supported by the
               `fit` function in `sklearn.linear_model.LogisticRegression` and will be passed to this function as such.
        :type kwargs: `dict`
        :return: `None`
        """
        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(
            x, y, fit=True)

        y_index = np.argmax(y_preprocessed, axis=1)
        self._model.fit(X=x_preprocessed, y=y_index, **kwargs)
        self._nb_classes = self._model.classes_.shape[0]
        self._input_shape = (self._model.coef_.shape[1],)

    def loss_gradient(self, x, y, **kwargs):
        """
        Compute the gradient of the loss function w.r.t. `x`.

        | Paper link: http://cs229.stanford.edu/proj2016/report/ItkinaWu-AdversarialAttacksonImageRecognition-report.pdf
        | Typo in https://arxiv.org/abs/1605.07277 (equation 6)

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param y: Correct labels, one-vs-rest encoding.
        :type y: `np.ndarray`
        :return: Array of gradients of the same shape as `x`.
        :rtype: `np.ndarray`
        """
        # pylint: disable=E0001
        from sklearn.utils.class_weight import compute_class_weight

        if not hasattr(self._model, 'coef_'):
            raise ValueError("""Model has not been fitted. Run function `fit(x, y)` of classifier first or provide a
            fitted model.""")

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(
            x, y, fit=False)

        num_samples, _ = x_preprocessed.shape
        gradients = np.zeros(x_preprocessed.shape)

        y_index = np.argmax(y_preprocessed, axis=1)
        if self._model.class_weight is None or self._model.class_weight == 'balanced':
            class_weight = np.ones(self.nb_classes)
        else:
            class_weight = compute_class_weight(class_weight=self._model.class_weight, classes=self._model.classes_,
                                                y=y_index)

        y_pred = self._model.predict_proba(X=x_preprocessed)
        weights = self._model.coef_
        w_weighted = np.matmul(y_pred, weights)

        for i_sample in range(num_samples):
            for i_class in range(self.nb_classes):
                gradients[i_sample, :] += class_weight[i_class] * (1.0 - y_preprocessed[i_sample, i_class]) * (
                    weights[i_class, :] - w_weighted[i_sample, :])

        gradients = self._apply_preprocessing_gradient(x, gradients)

        return gradients


class ScikitlearnSVC(ScikitlearnClassifier, ClassifierGradients):
    """
    Wrapper class for scikit-learn C-Support Vector Classification models.
    """

    def __init__(self, model, clip_values=None, defences=None, preprocessing=(0, 1)):
        """
        Create a `Classifier` instance from a scikit-learn C-Support Vector Classification model.

        :param model: scikit-learn C-Support Vector Classification model.
        :type model: `sklearn.svm.SVC` or `sklearn.svm.LinearSVC`
        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :type clip_values: `tuple`
        :param defences: Defences to be activated with the classifier.
        :type defences: :class:`.Preprocessor` or `list(Preprocessor)` instances
        :param preprocessing: Tuple of the form `(subtractor, divider)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :type preprocessing: `tuple`
        """
        # pylint: disable=E0001
        from sklearn.svm import SVC, LinearSVC

        if not isinstance(model, SVC) and not isinstance(model, LinearSVC):
            raise TypeError('Model must be of type sklearn.svm.SVC or sklearn.svm.LinearSVC')

        super(ScikitlearnSVC, self).__init__(model=model, clip_values=clip_values, defences=defences,
                                             preprocessing=preprocessing)
        self._model = model

        if hasattr(self._model, 'classes_'):
            self._nb_classes = len(self._model.classes_)
        else:
            self._nb_classes = None

    def class_gradient(self, x, label=None, **kwargs):
        """
        Compute per-class derivatives w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param label: Index of a specific per-class derivative. If an integer is provided, the gradient of that class
                      output is computed for all samples. If multiple values as provided, the first dimension should
                      match the batch size of `x`, and each value will be used as target for its corresponding sample in
                      `x`. If `None`, then gradients for all classes will be computed for each sample.
        :type label: `int` or `list`
        :raises: `NotImplementedException`
        :return: Array of gradients of input features w.r.t. each class in the form
                 `(batch_size, nb_classes, input_shape)` when computing for all classes, otherwise shape becomes
                 `(batch_size, 1, input_shape)` when `label` parameter is specified.
        :rtype: `np.ndarray`
        """
        raise NotImplementedError

    def fit(self, x, y, **kwargs):
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Training data.
        :type x: `np.ndarray`
        :param y: Labels, one-vs-rest encoding.
        :type y: `np.ndarray`
        :param kwargs: Dictionary of framework-specific arguments. These should be parameters supported by the
               `fit` function in `sklearn.linear_model.LogisticRegression` and will be passed to this function as such.
        :type kwargs: `dict`
        :return: `None`
        """
        y_index = np.argmax(y, axis=1)
        self._model.fit(X=x, y=y_index, **kwargs)
        self._nb_classes = len(self._model.classes_)

        if hasattr(self._model, 'coef_'):
            if len(self._model.coef_.shape) == 1:
                self._input_shape = (self._model.coef_.shape[0],)
            else:
                self._input_shape = (self._model.coef_.shape[1],)
        elif hasattr(self._model, 'support_vectors_'):
            self._input_shape = (self._model.support_vectors_.shape[1],)

    def _get_kernel_gradient(self, i_sv, x_sample):
        # pylint: disable=W0212

        x_i = self._model.support_vectors_[i_sv, :]

        if self._model.kernel == 'linear':
            grad = x_i
        elif self._model.kernel == 'poly':
            grad = self._model.degree * (self._model._gamma * np.sum(x_sample * x_i) + self._model.coef0) ** (
                self._model.degree - 1) * x_i
        elif self._model.kernel == 'rbf':
            grad = 2 * self._model._gamma * (-1) * np.exp(
                -self._model._gamma * np.linalg.norm(x_sample - x_i, ord=2)) * (x_sample - x_i)
        elif self._model.kernel == 'sigmoid':
            raise NotImplementedError
        else:
            raise NotImplementedError(
                'Loss gradients for kernel \'{}\' are not implemented.'.format(self._model.kernel))
        return grad

    def loss_gradient(self, x, y, **kwargs):
        """
        Compute the gradient of the loss function w.r.t. `x`.
        Following equation (1) with lambda=0.

        | Paper link: https://pralab.diee.unica.it/sites/default/files/biggio14-svm-chapter.pdf

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param y: Correct labels, one-vs-rest encoding.
        :type y: `np.ndarray`
        :return: Array of gradients of the same shape as `x`.
        :rtype: `np.ndarray`
        """
        # pylint: disable=E0001
        from sklearn.svm import SVC, LinearSVC

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=False)

        num_samples, _ = x_preprocessed.shape
        gradients = np.zeros_like(x_preprocessed)

        y_index = np.argmax(y_preprocessed, axis=1)

        if isinstance(self._model, SVC):

            if self._model.fit_status_:
                raise AssertionError('Model has not been fitted correctly.')

            if y_preprocessed.shape[1] == 2:
                sign_multiplier = 1
            else:
                sign_multiplier = -1

            i_not_label_i = None
            label_multiplier = None

            support_indices = [0] + list(np.cumsum(self._model.n_support_))

            for i_sample in range(num_samples):

                i_label = y_index[i_sample]

                for i_not_label in range(self.nb_classes):

                    if i_label != i_not_label:

                        if i_not_label < i_label:
                            i_not_label_i = i_not_label
                            label_multiplier = -1
                        elif i_not_label > i_label:
                            i_not_label_i = i_not_label - 1
                            label_multiplier = 1

                        for i_label_sv in range(support_indices[i_label], support_indices[i_label + 1]):
                            alpha_i_k_y_i = self._model.dual_coef_[i_not_label_i, i_label_sv] * label_multiplier
                            grad_kernel = self._get_kernel_gradient(i_label_sv, x_preprocessed[i_sample])
                            gradients[i_sample, :] += sign_multiplier * alpha_i_k_y_i * grad_kernel

                        for i_not_label_sv in range(support_indices[i_not_label], support_indices[i_not_label + 1]):
                            alpha_i_k_y_i = self._model.dual_coef_[i_not_label_i, i_not_label_sv] * label_multiplier
                            grad_kernel = self._get_kernel_gradient(i_not_label_sv, x_preprocessed[i_sample])
                            gradients[i_sample, :] += sign_multiplier * alpha_i_k_y_i * grad_kernel

        elif isinstance(self._model, LinearSVC):

            for i_sample in range(num_samples):

                i_label = y_index[i_sample]
                if self.nb_classes == 2:
                    i_label_i = 0
                    if i_label == 0:
                        label_multiplier = 1
                    elif i_label == 1:
                        label_multiplier = -1
                    else:
                        raise ValueError(
                            'Label index not recognized because it is not 0 or 1.')
                else:
                    i_label_i = i_label
                    label_multiplier = -1

                gradients[i_sample] = label_multiplier * self._model.coef_[i_label_i]
        else:
            raise TypeError('Model not recognized.')

        gradients = self._apply_preprocessing_gradient(x, gradients)

        return gradients

    def predict(self, x, **kwargs):
        """
        Perform prediction for a batch of inputs.

        :param x: Test set.
        :type x: `np.ndarray`
        :return: Array of predictions of shape `(nb_inputs, self.nb_classes)`.
        :rtype: `np.ndarray`
        """
        # pylint: disable=E0001
        from sklearn.svm import SVC

        # Apply defences
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        if isinstance(self._model, SVC) and self._model.probability:
            y_pred = self._model.predict_proba(X=x_preprocessed)
        else:
            y_pred_label = self._model.predict(X=x_preprocessed)
            targets = np.array(y_pred_label).reshape(-1)
            one_hot_targets = np.eye(self.nb_classes)[targets]
            y_pred = one_hot_targets

        return y_pred


ScikitlearnLinearSVC = ScikitlearnSVC
