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
This module implements the classifier `XGBoostClassifier` for XGBoost models.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np

from art.classifiers.classifier import Classifier, ClassifierDecisionTree

logger = logging.getLogger(__name__)


class XGBoostClassifier(Classifier, ClassifierDecisionTree):
    """
    Wrapper class for importing XGBoost models.
    """

    def __init__(self, model=None, clip_values=None, defences=None, preprocessing=None, nb_features=None,
                 nb_classes=None):
        """
        Create a `Classifier` instance from a XGBoost model.

        :param model: XGBoost model
        :type model: `xgboost.Booster` or `xgboost.XGBClassifier`
        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :type clip_values: `tuple`
        :param defences: Defences to be activated with the classifier.
        :type defences: :class:`.Preprocessor` or `list(Preprocessor)` instances
        :param preprocessing: Tuple of the form `(subtractor, divider)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :type preprocessing: `tuple`
        :param nb_features: The number of features in the training data. Only used if it cannot be extracted from
                             model.
        :type nb_features: `int` or `None`
        :param nb_classes: The number of classes in the training data. Only used if it cannot be extracted from model.
        :type nb_classes: `int` or `None`
        """
        from xgboost import Booster, XGBClassifier

        if not isinstance(model, Booster) and not isinstance(model, XGBClassifier):
            raise TypeError('Model must be of type xgboost.Booster or xgboost.XGBClassifier')

        super(XGBoostClassifier, self).__init__(clip_values=clip_values, defences=defences, preprocessing=preprocessing)

        self._model = model
        self._input_shape = (nb_features,)
        self._nb_classes = nb_classes

    def fit(self, x, y, **kwargs):
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Training data.
        :type x: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :type y: `np.ndarray`
        :param kwargs: Dictionary of framework-specific arguments. These should be parameters supported by the
                       `fit` function in `xgboost.Booster` or `xgboost.XGBClassifier` and will be passed to this
                       function as such.
        :type kwargs: `dict`
        :raises: `NotImplementedException`
        :return: `None`
        """
        raise NotImplementedError

    def predict(self, x, **kwargs):
        """
        Perform prediction for a batch of inputs.

        :param x: Test set.
        :type x: `np.ndarray`
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        :rtype: `np.ndarray`
        """
        from xgboost import Booster, XGBClassifier
        from art.utils import to_categorical

        # Apply defences
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        if isinstance(self._model, Booster):
            from xgboost import DMatrix
            train_data = DMatrix(x_preprocessed, label=None)
            predictions = self._model.predict(train_data)
            y_prediction = np.asarray([line for line in predictions])
            if len(y_prediction.shape) == 1:
                y_prediction = to_categorical(labels=y_prediction, nb_classes=self.nb_classes())
            return y_prediction

        if isinstance(self._model, XGBClassifier):
            return self._model.predict_proba(x_preprocessed)

        return None

    def nb_classes(self):
        """
        Return the number of output classes.

        :return: Number of classes in the data.
        :rtype: `int`
        """
        from xgboost import Booster, XGBClassifier
        if isinstance(self._model, Booster):
            try:
                return int(len(self._model.get_dump(dump_format='json')) / self._model.n_estimators)
            except AttributeError:
                if self._nb_classes is not None:
                    return self._nb_classes
                raise NotImplementedError('Number of classes cannot be determined automatically. ' +
                                          'Please manually set argument nb_classes in XGBoostClassifier.')

        if isinstance(self._model, XGBClassifier):
            return self._model.n_classes_

        return None

    def save(self, filename, path=None):
        import pickle
        with open(filename + '.pickle', 'wb') as file_pickle:
            pickle.dump(self.model, file=file_pickle)

    def get_trees(self):
        """
        Get the decision trees.

        :return: A list of decision trees.
        :rtype: `[Tree]`
        """

        import json
        from art.metrics.verification_decisions_trees import Box, Tree

        booster_dump = self._model.get_booster().get_dump(dump_format='json')
        trees = list()

        for i_tree, tree_dump in enumerate(booster_dump):
            box = Box()

            if self._model.n_classes_ == 2:
                class_label = -1
            else:
                class_label = i_tree % self._model.n_classes_

            tree_json = json.loads(tree_dump)
            trees.append(
                Tree(class_id=class_label, leaf_nodes=self._get_leaf_nodes(tree_json, i_tree, class_label, box)))

        return trees

    def _get_leaf_nodes(self, node, i_tree, class_label, box):
        from copy import deepcopy
        from art.metrics.verification_decisions_trees import LeafNode, Box, Interval

        leaf_nodes = list()

        if 'children' in node:
            if node['children'][0]['nodeid'] == node['yes'] and node['children'][1]['nodeid'] == node['no']:
                node_left = node['children'][0]
                node_right = node['children'][1]
            elif node['children'][1]['nodeid'] == node['yes'] and node['children'][0]['nodeid'] == node['no']:
                node_left = node['children'][1]
                node_right = node['children'][0]
            else:
                raise ValueError

            box_left = deepcopy(box)
            box_right = deepcopy(box)

            feature = int(node['split'][1:])
            box_split_left = Box(intervals={feature: Interval(-np.inf, node['split_condition'])})
            box_split_right = Box(intervals={feature: Interval(node['split_condition'], np.inf)})

            if box.intervals:
                box_left.intersect_with_box(box_split_left)
                box_right.intersect_with_box(box_split_right)
            else:
                box_left = box_split_left
                box_right = box_split_right

            leaf_nodes += self._get_leaf_nodes(node_left, i_tree, class_label, box_left)
            leaf_nodes += self._get_leaf_nodes(node_right, i_tree, class_label, box_right)

        if 'leaf' in node:
            leaf_nodes.append(LeafNode(tree_id=i_tree, class_label=class_label, node_id=node['nodeid'], box=box,
                                       value=node['leaf']))

        return leaf_nodes
