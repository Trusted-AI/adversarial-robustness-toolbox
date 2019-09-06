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
This module implements the classifier `LightGBMClassifier` for LightGBM models.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np

from art.classifiers.classifier import Classifier, ClassifierDecisionTree

logger = logging.getLogger(__name__)


class LightGBMClassifier(Classifier, ClassifierDecisionTree):
    """
    Wrapper class for importing LightGBM models.
    """

    def __init__(self, model=None, clip_values=None, defences=None, preprocessing=None):
        """
        Create a `Classifier` instance from a LightGBM model.

        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :type clip_values: `tuple`
        :param model: LightGBM model
        :type model: `lightgbm.Booster`
        :param defences: Defences to be activated with the classifier.
        :type defences: :class:`.Preprocessor` or `list(Preprocessor)` instances
        :param preprocessing: Tuple of the form `(subtractor, divider)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :type preprocessing: `tuple`
        """
        from lightgbm import Booster

        if not isinstance(model, Booster):
            raise TypeError('Model must be of type lightgbm.Booster')

        super(LightGBMClassifier, self).__init__(clip_values=clip_values, defences=defences,
                                                 preprocessing=preprocessing)

        self._model = model
        self._input_shape = (self._model.num_feature(),)

    def fit(self, x, y, **kwargs):
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Training data.
        :type x: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :type y: `np.ndarray`
        :param kwargs: Dictionary of framework-specific arguments. These should be parameters supported by the
               `fit` function in `lightgbm.Booster` and will be passed to this function as such.
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
        # Apply defences
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        return self._model.predict(x_preprocessed)

    def nb_classes(self):
        """
        Return the number of output classes.

        :return: Number of classes in the data.
        :rtype: `int`
        """
        # pylint: disable=W0212
        return self._model._Booster__num_class

    def save(self, filename, path=None):
        import pickle
        with open(filename + '.pickle', 'wb') as file_pickle:
            pickle.dump(self._model, file=file_pickle)

    def get_trees(self):
        """
        Get the decision trees.

        :return: A list of decision trees.
        :rtype: `[Tree]`
        """
        from art.metrics.verification_decisions_trees import Box, Tree

        booster_dump = self._model.dump_model()["tree_info"]
        trees = list()

        for i_tree, tree_dump in enumerate(booster_dump):
            box = Box()

            # pylint: disable=W0212
            if self._model._Booster__num_class == 2:
                class_label = -1
            else:
                class_label = i_tree % self._model._Booster__num_class

            trees.append(Tree(class_id=class_label,
                              leaf_nodes=self._get_leaf_nodes(tree_dump['tree_structure'], i_tree, class_label, box)))

        return trees

    def _get_leaf_nodes(self, node, i_tree, class_label, box):
        from copy import deepcopy
        from art.metrics.verification_decisions_trees import LeafNode, Box, Interval

        leaf_nodes = list()

        if 'split_index' in node:
            node_left = node['left_child']
            node_right = node['right_child']

            box_left = deepcopy(box)
            box_right = deepcopy(box)

            feature = node['split_feature']
            box_split_left = Box(intervals={feature: Interval(-np.inf, node['threshold'])})
            box_split_right = Box(intervals={feature: Interval(node['threshold'], np.inf)})

            if box.intervals:
                box_left.intersect_with_box(box_split_left)
                box_right.intersect_with_box(box_split_right)
            else:
                box_left = box_split_left
                box_right = box_split_right

            leaf_nodes += self._get_leaf_nodes(node_left, i_tree, class_label, box_left)
            leaf_nodes += self._get_leaf_nodes(node_right, i_tree, class_label, box_right)

        if 'leaf_index' in node:
            leaf_nodes.append(LeafNode(tree_id=i_tree, class_label=class_label, node_id=node['leaf_index'], box=box,
                                       value=node['leaf_value']))

        return leaf_nodes
