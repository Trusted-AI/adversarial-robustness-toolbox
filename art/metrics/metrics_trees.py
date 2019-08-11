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
This module implements robustness metrics for tree-based models.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np
from copy import deepcopy

logger = logging.getLogger(__name__)


class RobustnessMetricTreeModelsCliqueMethod:
    """
    Robustness metric for tree-based models.
    Following the implementation in https://github.com/chenhongge/treeVerification (MIT License, 9 August 2019)
    Paper link: https://arxiv.org/abs/1906.03849,
    """

    def __init__(self, classifier):
        """
        Create robustness metric for decision-tree-based classifier.

        :param classifier: A trained decision-tree-based classifier.
        :type classifier: `art.classifiers.ClassifierDecisionTree`
        """
        self._classifier = classifier
        self._trees = self._classifier.get_trees()
        self.x = None
        self.y = None
        self.max_clique = None
        self.max_level = None

    def verify(self, x, y, eps_init, norm=np.inf, num_search_steps=10, max_clique=2, max_level=2):
        """
        Verify the robustness of the classifier on the dataset `(x, y)`.

        :param x: Feature data.
        :type x: `np.ndarray`
        :param y: Labels, one-vs-rest encoding.
        :type : `np.ndarray`
        :param eps_init: Attack budget for the first search step.
        :type eps_init: `double`
        :param norm: The norm to apply epsilon.
        :type norm: `int` or `np.inf`
        :param num_search_steps: The number of search steps.
        :type num_search_steps: `int`
        :param max_clique: The maximum number of nodes in a clique.
        :type max_clique: `int`
        :param max_level: The maximum number of clique search levels.
        :type max_level: `int`
        :return: A tuple of the average robustness bound and the verification error at `eps`
        :rtype: `tuple`
        """
        self.x = x
        self.y = y
        self.max_clique = max_clique
        self.max_level = max_level

        average_bound = 0
        num_initial_successes = 0
        num_samples = x.shape[0]

        for i_sample in range(num_samples):

            eps = eps_init
            robust_log = list()
            i_robust = None
            i_not_robust = None

            for i_step in range(num_search_steps):
                logger.info('Search step {}: eps = {}'.format(i_step, eps))

                is_robust = True

                if self._classifier.num_classes <= 2:
                    best_score = self._get_best_score(i_sample, eps, norm, target_label=None)
                    is_robust = (self.y[i_sample] < 0.5 and best_score < 0) or (
                            self.y[i_sample] > 0.5 and best_score > 0)
                else:
                    for i_class in range(self._classifier.num_classes):
                        if i_class != self.y[i_sample]:
                            best_score = self._get_best_score(i_sample, eps, norm, target_label=i_class)
                            is_robust = is_robust and (best_score > 0)
                            if not is_robust:
                                break

                robust_log.append(is_robust)

                if is_robust:
                    if i_step == 0:
                        num_initial_successes += 1
                    logger.info('Model is robust at eps = {}'.format(eps))
                    i_robust = i_step
                    eps_robust = eps
                else:
                    logger.info('Model is not robust at eps = {}'.format(eps))
                    i_not_robust = i_step
                    eps_not_robust = eps

                if i_robust is None:
                    eps /= 2.0
                else:
                    if i_not_robust is None:
                        if eps >= 1.0:
                            logger.info('Abort binary search because eps increased above 1.0')
                            break
                        eps = min(eps * 2.0, 1.0)
                    else:
                        eps = (eps_robust + eps_not_robust) / 2.0

            if i_robust is not None:
                clique_bound = eps_robust
                average_bound += clique_bound
            else:
                logger.info('point', i_sample, ': WARNING! no robust eps found, verification bound is set as 0 !')

        verified_error = 1.0 - num_initial_successes / num_samples
        average_bound = average_bound / num_samples

        logger.info('The average interval bound is: {}'.format(average_bound))
        logger.info('The verified error at eps = {} is: {}'.format(eps_init, verified_error))

        return average_bound, verified_error

    def _get_k_partite_clique(self, accessible_leafs, label, target_label):
        """
        Find the K partite cliques among the accessible leaf nodes.

        :param accessible_leafs: List of lists of accessible leaf nodes.
        :type accessible_leafs: `list(list(LeafNode))`
        :param label: The try label of the current sample.
        :type label: `int`
        :param target_label: The target label.
        :type target_label: `int` or `None`
        :return: `None`
        """
        new_nodes_list = list()
        best_scores_sum = 0.0

        for start_tree in range(0, len(accessible_leafs), self.max_clique):

            cliques_old = list()
            cliques_new = list()

            # Start searching for cliques
            for accessible_leaf in accessible_leafs[start_tree]:
                if self._classifier.num_classes > 2 and target_label is not None \
                        and target_label == accessible_leaf.class_label:
                    new_leaf_value = - accessible_leaf.value
                else:
                    new_leaf_value = accessible_leaf.value
                cliques_old.append({'box': accessible_leaf.box, 'value': new_leaf_value})

            # Loop over all all trees
            for i_tree in range(start_tree + 1, min(len(accessible_leafs), start_tree + self.max_clique)):
                cliques_new.clear()
                # Loop over all existing cliques
                for clique in cliques_old:
                    # Loop over leaf nodes in tree
                    for accessible_leaf in accessible_leafs[i_tree]:
                        leaf_box = deepcopy(accessible_leaf.box)
                        leaf_box.intersect_with_box(clique['box'])
                        if leaf_box.intervals:
                            if self._classifier.num_classes > 2 and target_label is not None \
                                    and target_label == accessible_leaf.class_label:
                                new_leaf_value = - accessible_leaf.value
                            else:
                                new_leaf_value = accessible_leaf.value
                            cliques_new.append({'box': leaf_box, 'value': new_leaf_value + clique['value']})

                cliques_old = cliques_new.copy()

            new_nodes = list()
            best_score = 0.0
            for i, clique in enumerate(cliques_old):
                # Create a new node without tree_id and node_id to represent clique
                new_nodes.append(
                    LeafNode(tree_id=None, class_label=label, node_id=None, box=clique['box'], value=clique['value']))

                if i == 0:
                    best_score = clique['value']
                else:
                    if label < 0.5 and self._classifier.num_classes <= 2:
                        best_score = max(best_score, clique['value'])
                    else:
                        best_score = min(best_score, clique['value'])

            new_nodes_list.append(new_nodes)
            best_scores_sum += best_score

        return best_scores_sum, new_nodes_list

    def _get_best_score(self, i_sample, eps, norm, target_label):
        """
        Get the list of best scores.

        :param i_sample: Index of training sample in `x`.
        :type i_sample: `int`
        :param eps: Attack budget epsilon.
        :type eps: `double`
        :param norm: The norm to apply epsilon.
        :type norm: `int` or `np.inf`
        :param target_label: The target label.
        :type target_label: `int` or `None`
        :return: The best scores.
        :rtype: `double`
        """
        nodes = self._get_accessible_leafs(i_sample, eps, norm, target_label)
        best_score = None

        for i_level in range(self.max_level):
            if self._classifier.num_classes > 2 and i_level > 0:
                target_label = None
            best_score, nodes = self._get_k_partite_clique(nodes, label=self.y[i_sample], target_label=target_label)

            # Stop if the root node has been reached
            if len(nodes) <= 1:
                break

        return best_score

    def _get_distance(self, box, i_sample, norm):
        """
        Determine the distance between sample and interval box.

        :param box: Interval box.
        :type box: `Box`
        :param i_sample: Index of training sample in `x`.
        :type i_sample: `int`
        :param norm: The norm to apply epsilon.
        :type norm: `int` or `np.inf`
        :return: The distance.
        :rtype: `double`
        """
        resulting_distance = 0.0

        for feature, interval in box.intervals.items():

            feature_value = self.x[i_sample, feature]

            if interval.lower_bound < feature_value < interval.upper_bound:
                distance = 0.0
            else:
                difference = max(feature_value - interval.upper_bound, interval.lower_bound - feature_value)
                if norm == 0:
                    distance = 1.0
                elif norm == np.inf:
                    distance = difference
                else:
                    distance = pow(difference, norm)

            if norm == np.inf:
                resulting_distance = max(resulting_distance, distance)
            else:
                resulting_distance += distance

        if norm == 0:
            resulting_distance = resulting_distance
        elif norm == np.inf:
            resulting_distance = resulting_distance
        else:
            resulting_distance = pow(resulting_distance, 1.0 / norm)

        return resulting_distance

    def _get_accessible_leafs(self, i_sample, eps, norm, target_label):
        """
        Determine the leaf nodes accessible within the attack budget.

        :param i_sample: Index of training sample in `x`.
        :type i_sample: `int`
        :param eps: Attack budget epsilon.
        :type eps: `double`
        :param norm: The norm to apply epsilon.
        :type norm: `int` or `np.inf`
        :param target_label: The target label.
        :type target_label: `int`
        :return: A list of lists of leaf nodes.
        :rtype: list(list(LeafNode))
        """
        accessible_leafs = list()

        for tree in self._trees:
            if self._classifier.num_classes <= 2 or target_label is None or tree.class_id in [self.y[i_sample],
                                                                                              target_label]:

                leafs = list()

                for leaf_node in tree.leaf_nodes:
                    distance = self._get_distance(leaf_node.box, i_sample, norm)
                    if leaf_node.box and distance <= eps:
                        leafs.append(leaf_node)

                if not leafs:
                    raise ValueError('No accessible leafs found.')

                accessible_leafs.append(leafs)

        return accessible_leafs


class Interval:

    def __init__(self, lower_bound, upper_bound):
        """
        An of interval of a feature.

        :param lower_bound: The lower boundary of the feature.
        :type lower_bound: `double` or `-np.inf`
        :param upper_bound: The upper boundary of the feature.
        :type upper_bound: `double` or `-np.inf`
        """
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound


class Box:

    def __init__(self, intervals=None):
        """
        A box of intervals.

        :param intervals: A dictionary of intervals with features as keys.
        :type intervals: `dict(feature: Interval)`
        """
        if intervals is None:
            self.intervals = dict()
        else:
            self.intervals = intervals

    def intersect_with_box(self, box):
        """
        Intersect two interval boxes.

        :param box: Interval box to intersect with.
        :type box: `Box`
        """
        for key, value in box.intervals.items():
            if key not in self.intervals:
                self.intervals[key] = value
            else:
                lower_bound = max(self.intervals[key].lower_bound, value.lower_bound)
                upper_bound = min(self.intervals[key].upper_bound, value.upper_bound)

                if lower_bound >= upper_bound:
                    self.intervals.clear()
                    break

                self.intervals[key] = Interval(lower_bound, upper_bound)

    def __repr__(self):
        return self.__class__.__name__ + '({})'.format(self.intervals)


class LeafNode:

    def __init__(self, tree_id, class_label, node_id, box, value):
        self.tree_id = tree_id
        self.class_label = class_label
        self.node_id = node_id
        self.box = box
        self.value = value

    def __repr__(self):
        return self.__class__.__name__ + '({}, {}, {}, {}, {})'.format(self.tree_id, self.class_label, self.node_id,
                                                                       self.box, self.value)


class Tree:

    def __init__(self, class_id, leaf_nodes):
        self.class_id = class_id
        self.leaf_nodes = leaf_nodes
