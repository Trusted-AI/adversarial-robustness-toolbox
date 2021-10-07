# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
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
This module implements robustness verifications for decision-tree-based models.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
from tqdm.auto import trange

if TYPE_CHECKING:
    from art.estimators.classification.classifier import ClassifierDecisionTree

logger = logging.getLogger(__name__)


class Interval:
    """
    Representation of an intervals bound.
    """

    def __init__(self, lower_bound: float, upper_bound: float) -> None:
        """
        An interval of a feature.

        :param lower_bound: The lower boundary of the feature.
        :param upper_bound: The upper boundary of the feature.
        """
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound


class Box:
    """
    Representation of a box of intervals bounds.
    """

    def __init__(self, intervals: Optional[Dict[int, Interval]] = None) -> None:
        """
        A box of intervals.

        :param intervals: A dictionary of intervals with features as keys.
        """
        if intervals is None:
            self.intervals = dict()
        else:
            self.intervals = intervals

    def intersect_with_box(self, box: "Box") -> None:
        """
        Get the intersection of two interval boxes. This function modifies this box instance.

        :param box: Interval box to intersect with this box.
        """
        for key, value in box.intervals.items():
            if key not in self.intervals:
                self.intervals[key] = value
            else:
                lower_bound = max(self.intervals[key].lower_bound, value.lower_bound)
                upper_bound = min(self.intervals[key].upper_bound, value.upper_bound)

                if lower_bound >= upper_bound:  # pragma: no cover
                    self.intervals.clear()
                    break

                self.intervals[key] = Interval(lower_bound, upper_bound)

    def get_intersection(self, box: "Box") -> "Box":
        """
        Get the intersection of two interval boxes. This function creates a new box instance.

        :param box: Interval box to intersect with this box.
        """
        box_new = Box(intervals=self.intervals.copy())

        for key, value in box.intervals.items():
            if key not in box_new.intervals:
                box_new.intervals[key] = value
            else:
                lower_bound = max(box_new.intervals[key].lower_bound, value.lower_bound)
                upper_bound = min(box_new.intervals[key].upper_bound, value.upper_bound)

                if lower_bound >= upper_bound:
                    box_new.intervals.clear()
                    return box_new

                box_new.intervals[key] = Interval(lower_bound, upper_bound)

        return box_new

    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.intervals)


class LeafNode:
    """
    Representation of a leaf node of a decision tree.
    """

    def __init__(
        self,
        tree_id: Optional[int],
        class_label: int,
        node_id: Optional[int],
        box: Box,
        value: float,
    ) -> None:
        """
        Create a leaf node representation.

        :param tree_id: ID of the decision tree.
        :param class_label: ID of class to which this leaf node is contributing.
        :param box: A box representing the n_feature-dimensional bounding intervals that reach this leaf node.
        :param value: Prediction value at this leaf node.
        """
        self.tree_id = tree_id
        self.class_label = class_label
        self.node_id = node_id
        self.box = box
        self.value = value

    def __repr__(self):
        return self.__class__.__name__ + "({}, {}, {}, {}, {})".format(
            self.tree_id, self.class_label, self.node_id, self.box, self.value
        )


class Tree:
    """
    Representation of a decision tree.
    """

    def __init__(self, class_id: Optional[int], leaf_nodes: List[LeafNode]) -> None:
        """
        Create a decision tree representation.

        :param class_id: ID of the class to which this decision tree contributes.
        :param leaf_nodes: A list of leaf nodes of this decision tree.
        """
        self.class_id = class_id
        self.leaf_nodes = leaf_nodes


class RobustnessVerificationTreeModelsCliqueMethod:
    """
    Robustness verification for decision-tree-based models.
    Following the implementation in https://github.com/chenhongge/treeVerification (MIT License, 9 August 2019)

    | Paper link: https://arxiv.org/abs/1906.03849
    """

    def __init__(self, classifier: "ClassifierDecisionTree", verbose: bool = True) -> None:
        """
        Create robustness verification for a decision-tree-based classifier.

        :param classifier: A trained decision-tree-based classifier.
        :param verbose: Show progress bars.
        """
        self._classifier = classifier
        self.verbose = verbose
        self._trees = self._classifier.get_trees()

    def verify(
        self,
        x: np.ndarray,
        y: np.ndarray,
        eps_init: float,
        norm: int = np.inf,
        nb_search_steps: int = 10,
        max_clique: int = 2,
        max_level: int = 2,
    ) -> Tuple[float, float]:
        """
        Verify the robustness of the classifier on the dataset `(x, y)`.

        :param x: Feature data of shape `(nb_samples, nb_features)`.
        :param y: Labels, one-vs-rest encoding of shape `(nb_samples, nb_classes)`.
        :param eps_init: Attack budget for the first search step.
        :param norm: The norm to apply epsilon.
        :param nb_search_steps: The number of search steps.
        :param max_clique: The maximum number of nodes in a clique.
        :param max_level: The maximum number of clique search levels.
        :return: A tuple of the average robustness bound and the verification error at `eps`.
        """
        self.x: np.ndarray = x
        self.y: np.ndarray = np.argmax(y, axis=1)
        self.max_clique: int = max_clique
        self.max_level: int = max_level

        average_bound: float = 0.0
        num_initial_successes: int = 0
        num_samples: int = x.shape[0]

        # pylint: disable=R1702
        pbar = trange(num_samples, desc="Decision tree verification", disable=not self.verbose)
        for i_sample in pbar:

            eps: float = eps_init
            robust_log: List[bool] = list()
            i_robust = None
            i_not_robust = None
            eps_robust: float = 0.0
            eps_not_robust: float = 0.0
            best_score: Optional[float]

            for i_step in range(nb_search_steps):
                logger.info("Search step {0:d}: eps = {1:.4g}".format(i_step, eps))

                is_robust = True

                if self._classifier.nb_classes <= 2:
                    best_score = self._get_best_score(i_sample, eps, norm, target_label=None)
                    is_robust = (self.y[i_sample] < 0.5 and best_score < 0) or (
                        self.y[i_sample] > 0.5 and best_score > 0.0
                    )
                else:
                    for i_class in range(self._classifier.nb_classes):
                        if i_class != self.y[i_sample]:
                            best_score = self._get_best_score(i_sample, eps, norm, target_label=i_class)
                            is_robust = is_robust and (best_score > 0.0)
                            if not is_robust:
                                break

                robust_log.append(is_robust)

                if is_robust:
                    if i_step == 0:
                        num_initial_successes += 1
                    logger.info("Model is robust at eps = {:.4g}".format(eps))
                    i_robust = i_step
                    eps_robust = eps
                else:
                    logger.info("Model is not robust at eps = {:.4g}".format(eps))
                    i_not_robust = i_step
                    eps_not_robust = eps

                if i_robust is None:
                    eps /= 2.0
                else:
                    if i_not_robust is None:
                        if eps >= 1.0:  # pragma: no cover
                            logger.info("Abort binary search because eps increased above 1.0")
                            break
                        eps = min(eps * 2.0, 1.0)
                    else:
                        eps = (eps_robust + eps_not_robust) / 2.0

            if i_robust is not None:
                clique_bound = eps_robust
                average_bound += clique_bound
            else:
                logger.info(
                    "point %s: WARNING! no robust eps found, verification bound is set as 0 !",
                    i_sample,
                )

        verified_error = 1.0 - num_initial_successes / num_samples
        average_bound = average_bound / num_samples

        logger.info("The average interval bound is: {:.4g}".format(average_bound))
        logger.info("The verified error at eps = {0:.4g} is: {1:.4g}".format(eps_init, verified_error))

        return average_bound, verified_error

    def _get_k_partite_clique(
        self,
        accessible_leaves: List[List[LeafNode]],
        label: int,
        target_label: Optional[int],
    ) -> Tuple[float, List]:
        """
        Find the K partite cliques among the accessible leaf nodes.

        :param accessible_leaves: List of lists of accessible leaf nodes.
        :param label: The try label of the current sample.
        :param target_label: The target label.
        :return: The best score and a list of new cliques.
        """
        new_nodes_list = list()
        best_scores_sum = 0.0

        # pylint: disable=R1702
        for start_tree in range(0, len(accessible_leaves), self.max_clique):
            cliques_old: List[Dict[str, Union[Box, float]]] = list()
            cliques_new: List[Dict[str, Union[Box, float]]] = list()

            # Start searching for cliques
            for accessible_leaf in accessible_leaves[start_tree]:
                if (
                    self._classifier.nb_classes > 2
                    and target_label is not None
                    and target_label == accessible_leaf.class_label
                ):
                    new_leaf_value = -accessible_leaf.value
                else:
                    new_leaf_value = accessible_leaf.value
                cliques_old.append({"box": accessible_leaf.box, "value": new_leaf_value})

            # Loop over all all trees
            for i_tree in range(
                start_tree + 1,
                min(len(accessible_leaves), start_tree + self.max_clique),
            ):
                cliques_new.clear()
                # Loop over all existing cliques
                for clique in cliques_old:
                    # Loop over leaf nodes in tree
                    for accessible_leaf in accessible_leaves[i_tree]:
                        leaf_box = accessible_leaf.box.get_intersection(clique["box"])  # type: ignore
                        if leaf_box.intervals:
                            if (
                                self._classifier.nb_classes > 2
                                and target_label is not None
                                and target_label == accessible_leaf.class_label
                            ):
                                new_leaf_value = -accessible_leaf.value
                            else:
                                new_leaf_value = accessible_leaf.value
                            cliques_new.append(
                                {
                                    "box": leaf_box,
                                    "value": new_leaf_value + clique["value"],  # type: ignore
                                }
                            )

                cliques_old = cliques_new.copy()

            new_nodes = list()
            best_score = 0.0
            for i, clique in enumerate(cliques_old):
                # Create a new node without tree_id and node_id to represent clique
                new_nodes.append(
                    LeafNode(
                        tree_id=None,
                        class_label=label,
                        node_id=None,
                        box=clique["box"],  # type: ignore
                        value=clique["value"],  # type: ignore
                    )
                )

                if i == 0:
                    best_score = clique["value"]  # type: ignore
                else:
                    if label < 0.5 and self._classifier.nb_classes <= 2:
                        best_score = max(best_score, clique["value"])  # type: ignore
                    else:
                        best_score = min(best_score, clique["value"])  # type: ignore

            new_nodes_list.append(new_nodes)
            best_scores_sum += best_score

        return best_scores_sum, new_nodes_list

    def _get_best_score(self, i_sample: int, eps: float, norm: int, target_label: Optional[int]) -> float:
        """
        Get the list of best scores.

        :param i_sample: Index of training sample in `x`.
        :param eps: Attack budget epsilon.
        :param norm: The norm to apply epsilon.
        :param target_label: The target label.
        :return: The best scores.
        """
        nodes = self._get_accessible_leaves(i_sample, eps, norm, target_label)
        best_score: float = 0.0

        for i_level in range(self.max_level):
            if self._classifier.nb_classes > 2 and i_level > 0:
                target_label = None
            best_score, nodes = self._get_k_partite_clique(nodes, label=self.y[i_sample], target_label=target_label)

            # Stop if the root node has been reached
            if len(nodes) <= 1:
                break

        return best_score

    def _get_distance(self, box: Box, i_sample: int, norm: int) -> float:
        """
        Determine the distance between sample and interval box.

        :param box: Interval box.
        :param i_sample: Index of training sample in `x`.
        :param norm: The norm to apply epsilon.
        :return: The distance.
        """
        resulting_distance = 0.0

        for feature, interval in box.intervals.items():
            feature_value = self.x[i_sample, feature]

            if interval.lower_bound < feature_value < interval.upper_bound:
                distance = 0.0
            else:
                difference = max(
                    feature_value - interval.upper_bound,
                    interval.lower_bound - feature_value,
                )
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

        if norm not in [0, np.inf]:
            resulting_distance = pow(resulting_distance, 1.0 / norm)

        return resulting_distance

    def _get_accessible_leaves(
        self, i_sample: int, eps: float, norm: int, target_label: Optional[int]
    ) -> List[List[LeafNode]]:
        """
        Determine the leaf nodes accessible within the attack budget.

        :param i_sample: Index of training sample in `x`.
        :param eps: Attack budget epsilon.
        :param norm: The norm to apply epsilon.
        :param target_label: The target label.
        :return: A list of lists of leaf nodes.
        """
        accessible_leaves = list()

        for tree in self._trees:
            if (
                self._classifier.nb_classes <= 2
                or target_label is None
                or tree.class_id in [self.y[i_sample], target_label]
            ):

                leaves = list()

                for leaf_node in tree.leaf_nodes:
                    distance = self._get_distance(leaf_node.box, i_sample, norm)
                    if leaf_node.box and distance <= eps:
                        leaves.append(leaf_node)

                if not leaves:  # pragma: no cover
                    raise ValueError("No accessible leaves found.")

                accessible_leaves.append(leaves)

        return accessible_leaves
