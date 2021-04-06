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
Scanning operations
"""
from typing import Callable, Tuple

import numpy as np


class ScanningOps:
    """
    Specific operations done during scanning
    """

    @staticmethod
    def optimize_in_single_dimension(
        pvalues: np.ndarray,
        a_max: float,
        image_to_node: bool,
        score_function: Callable[[list, list, np.ndarray], np.ndarray],
    ) -> Tuple[float, np.ndarray, float]:
        """
        Optimizes over all subsets of nodes for a given subset of images or over all subsets of images for a given
        subset of nodes.

        :param pvalues: pvalue ranges.
        :param a_max: Determines the significance level threshold.
        :param image_to_node: Informs the direction for optimization.
        :param score_function: Scoring function.
        :return: (best_score_so_far, subset, best_alpha).
        """
        alpha_thresholds = np.unique(pvalues[:, :, 1])

        # alpha_thresholds = alpha_thresholds[0::5] #take every 5th for speed purposes
        # where does a_max fall in check
        last_alpha_index = np.searchsorted(alpha_thresholds, a_max)
        # resize check for only ones smaller than a_max
        alpha_thresholds = alpha_thresholds[0:last_alpha_index]

        step_for_50 = len(alpha_thresholds) / 50
        alpha_thresholds = alpha_thresholds[0 :: int(step_for_50) + 1]
        # add on the max value to check as well as it may not have been part of unique
        alpha_thresholds = np.append(alpha_thresholds, a_max)

        # alpha_thresholds = np.arange(a_max/50, a_max, a_max/50)

        if image_to_node:
            number_of_elements = pvalues.shape[1]  # searching over j columns
            size_of_given = pvalues.shape[0]  # for fixed this many images
            unsort_priority = np.zeros((pvalues.shape[1], alpha_thresholds.shape[0]))  # number of columns
        else:
            number_of_elements = pvalues.shape[0]  # searching over i rows
            size_of_given = pvalues.shape[1]  # for this many fixed nodes
            unsort_priority = np.zeros((pvalues.shape[0], alpha_thresholds.shape[0]))  # number of rows

        for elem_indx in range(0, number_of_elements):
            # sort all the range maxes
            if image_to_node:
                # collect ranges over images(rows)
                arg_sort_max = np.argsort(pvalues[:, elem_indx, 1])
                # arg_sort_min = np.argsort(pvalues[:,e,0]) #collect ranges over images(rows)
                completely_included = np.searchsorted(
                    pvalues[:, elem_indx, 1][arg_sort_max],
                    alpha_thresholds,
                    side="right",
                )
            else:
                # collect ranges over nodes(columns)
                arg_sort_max = np.argsort(pvalues[elem_indx, :, 1])
                # arg_sort_min = np.argsort(pvalues[elem_indx,:,0])

                completely_included = np.searchsorted(
                    pvalues[elem_indx, :, 1][arg_sort_max],
                    alpha_thresholds,
                    side="right",
                )

            # should be num elements by num thresh
            unsort_priority[elem_indx, :] = completely_included

        # want to sort for a fixed thresh (across?)
        arg_sort_priority = np.argsort(-unsort_priority, axis=0)

        best_score_so_far = -10000
        best_alpha = -2

        alpha_count = 0
        for alpha_threshold in alpha_thresholds:

            # score each threshold by itself, cumulating priority,
            # cumulating count, alpha stays same.
            alpha_v = np.ones(number_of_elements) * alpha_threshold

            n_alpha_v = np.cumsum(unsort_priority[:, alpha_count][arg_sort_priority][:, alpha_count])
            count_increments_this = np.ones(number_of_elements) * size_of_given
            n_v = np.cumsum(count_increments_this)

            vector_of_scores = score_function(n_alpha_v, n_v, alpha_v)

            best_score_for_this_alpha_idx = np.argmax(vector_of_scores)
            best_score_for_this_alpha = vector_of_scores[best_score_for_this_alpha_idx]

            if best_score_for_this_alpha > best_score_so_far:
                best_score_so_far = best_score_for_this_alpha
                best_size = best_score_for_this_alpha_idx + 1  # not sure 1 is needed?
                best_alpha = alpha_threshold
                best_alpha_count = alpha_count
            alpha_count = alpha_count + 1

        # after the alpha for loop we now have best score, best alpha, size of best subset,
        # and alpha counter use these with the priority argsort to reconstruct the best subset
        unsort = arg_sort_priority[:, best_alpha_count]

        subset = np.zeros(best_size).astype(int)
        for loc in range(0, best_size):
            subset[loc] = unsort[loc]

        return best_score_so_far, subset, best_alpha

    @staticmethod
    def single_restart(
        pvalues: np.ndarray,
        a_max: float,
        indices_of_seeds: np.ndarray,
        image_to_node: bool,
        score_function: Callable[[list, list, np.ndarray], np.ndarray],
    ) -> Tuple[float, np.ndarray, np.ndarray, float]:
        """
        Here we control the iteration between images->nodes and nodes->images. It starts with a fixed subset of nodes by
        default.

        :param pvalues: pvalue ranges.
        :param a_max: Determines the significance level threshold.
        :param indices_of_seeds: Indices of initial sets of images or nodes to perform optimization.
        :param image_to_node: Informs the direction for optimization.
        :param score_function: Scoring function.
        :return: (best_score_so_far, best_sub_of_images, best_sub_of_nodes, best_alpha).
        """
        best_score_so_far = -100000.0
        count = 0

        while True:
            # These can be moved outside the while loop as only executed first time through??
            if count == 0:  # first time through, we need something initialized depending on direction.
                if image_to_node:
                    sub_of_images = indices_of_seeds
                else:
                    sub_of_nodes = indices_of_seeds

            if image_to_node:  # passed pvalues are only those belonging to fixed images, update nodes in return
                # only sending sub of images
                (score_from_optimization, sub_of_nodes, optimal_alpha,) = ScanningOps.optimize_in_single_dimension(
                    pvalues[sub_of_images, :, :], a_max, image_to_node, score_function
                )
            else:  # passed pvalues are only those belonging to fixed nodes, update images in return
                # only sending sub of nodes
                (score_from_optimization, sub_of_images, optimal_alpha,) = ScanningOps.optimize_in_single_dimension(
                    pvalues[:, sub_of_nodes, :], a_max, image_to_node, score_function
                )

            if score_from_optimization > best_score_so_far:  # haven't converged yet
                # update
                best_score_so_far = score_from_optimization
                best_sub_of_nodes = sub_of_nodes
                best_sub_of_images = sub_of_images
                best_alpha = optimal_alpha

                image_to_node = not image_to_node  # switch direction!
                count = count + 1  # for printing and
            else:  # converged!  Don't update from most recent optimization, return current best
                return (
                    best_score_so_far,
                    best_sub_of_images,
                    best_sub_of_nodes,
                    best_alpha,
                )
