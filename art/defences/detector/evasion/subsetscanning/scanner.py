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
Subset scanning based on FGSS
"""
from typing import Callable, Tuple

import numpy as np

from art.defences.detector.evasion.subsetscanning.scoring_functions import ScoringFunctions
from art.defences.detector.evasion.subsetscanning.scanningops import ScanningOps


class Scanner:
    """
    Fast generalized subset scan

    | Paper link: https://www.cs.cmu.edu/~neill/papers/mcfowland13a.pdf
    """

    @staticmethod
    def fgss_individ_for_nets(
        pvalues: np.ndarray,
        a_max: float = 0.5,
        score_function: Callable[[list, list, np.ndarray], np.ndarray] = ScoringFunctions.get_score_bj_fast,
    ) -> Tuple[float, np.ndarray, np.ndarray, float]:
        """
        Finds the highest scoring subset of records and attribute. Return the subsets, the score, and the alpha that
        maximizes the score.

        A simplified, faster, exact method but only useable when scoring an individual input. This method recognizes
        that for an individual input, the priority function for a fixed alpha threshold results in all nodes having
        either priority 1 or 0. That is, the pmax is either below the threshold or not. Due to convexity of the scoring
        function we know elements with tied priority are either all included or all excluded. Therefore, each alpha
        threshold uniquely defines a single subset of nodes to be scored. These are the nodes that have pmax less than
        threshold. This means the individual-input scanner is equivalent to sorting pmax values and iteratively adding
        the next largest pmax. There are at most O(N) of these subsets to consider. Sorting requires O(N ln N). There is
        no iterative ascent required and no special choice of alpha thresholds for speed improvements.

        :param pvalues: pvalue ranges.
        :param a_max: alpha max. determines the significance level threshold
        :param score_function: scoring function
        :return: (best_score, image_sub, node_sub, optimal_alpha)
        """
        pmaxes = np.reshape(pvalues[:, 1], pvalues.shape[0])  # should be number of columns/nodes
        # we can ignore any pmax that is greater than a_max; this makes sorting faster
        # all the pvalues less than equal a_max are kept by nonzero result of the comparison
        potential_thresholds = pmaxes[np.flatnonzero(pmaxes <= a_max)]

        # sorrted_unique provides our alpha thresholds that we will scan
        # count_unique (in cumulative format) will provide the number of observations less than corresponding alpha
        sorted_unique, count_unique = np.unique(potential_thresholds, return_counts=True)

        cumulative_count = np.cumsum(count_unique)
        # In individual input case we have n_a = N, so cumulative count is used for both.
        # sorted_unique provides the increasing alpha values that need to be checked.
        vector_of_scores = score_function(cumulative_count, cumulative_count, sorted_unique)

        # scoring completed, now grab the max (and index)
        best_score_idx = np.argmax(vector_of_scores)
        best_score = vector_of_scores[best_score_idx]
        optimal_alpha = sorted_unique[best_score_idx]
        # best_size = cumulative_count[best_score_idx]

        # In order to determine which nodes are included, we look for all pmaxes less than or equal best alpha
        node_sub = np.flatnonzero(pvalues[:, 1] <= optimal_alpha)
        # in the individual input case there's only 1 possible subset of inputs to return - a 1x1 with index 0
        image_sub = np.array([0])

        return best_score, image_sub, node_sub, optimal_alpha

    @staticmethod
    def fgss_for_nets(
        pvalues: np.ndarray,
        a_max: float = 0.5,
        restarts: int = 10,
        image_to_node_init: bool = False,
        score_function: Callable[[list, list, np.ndarray], np.ndarray] = ScoringFunctions.get_score_bj_fast,
    ) -> Tuple[float, np.ndarray, np.ndarray, float]:
        """
        Finds the highest scoring subset of records and attribute. Return the subsets, the score, and the alpha that
        maximizes the score iterates between images and nodes, each time performing NPSS efficient maximization.

        :param pvalues: pvalue ranges.
        :param a_max: alpha threshold
        :param restarts: number of iterative restarts
        :param image_to_node_init: intializes what direction to begin the search: image to node or vice-versa
        :param score_function: scoring function
        :return: (best_score, image_sub, node_sub, optimal_alpha)
        """
        best_score = -100000.0

        if len(pvalues) < restarts:
            restarts = len(pvalues)

        for r_indx in range(0, restarts):  # do random restarts to come close to global maximum
            image_to_node = image_to_node_init
            if r_indx == 0:
                if image_to_node:
                    # all 1's for number of rows
                    indices_of_seeds = np.arange(pvalues.shape[0])
                else:
                    # all 1's for number of cols
                    indices_of_seeds = np.arange(pvalues.shape[1])

                (
                    best_score_from_restart,
                    best_image_sub_from_restart,
                    best_node_sub_from_restart,
                    best_alpha_from_restart,
                ) = ScanningOps.single_restart(pvalues, a_max, indices_of_seeds, image_to_node, score_function)

                if best_score_from_restart > best_score:
                    best_score = best_score_from_restart
                    image_sub = best_image_sub_from_restart
                    node_sub = best_node_sub_from_restart
                    optimal_alpha = best_alpha_from_restart

                # Finished A Restart
            else:
                # New Restart
                # some some randomizing and only leave in a random number of rows of pvalues TODO
                prob = np.random.uniform(0, 1)
                if image_to_node:
                    indices_of_seeds = np.random.choice(
                        np.arange(pvalues.shape[0]),
                        int(pvalues.shape[0] * prob),
                        replace=False,
                    )
                else:
                    indices_of_seeds = np.random.choice(
                        np.arange(pvalues.shape[1]),
                        int(pvalues.shape[1] * prob),
                        replace=False,
                    )
                while indices_of_seeds.size == 0:
                    # eventually will make non zero
                    prob = np.random.uniform(0, 1)
                    if image_to_node:
                        indices_of_seeds = np.random.choice(
                            np.arange(pvalues.shape[0]),
                            int(pvalues.shape[0] * prob),
                            replace=False,
                        )
                    else:
                        indices_of_seeds = np.random.choice(
                            np.arange(pvalues.shape[1]),
                            int(pvalues.shape[1] * prob),
                            replace=False,
                        )

                indices_of_seeds.astype(int)
                # process a random subset of rows of pvalues array
                (
                    best_score_from_restart,
                    best_image_sub_from_restart,
                    best_node_sub_from_restart,
                    best_alpha_from_restart,
                ) = ScanningOps.single_restart(pvalues, a_max, indices_of_seeds, image_to_node, score_function)

                if best_score_from_restart > best_score:
                    best_score = best_score_from_restart
                    image_sub = best_image_sub_from_restart
                    node_sub = best_node_sub_from_restart
                    optimal_alpha = best_alpha_from_restart

                # Finished A Restart

        return best_score, image_sub, node_sub, optimal_alpha
