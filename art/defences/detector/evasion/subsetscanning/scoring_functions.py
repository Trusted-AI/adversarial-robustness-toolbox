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
Scanner scoring functions.
"""
import numpy as np


class ScoringFunctions:
    """
    Scanner scoring functions. These functions are used in the scanner to determine the score of a subset.
    """

    @staticmethod
    def get_score_bj_fast(n_alpha: list, no_records: list, alpha: np.ndarray) -> np.ndarray:
        """
        BerkJones

        :param n_alpha: Number of records less than alpha.
        :param no_records: Number of records.
        :param alpha: Alpha threshold.
        :return: Score.
        """
        score = np.zeros(alpha.shape[0])
        inds_tie = n_alpha == no_records
        inds_not_tie = np.logical_not(inds_tie)
        inds_pos = n_alpha > no_records * alpha
        inds_pos_not_tie = np.logical_and(inds_pos, inds_not_tie)
        score[inds_tie] = no_records[inds_tie] * np.log(np.true_divide(1, alpha[inds_tie]))

        factor1 = n_alpha[inds_pos_not_tie] * np.log(
            np.true_divide(
                n_alpha[inds_pos_not_tie],
                no_records[inds_pos_not_tie] * alpha[inds_pos_not_tie],
            )
        )

        factor2 = no_records[inds_pos_not_tie] - n_alpha[inds_pos_not_tie]

        factor3 = np.log(
            np.true_divide(
                no_records[inds_pos_not_tie] - n_alpha[inds_pos_not_tie],
                no_records[inds_pos_not_tie] * (1 - alpha[inds_pos_not_tie]),
            )
        )

        score[inds_pos_not_tie] = factor1 + factor2 * factor3
        return score

    @staticmethod
    def get_score_hc_fast(n_alpha: list, no_records: list, alpha: np.ndarray) -> np.ndarray:
        """
        Higher criticism
        Similar to a traditional wald test statistic: (Observed - expected) / standard deviation.
        In this case we use the binomial distribution. The observed is N_a.  The expected (under null) is N*a
        and the standard deviation is sqrt(N*a(1-a)).

        :param n_alpha: Number of records less than alpha.
        :param no_records: Number of records.
        :param alpha: Alpha threshold.
        :return: Score.
        """
        score = np.zeros(alpha.shape[0])
        inds = n_alpha > no_records * alpha
        factor1 = n_alpha[inds] - no_records[inds] * alpha[inds]
        factor2 = np.sqrt(no_records[inds] * alpha[inds] * (1.0 - alpha[inds]))
        score[inds] = np.true_divide(factor1, factor2)
        return score

    @staticmethod
    def get_score_ks_fast(n_alpha: list, no_records: list, alpha: np.ndarray) -> np.ndarray:
        """
        KolmarovSmirnov

        :param n_alpha: Number of records less than alpha.
        :param no_records: Number of records.
        :param alpha: Alpha threshold.
        :return: Score.
        """
        score = np.zeros(alpha.shape[0])
        inds = n_alpha > no_records * alpha
        score[inds] = np.true_divide(n_alpha[inds] - no_records[inds] * alpha[inds], np.sqrt(no_records[inds]))
        return score
