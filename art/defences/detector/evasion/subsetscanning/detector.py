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
This module implements the fast generalized subset scan based detector.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import sys
from typing import Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
from sklearn import metrics
from tqdm.auto import trange, tqdm

from art.defences.detector.evasion.evasion_detector import EvasionDetector
from art.defences.detector.evasion.subsetscanning.scanner import Scanner
from art.defences.detector.evasion.subsetscanning.scoring_functions import ScoringFunctions

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_NEURALNETWORK_TYPE

logger = logging.getLogger(__name__)


class SubsetScanningDetector(EvasionDetector):
    """
    Fast generalized subset scan based detector by McFowland, E., Speakman, S., and Neill, D. B. (2013).

    | Paper link: https://www.cs.cmu.edu/~neill/papers/mcfowland13a.pdf
    """

    defence_params = ["classifier", "bgd_data", "layer", "scoring_function", "verbose"]

    def __init__(
        self,
        classifier: "CLASSIFIER_NEURALNETWORK_TYPE",
        bgd_data: np.ndarray,
        layer: Union[int, str],
        scoring_function: Literal["BerkJones", "HigherCriticism", "KolmarovSmirnov"] = "BerkJones",
        verbose: bool = True,
    ) -> None:
        """
        Create a `SubsetScanningDetector` instance which is used to the detect the presence of adversarial samples.

        :param classifier: The model being evaluated for its robustness to anomalies (e.g. adversarial samples).
        :param bgd_data: The background data used to learn a null model. Typically dataset used to train the classifier.
        :param layer: The layer from which to extract activations to perform scan.
        :param verbose: Show progress bars.
        """
        super().__init__()
        self.classifier = classifier
        self.bgd_data = bgd_data
        self.layer = layer
        self.verbose = verbose

        if scoring_function == "BerkJones":
            self.scoring_function = ScoringFunctions.get_score_bj_fast
        elif scoring_function == "HigherCriticism":
            self.scoring_function = ScoringFunctions.get_score_hc_fast
        elif scoring_function == "KolmarovSmirnov":
            self.scoring_function = ScoringFunctions.get_score_ks_fast
        else:
            raise ValueError("The provided scoring function is not supported:", scoring_function)

        # Ensure that layer is well-defined
        if classifier.layer_names is None:
            raise ValueError("No layer names identified.")

        if isinstance(layer, int):
            if layer < 0 or layer >= len(classifier.layer_names):
                raise ValueError(
                    f"Layer index {layer} is outside of range (0 to {len(classifier.layer_names) - 1} included)."
                )
            self._layer_name = classifier.layer_names[layer]
        else:
            if layer not in classifier.layer_names:
                raise ValueError(f"Layer name {layer} is not part of the graph.")
            self._layer_name = layer

        # Background data activations
        bgd_activations = self._get_activations(bgd_data, self._layer_name, batch_size=128)
        if len(bgd_activations.shape) == 4:
            dim2 = bgd_activations.shape[1] * bgd_activations.shape[2] * bgd_activations.shape[3]
            bgd_activations = np.reshape(bgd_activations, (bgd_activations.shape[0], dim2))
        self.sorted_bgd_activations = np.sort(bgd_activations, axis=0)

        # Background data scores
        pval_ranges = self._calculate_pvalue_ranges(bgd_data)
        bgd_scores = []
        for pval_range in pval_ranges:
            best_score, _, _, _ = Scanner.fgss_individ_for_nets(pval_range, score_function=self.scoring_function)
            bgd_scores.append(best_score)
        self.bgd_scores = np.asarray(bgd_scores)

    def _get_activations(
        self, x: np.ndarray, layer: Union[int, str], batch_size: int, framework: bool = False
    ) -> np.ndarray:
        x_activations = self.classifier.get_activations(x, layer, batch_size, framework)
        if x_activations is None:
            raise ValueError("Classifier activations are null.")

        if isinstance(x_activations, np.ndarray):
            return x_activations

        return x_activations.numpy()

    def _calculate_pvalue_ranges(self, x: np.ndarray, batch_size: int = 128) -> np.ndarray:
        """
        Returns computed p-value ranges.

        :param x: Data being evaluated for anomalies.
        :return: p-value ranges.
        """
        bgd_activations = self.sorted_bgd_activations
        eval_activations = self._get_activations(x, self._layer_name, batch_size)

        if len(eval_activations.shape) == 4:
            dim2 = eval_activations.shape[1] * eval_activations.shape[2] * eval_activations.shape[3]
            eval_activations = np.reshape(eval_activations, (eval_activations.shape[0], dim2))

        bgrecords_n = bgd_activations.shape[0]
        records_n = eval_activations.shape[0]
        atrr_n = eval_activations.shape[1]

        pvalue_ranges = np.empty((records_n, atrr_n, 2))

        for j in range(atrr_n):
            pvalue_ranges[:, j, 0] = np.searchsorted(bgd_activations[:, j], eval_activations[:, j], side="right")
            pvalue_ranges[:, j, 1] = np.searchsorted(bgd_activations[:, j], eval_activations[:, j], side="left")

        pvalue_ranges = bgrecords_n - pvalue_ranges

        pvalue_ranges[:, :, 0] = np.divide(pvalue_ranges[:, :, 0], bgrecords_n + 1)
        pvalue_ranges[:, :, 1] = np.divide(pvalue_ranges[:, :, 1] + 1, bgrecords_n + 1)

        return pvalue_ranges

    def scan(
        self,
        clean_x: np.ndarray,
        adv_x: np.ndarray,
        clean_size: Optional[int] = None,
        adv_size: Optional[int] = None,
        run: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Returns scores of highest scoring subsets.

        :param clean_x: Data presumably without anomalies.
        :param adv_x: Data presumably with anomalies (adversarial samples).
        :param clean_size:
        :param adv_size:
        :param run:
        :return: (clean_scores, adv_scores, detection_power).
        """
        clean_pval_ranges = self._calculate_pvalue_ranges(clean_x)
        adv_pval_ranges = self._calculate_pvalue_ranges(adv_x)

        clean_scores = []
        adv_scores = []

        if clean_size is None or adv_size is None:
            # Individual scan
            with tqdm(
                total=len(clean_pval_ranges) + len(adv_pval_ranges), desc="Subset scanning", disable=not self.verbose
            ) as pbar:
                for _, c_p in enumerate(clean_pval_ranges):
                    best_score, _, _, _ = Scanner.fgss_individ_for_nets(c_p, score_function=self.scoring_function)
                    clean_scores.append(best_score)
                    pbar.update(1)
                for _, a_p in enumerate(adv_pval_ranges):
                    best_score, _, _, _ = Scanner.fgss_individ_for_nets(a_p, score_function=self.scoring_function)
                    adv_scores.append(best_score)
                    pbar.update(1)

        else:
            len_adv_x = len(adv_x)
            len_clean_x = len(clean_x)

            for _ in trange(run, desc="Subset scanning", disable=not self.verbose):
                np.random.seed()

                clean_choice = np.random.choice(range(len_clean_x), clean_size, replace=False)
                adv_choice = np.random.choice(range(len_adv_x), adv_size, replace=False)

                combined_pvals = np.concatenate((clean_pval_ranges[clean_choice], adv_pval_ranges[adv_choice]), axis=0)

                best_score, _, _, _ = Scanner.fgss_for_nets(
                    clean_pval_ranges[clean_choice], score_function=self.scoring_function
                )
                clean_scores.append(best_score)
                best_score, _, _, _ = Scanner.fgss_for_nets(combined_pvals, score_function=self.scoring_function)
                adv_scores.append(best_score)

        clean_scores_array = np.asarray(clean_scores)
        adv_scores_array = np.asarray(adv_scores)

        y_true = np.concatenate([np.ones(len(adv_scores)), np.zeros(len(clean_scores))])
        all_scores = np.concatenate([adv_scores, clean_scores])

        fpr, tpr, _ = metrics.roc_curve(y_true, all_scores)
        roc_auc: float = metrics.auc(fpr, tpr)
        detection_power = roc_auc

        return clean_scores_array, adv_scores_array, detection_power

    def detect(self, x: np.ndarray, batch_size: int = 128, **kwargs) -> Tuple[dict, np.ndarray]:
        """
        Perform detection of adversarial data and return prediction as tuple.

        :param x: Data sample on which to perform detection.
        :param batch_size: Size of batches.
        :return: (report, is_adversarial):
                where report is a dictionary containing contains information specified by the subset scanning method;
                where is_adversarial is a boolean list of per-sample prediction whether the sample is adversarial
                or not and has the same `batch_size` (first dimension) as `x`.
        """
        pval_ranges = self._calculate_pvalue_ranges(x, batch_size)
        scores = []

        for pval_range in tqdm(pval_ranges, desc="Subset scanning", disable=not self.verbose):
            best_score, _, _, _ = Scanner.fgss_individ_for_nets(pval_range, score_function=self.scoring_function)
            scores.append(best_score)
        scores_array = np.asarray(scores)

        is_adversarial = np.greater(scores_array, self.bgd_scores.max())
        report = {"scores": scores_array}

        return report, is_adversarial

    def fit(self, x: np.ndarray, y: np.ndarray, batch_size: int = 128, nb_epochs: int = 20, **kwargs) -> None:
        """
        Fit the detector using training data. Assumes that the classifier is already trained.

        :raises `NotImplementedException`: This method is not supported for this detector.
        """
        raise NotImplementedError
