# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
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
This module implements methods performing backdoor poisoning detection based on spectral signatures.

| Paper link: https://papers.nips.cc/paper/8024-spectral-signatures-in-backdoor-attacks.pdf

| Please keep in mind the limitations of defenses. For more information on the limitations of this
    specific defense, see https://arxiv.org/abs/1905.13409 .
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from typing import List, Tuple, TYPE_CHECKING

import numpy as np

from art.defences.detector.poison.ground_truth_evaluator import GroundTruthEvaluator
from art.defences.detector.poison.poison_filtering_defence import PoisonFilteringDefence

from art.utils import segment_by_class

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_NEURALNETWORK_TYPE


class SpectralSignatureDefense(PoisonFilteringDefence):
    """
    Method from Tran et al., 2018 performing poisoning detection based on Spectral Signatures
    """

    defence_params = PoisonFilteringDefence.defence_params + [
        "x_train",
        "y_train",
        "batch_size",
        "eps_multiplier",
        "expected_pp_poison",
    ]

    def __init__(
        self,
        classifier: "CLASSIFIER_NEURALNETWORK_TYPE",
        x_train: np.ndarray,
        y_train: np.ndarray,
        expected_pp_poison: float = 0.33,
        batch_size: int = 128,
        eps_multiplier: float = 1.5,
    ) -> None:
        """
        Create an :class:`.SpectralSignatureDefense` object with the provided classifier.

        :param classifier: Model evaluated for poison.
        :param x_train: Dataset used to train the classifier.
        :param y_train: Labels used to train the classifier.
        :param expected_pp_poison: The expected percentage of poison in the dataset
        :param batch_size: The batch size for predictions
        :param eps_multiplier: The multiplier to add to the previous expectation. Numbers higher than one represent
                               a potentially higher false positive rate, but may detect more poison samples
        """
        super().__init__(classifier, x_train, y_train)
        self.classifier: "CLASSIFIER_NEURALNETWORK_TYPE" = classifier
        self.batch_size = batch_size
        self.eps_multiplier = eps_multiplier
        self.expected_pp_poison = expected_pp_poison
        self.y_train = y_train
        self.evaluator = GroundTruthEvaluator()
        self._check_params()

    def evaluate_defence(self, is_clean: np.ndarray, **kwargs) -> str:
        """
        If ground truth is known, this function returns a confusion matrix in the form of a JSON object.

        :param is_clean: Ground truth, where is_clean[i]=1 means that x_train[i] is clean and is_clean[i]=0 means
                         x_train[i] is poisonous.
        :param kwargs: A dictionary of defence-specific parameters.
        :return: JSON object with confusion matrix.
        """
        if is_clean is None or is_clean.size == 0:
            raise ValueError("is_clean was not provided while invoking evaluate_defence.")
        is_clean_by_class = segment_by_class(is_clean, self.y_train, self.classifier.nb_classes)
        _, predicted_clean = self.detect_poison()
        predicted_clean_by_class = segment_by_class(predicted_clean, self.y_train, self.classifier.nb_classes)

        _, conf_matrix_json = self.evaluator.analyze_correctness(predicted_clean_by_class, is_clean_by_class)

        return conf_matrix_json

    def detect_poison(self, **kwargs) -> Tuple[dict, List[int]]:
        """
        Returns poison detected and a report.

        :return: (report, is_clean_lst):
                where a report is a dictionary containing the index as keys the outlier score of suspected poisons as
                values where is_clean is a list, where is_clean_lst[i]=1 means that x_train[i] there is clean and
                is_clean_lst[i]=0, means that x_train[i] was classified as poison.
        """
        self.set_params(**kwargs)

        if self.classifier.layer_names is not None:
            nb_layers = len(self.classifier.layer_names)
        else:
            raise ValueError("No layer names identified.")
        features_x_poisoned = self.classifier.get_activations(
            self.x_train, layer=nb_layers - 1, batch_size=self.batch_size
        )

        features_split = segment_by_class(features_x_poisoned, self.y_train, self.classifier.nb_classes)
        score_by_class = []
        keep_by_class = []

        for idx, feature in enumerate(features_split):
            # Check for empty list
            if len(feature):  # pylint: disable=C1801
                score = SpectralSignatureDefense.spectral_signature_scores(np.vstack(feature))
                score_cutoff = np.quantile(score, max(1 - self.eps_multiplier * self.expected_pp_poison, 0.0))
                score_by_class.append(score)
                keep_by_class.append(score < score_cutoff)
            else:
                score_by_class.append([0])
                keep_by_class.append([True])

        base_indices_by_class = segment_by_class(
            np.arange(self.y_train.shape[0]),
            self.y_train,
            self.classifier.nb_classes,
        )
        is_clean_lst = [0] * self.y_train.shape[0]
        report = {}

        for keep_booleans, all_scores, indices in zip(keep_by_class, score_by_class, base_indices_by_class):
            for keep_boolean, all_score, idx in zip(keep_booleans, all_scores, indices):
                if keep_boolean:
                    is_clean_lst[idx] = 1
                else:
                    report[idx] = all_score[0]

        return report, is_clean_lst

    def _check_params(self) -> None:
        if self.batch_size < 0:
            raise ValueError("Batch size must be positive integer. Unsupported batch size: " + str(self.batch_size))
        if self.eps_multiplier < 0:
            raise ValueError("eps_multiplier must be positive. Unsupported value: " + str(self.eps_multiplier))
        if self.expected_pp_poison < 0 or self.expected_pp_poison > 1:
            raise ValueError(
                "expected_pp_poison must be between 0 and 1. Unsupported value: " + str(self.expected_pp_poison)
            )

    @staticmethod
    def spectral_signature_scores(matrix_r: np.ndarray) -> np.ndarray:
        """
        :param matrix_r: Matrix of feature representations.
        :return: Outlier scores for each observation based on spectral signature.
        """
        matrix_m = matrix_r - np.mean(matrix_r, axis=0)
        # Following Algorithm #1 in paper, use SVD of centered features, not of covariance
        _, _, matrix_v = np.linalg.svd(matrix_m, full_matrices=False)
        eigs = matrix_v[:1]
        corrs = np.matmul(eigs, np.transpose(matrix_r))
        score = np.expand_dims(np.linalg.norm(corrs, axis=0), axis=1)
        return score
