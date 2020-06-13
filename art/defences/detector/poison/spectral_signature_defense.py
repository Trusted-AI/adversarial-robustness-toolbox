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

import numpy as np
from typing import List, Tuple, TYPE_CHECKING

from art.defences.detector.poison.ground_truth_evaluator import GroundTruthEvaluator
from art.defences.detector.poison.poison_filtering_defence import PoisonFilteringDefence

if TYPE_CHECKING:
    from art.estimators.classification.classifier import Classifier


class SpectralSignatureDefense(PoisonFilteringDefence):
    """
    Method from Tran et al., 2018 performing poisoning detection based on Spectral Signatures
    """

    defence_params = PoisonFilteringDefence.defence_params + [
        "x_train",
        "y_train",
        "batch_size",
        "eps_multiplier",
        "ub_pct_poison",
        "nb_classes",
    ]

    def __init__(
        self,
        classifier: "Classifier",
        x_train: np.ndarray,
        y_train: np.ndarray,
        batch_size: int,
        eps_multiplier: float,
        ub_pct_poison,
        nb_classes: int,
    ) -> None:
        """
        Create an :class:`.SpectralSignatureDefense` object with the provided classifier.

        :param classifier: Model evaluated for poison.
        :param x_train: Dataset used to train the classifier.
        :param y_train: Labels used to train the classifier.
        :param batch_size: Size of batches.
        :param eps_multiplier:
        :param ub_pct_poison:
        :param nb_classes: Number of classes.
        """
        super().__init__(classifier, x_train, y_train)
        self.batch_size = batch_size
        self.eps_multiplier = eps_multiplier
        self.ub_pct_poison = ub_pct_poison
        self.nb_classes = nb_classes
        self.y_train_sparse = np.argmax(y_train, axis=1)
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
        is_clean_by_class = SpectralSignatureDefense.split_by_class(is_clean, self.y_train_sparse, self.nb_classes)
        _, predicted_clean = self.detect_poison()
        predicted_clean_by_class = SpectralSignatureDefense.split_by_class(
            predicted_clean, self.y_train_sparse, self.nb_classes
        )

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

        nb_layers = len(self.classifier.layer_names)
        features_x_poisoned = self.classifier.get_activations(
            self.x_train, layer=nb_layers - 1, batch_size=self.batch_size
        )

        features_split = SpectralSignatureDefense.split_by_class(
            features_x_poisoned, self.y_train_sparse, self.nb_classes
        )
        score_by_class, keep_by_class = [], []
        for idx, feature in enumerate(features_split):
            score = SpectralSignatureDefense.spectral_signature_scores(feature)
            score_cutoff = np.quantile(score, max(1 - self.eps_multiplier * self.ub_pct_poison, 0.0))
            score_by_class.append(score)
            keep_by_class.append(score < score_cutoff)

        base_indices_by_class = SpectralSignatureDefense.split_by_class(
            np.arange(self.y_train_sparse.shape[0]), self.y_train_sparse, self.nb_classes,
        )
        is_clean_lst = np.zeros_like(self.y_train_sparse, dtype=np.int)
        report = {}

        for keep_booleans, all_scores, indices in zip(keep_by_class, score_by_class, base_indices_by_class):
            for keep_boolean, all_score, idx in zip(keep_booleans, all_scores, indices):
                if keep_boolean:
                    is_clean_lst[idx] = 1
                else:
                    report[idx] = all_score[0]
        return report, is_clean_lst

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
        score = np.matmul(matrix_m, np.transpose(eigs)) ** 2
        return score

    @staticmethod
    def split_by_class(data: np.ndarray, labels: np.ndarray, num_classes: int) -> List[np.ndarray]:
        """
        :param data: Features.
        :param labels: Labels, not in one-hot representations.
        :param num_classes: Number of classes of labels.
        :return: List of numpy arrays of features split by labels.
        """
        split: List[List[int]] = [[] for _ in range(num_classes)]
        for idx, label in enumerate(labels):
            split[int(label)].append(data[idx])
        return [np.asarray(dat) for dat in split]

    def _check_params(self) -> None:
        if self.batch_size < 0:
            raise ValueError("Batch size must be positive integer. Unsupported batch size: " + str(self.batch_size))
        if self.eps_multiplier < 0:
            raise ValueError("eps_multiplier must be positive. Unsupported value: " + str(self.eps_multiplier))
        if self.ub_pct_poison < 0 or self.ub_pct_poison > 1:
            raise ValueError("ub_pct_poison must be between 0 and 1. Unsupported value: " + str(self.ub_pct_poison))
