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
This module implements the Reject on Negative Impact (RONI) defense by Nelson et al. (2019)

| Paper link: https://people.eecs.berkeley.edu/~tygar/papers/SML/misleading.learners.pdf
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from copy import deepcopy
from typing import Callable, List, Tuple, Union, TYPE_CHECKING

import numpy as np
from sklearn.model_selection import train_test_split

from art.defences.detector.poison.ground_truth_evaluator import GroundTruthEvaluator
from art.defences.detector.poison.poison_filtering_defence import PoisonFilteringDefence
from art.utils import performance_diff

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE

logger = logging.getLogger(__name__)


class RONIDefense(PoisonFilteringDefence):
    """
    Close implementation based on description in Nelson
    'Behavior of Machine Learning Algorithms in Adversarial Environments' Ch. 4.4

    | Textbook link: https://people.eecs.berkeley.edu/~adj/publications/paper-files/EECS-2010-140.pdf
    """

    defence_params = [
        "classifier",
        "x_train",
        "y_train",
        "x_val",
        "y_val",
        "perf_func",
        "calibrated",
        "eps",
    ]

    def __init__(
        self,
        classifier: "CLASSIFIER_TYPE",
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        perf_func: Union[str, Callable] = "accuracy",
        pp_cal: float = 0.2,
        pp_quiz: float = 0.2,
        calibrated: bool = True,
        eps: float = 0.1,
    ):
        """
        Create an :class:`.RONIDefense` object with the provided classifier.

        :param classifier: Model evaluated for poison.
        :param x_train: Dataset used to train the classifier.
        :param y_train: Labels used to train the classifier.
        :param x_val: Trusted data points.
        :param y_train: Trusted data labels.
        :param perf_func: Performance function to use.
        :param pp_cal: Percent of training data used for calibration.
        :param pp_quiz: Percent of training data used for quiz set.
        :param calibrated: True if using the calibrated form of RONI.
        :param eps: performance threshold if using uncalibrated RONI.
        """
        super().__init__(classifier, x_train, y_train)
        n_points = len(x_train)
        quiz_idx = np.random.randint(n_points, size=int(pp_quiz * n_points))
        self.calibrated = calibrated
        self.x_quiz = np.copy(self.x_train[quiz_idx])
        self.y_quiz = np.copy(self.y_train[quiz_idx])
        if self.calibrated:
            _, self.x_cal, _, self.y_cal = train_test_split(self.x_train, self.y_train, test_size=pp_cal, shuffle=True)
        self.eps = eps
        self.evaluator = GroundTruthEvaluator()
        self.x_val = x_val
        self.y_val = y_val
        self.perf_func = perf_func
        self.is_clean_lst: List[int] = list()
        self._check_params()

    def evaluate_defence(self, is_clean: np.ndarray, **kwargs) -> str:
        """
        Returns confusion matrix.

        :param is_clean: Ground truth, where is_clean[i]=1 means that x_train[i] is clean and is_clean[i]=0 means
                         x_train[i] is poisonous.
        :param kwargs: A dictionary of defence-specific parameters.
        :return: JSON object with confusion matrix.
        """
        self.set_params(**kwargs)
        if len(self.is_clean_lst) == 0:
            self.detect_poison()

        if is_clean is None or len(is_clean) != len(self.is_clean_lst):
            raise ValueError("Invalid value for is_clean.")

        _, conf_matrix = self.evaluator.analyze_correctness([self.is_clean_lst], [is_clean])
        return conf_matrix

    def detect_poison(self, **kwargs) -> Tuple[dict, List[int]]:
        """
        Returns poison detected and a report.

        :param kwargs: A dictionary of detection-specific parameters.
        :return: (report, is_clean_lst):
                where a report is a dict object that contains information specified by the provenance detection method
                where is_clean is a list, where is_clean_lst[i]=1 means that x_train[i]
                there is clean and is_clean_lst[i]=0, means that x_train[i] was classified as poison.
        """
        self.set_params(**kwargs)

        x_suspect = self.x_train
        y_suspect = self.y_train
        x_trusted = self.x_val
        y_trusted = self.y_val

        self.is_clean_lst = [1 for _ in range(len(x_suspect))]
        report = {}

        before_classifier = deepcopy(self.classifier)
        before_classifier.fit(x_suspect, y_suspect)

        for idx in np.random.permutation(len(x_suspect)):
            x_i = x_suspect[idx]
            y_i = y_suspect[idx]

            after_classifier = deepcopy(before_classifier)
            after_classifier.fit(x=np.vstack([x_trusted, x_i]), y=np.vstack([y_trusted, y_i]))
            acc_shift = performance_diff(
                before_classifier,
                after_classifier,
                self.x_quiz,
                self.y_quiz,
                perf_function=self.perf_func,
            )
            # print(acc_shift, median, std_dev)
            if self.is_suspicious(before_classifier, acc_shift):
                self.is_clean_lst[idx] = 0
                report[idx] = acc_shift
            else:
                before_classifier = after_classifier
                x_trusted = np.vstack([x_trusted, x_i])
                y_trusted = np.vstack([y_trusted, y_i])

        return report, self.is_clean_lst

    def is_suspicious(self, before_classifier: "CLASSIFIER_TYPE", perf_shift: float) -> bool:
        """
        Returns True if a given performance shift is suspicious

        :param before_classifier: The classifier without untrusted data.
        :param perf_shift: A shift in performance.
        :return: True if a given performance shift is suspicious, false otherwise.
        """
        if self.calibrated:
            median, std_dev = self.get_calibration_info(before_classifier)
            return perf_shift < median - 3 * std_dev

        return perf_shift < -self.eps

    def get_calibration_info(self, before_classifier: "CLASSIFIER_TYPE") -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the median and standard deviation of the accuracy shifts caused
        by the calibration set.

        :param before_classifier: The classifier trained without suspicious point.
        :return: A tuple consisting of `(median, std_dev)`.
        """
        accs = []

        for x_c, y_c in zip(self.x_cal, self.y_cal):
            after_classifier = deepcopy(before_classifier)
            after_classifier.fit(x=np.vstack([self.x_val, x_c]), y=np.vstack([self.y_val, y_c]))
            accs.append(
                performance_diff(
                    before_classifier,
                    after_classifier,
                    self.x_quiz,
                    self.y_quiz,
                    perf_function=self.perf_func,
                )
            )

        return np.median(accs), np.std(accs)

    def _check_params(self) -> None:
        if len(self.x_train) != len(self.y_train):
            raise ValueError("`x_train` and `y_train` do not match shape.")

        if self.eps < 0:
            raise ValueError("Value of `eps` must be at least 0.")
