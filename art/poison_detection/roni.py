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
This module implements the Reject on Negative Impact (RONI) defense by Nelson et al. (2019)

| Paper link: https://people.eecs.berkeley.edu/~tygar/papers/SML/misleading.learners.pdf
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from copy import deepcopy

import numpy as np
from sklearn.model_selection import train_test_split

from art.poison_detection.ground_truth_evaluator import GroundTruthEvaluator
from art.poison_detection.poison_filtering_defence import PoisonFilteringDefence
from art.utils import performance_diff

logger = logging.getLogger(__name__)


class RONIDefense(PoisonFilteringDefence):
    """
    Close implementation based on description in Nelson
    'Behavior of Machine Learning Algorithms in Adversarial Environments' Ch. 4.4

    | Textbook link: https://people.eecs.berkeley.edu/~adj/publications/paper-files/EECS-2010-140.pdf
    """
    defence_params = ['classifier', 'x_train', 'y_train', 'x_val', 'y_val', 'perf_func', 'calibrated', 'eps']

    def __init__(self, classifier, x_train, y_train, x_val, y_val, perf_func='accuracy', pp_cal=0.2, pp_quiz=0.2,
                 calibrated=True, eps=0.1, **kwargs):
        """
        Create an :class:`.ActivationDefence` object with the provided classifier.

        :param classifier: Model evaluated for poison.
        :type classifier: :class:`art.classifiers.Classifier`
        :param x_train: dataset used to train the classifier.
        :type x_train: `np.ndarray`
        :param y_train: labels used to train the classifier.
        :type y_train: `np.ndarray`
        :param x_val: trusted data points
        :type x_val: `np.ndarray`
        :param y_train: trusted data labels
        :type y_train: `np.ndarray`
        :param perf_func: performance function to use
        :type perf_func: `str` or `callable`
        :param pp_cal: percent of training data used for calibration
        :type pp_cal: `float`
        :param pp_quiz: percent of training data used for quiz set
        :type pp_quiz: `float`
        :param calibrated: True if using the calibrated form of RONI
        :type calibrated: `bool`
        :param eps: performance threshold if using uncalibrated RONI
        :type eps: `float`
        """
        super(RONIDefense, self).__init__(classifier, x_train, y_train)
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
        self.is_clean_lst = list()
        self.set_params(**kwargs)

    def evaluate_defence(self, is_clean, **kwargs):
        """
        Returns confusion matrix.

        :param is_clean: Ground truth, where is_clean[i]=1 means that x_train[i] is clean and is_clean[i]=0 means
                         x_train[i] is poisonous.
        :type is_clean: :class `np.ndarray`
        :param kwargs: A dictionary of defence-specific parameters.
        :type kwargs: `dict`
        :return: JSON object with confusion matrix.
        :rtype: `jsonObject`
        """
        self.set_params(**kwargs)
        if len(self.is_clean_lst) == 0:
            self.detect_poison()

        if is_clean is None or len(is_clean) != len(self.is_clean_lst):
            raise ValueError("Invalid value for is_clean.")

        _, conf_matrix = self.evaluator.analyze_correctness([self.is_clean_lst], [is_clean])
        return conf_matrix

    def detect_poison(self, **kwargs):
        """
        Returns poison detected and a report.

        :param kwargs: A dictionary of detection-specific parameters.
        :type kwargs: `dict`
        :return: (report, is_clean_lst):
                where a report is a dict object that contains information specified by the provenance detection method
                where is_clean is a list, where is_clean_lst[i]=1 means that x_train[i]
                there is clean and is_clean_lst[i]=0, means that x_train[i] was classified as poison.
        :rtype: `tuple`
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
            acc_shift = performance_diff(before_classifier, after_classifier, self.x_quiz, self.y_quiz,
                                         perf_function=self.perf_func)
            # print(acc_shift, median, std_dev)
            if self.is_suspicious(before_classifier, acc_shift):
                self.is_clean_lst[idx] = 0
                report[idx] = acc_shift
            else:
                before_classifier = after_classifier
                x_trusted = np.vstack([x_trusted, x_i])
                y_trusted = np.vstack([y_trusted, y_i])

        return report, self.is_clean_lst

    def is_suspicious(self, before_classifier, perf_shift):
        """
        Returns True if a given performance shift is suspicious

        :param before_classifier: The classifier without untrusted data
        :type before_classifier: `art.classifiers.classifier.Classifier`
        :param perf_shift: a shift in performance
        :type perf_shift: `float`
        :return: True if a given performance shift is suspicious. False otherwise.
        :rtype: `bool`
        """
        if self.calibrated:
            median, std_dev = self.get_calibration_info(before_classifier)
            return perf_shift < median - 3 * std_dev

        return perf_shift < -self.eps

    def get_calibration_info(self, before_classifier):
        """
        Calculate the median and standard deviation of the accuracy shifts caused
        by the calibration set.

        :param before_classifier: The classifier trained without suspicious point
        :type before_classifier: `art.classifiers.classifier.Classifier`
        :return: a tuple consisting of (`median`, `std_dev`)
        :rtype: (`float`, `float`)
        """
        accs = []

        for x_c, y_c in zip(self.x_cal, self.y_cal):
            after_classifier = deepcopy(before_classifier)
            after_classifier.fit(x=np.vstack([self.x_val, x_c]), y=np.vstack([self.y_val, y_c]))
            accs.append(performance_diff(before_classifier, after_classifier, self.x_quiz, self.y_quiz,
                                         perf_function=self.perf_func))

        return np.median(accs), np.std(accs)

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and applies defence-specific checks before saving them as attributes.
        If a parameter is not provided, it takes its default value.
        """
        super(RONIDefense, self).set_params(**kwargs)

        if len(self.x_train) != len(self.y_train):
            raise ValueError("x_train and y_train do not match shape")

        if self.eps < 0:
            raise ValueError("Value of epsilon must be at least 0")

        return True
