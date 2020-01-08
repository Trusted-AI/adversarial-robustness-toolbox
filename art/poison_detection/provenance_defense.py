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
This module implements methods performing poisoning detection based on data provenance.

| Paper link: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8473440
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from copy import deepcopy

import numpy as np
from sklearn.model_selection import train_test_split

from art.poison_detection.ground_truth_evaluator import GroundTruthEvaluator
from art.poison_detection.poison_filtering_defence import PoisonFilteringDefence
from art.utils import segment_by_class, performance_diff

logger = logging.getLogger(__name__)


class ProvenanceDefense(PoisonFilteringDefence):
    """
    Implements methods performing poisoning detection based on data provenance.

    | Paper link: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8473440
    """

    defence_params = ['classifier', 'x_train', 'y_train', 'p_train', 'x_val', 'y_val', 'eps', 'perf_func', 'pp_valid']

    def __init__(self, classifier, x_train, y_train, p_train, x_val=None, y_val=None, eps=0.2, perf_func='accuracy',
                 pp_valid=0.2, **kwargs):
        """
        Create an :class:`.ProvenanceDefense` object with the provided classifier.

        :param classifier: Model evaluated for poison.
        :type classifier: :class:`art.classifiers.Classifier`
        :param x_train: dataset used to train the classifier.
        :type x_train: `np.ndarray`
        :param y_train: labels used to train the classifier.
        :type y_train: `np.ndarray`
        :param p_train: provenance features for each training data point as one hot vectors
        :type p_train: `np.ndarray`
        :param x_val: validation data for defense (optional)
        :type x_val: `np.ndarray`
        :param y_val: validation labels for defense (optional)
        :type y_val: `np.ndarray`
        :param eps: threshold for performance shift in suspicious data
        :type eps: `float`
        :param perf_func: performance function used to evaluate effectiveness of defense
        :type eps: `str` or `callable`
        :param pp_valid: The percent of training data to use as validation data (for defense without validation data)
        :type eps: `str` or `callable`
        """
        super(ProvenanceDefense, self).__init__(classifier, x_train, y_train)
        self.p_train = p_train
        self.num_devices = self.p_train.shape[1]
        self.x_val = x_val
        self.y_val = y_val
        self.eps = eps
        self.perf_func = perf_func
        self.pp_valid = pp_valid
        self.assigned_clean_by_device = []
        self.is_clean_by_device = []
        self.errors_by_device = []
        self.evaluator = GroundTruthEvaluator()
        self.is_clean_lst = []
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
        if is_clean is None or is_clean.size == 0:
            raise ValueError("is_clean was not provided while invoking evaluate_defence.")
        self.set_params(**kwargs)

        if not self.assigned_clean_by_device:
            self.detect_poison()

        self.is_clean_by_device = segment_by_class(is_clean, self.p_train, self.num_devices)
        self.errors_by_device, conf_matrix_json = self.evaluator.analyze_correctness(self.assigned_clean_by_device,
                                                                                     self.is_clean_by_device)
        return conf_matrix_json

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

        if self.x_val is None:
            report = self.detect_poison_untrusted()
        else:
            report = self.detect_poison_partially_trusted()

        n_train = len(self.x_train)
        indices_by_provenance = segment_by_class(np.arange(n_train), self.p_train, self.num_devices)
        self.is_clean_lst = np.array([1] * n_train)

        for device in report:
            self.is_clean_lst[indices_by_provenance[device]] = 0
        self.assigned_clean_by_device = segment_by_class(np.array(self.is_clean_lst), self.p_train, self.num_devices)

        return report, self.is_clean_lst

    def detect_poison_partially_trusted(self, **kwargs):
        """
        Detect poison given trusted validation data

        :return: dictionary where keys are suspected poisonous device indices and values are performance differences
        :rtype: `dict`
        """
        self.set_params(**kwargs)

        if self.x_val is None or self.y_val is None:
            raise ValueError("Trusted data unavailable")

        suspected = {}

        unfiltered_data = np.copy(self.x_train)
        unfiltered_labels = np.copy(self.y_train)

        segments = segment_by_class(self.x_train, self.p_train, self.num_devices)
        for device_idx, segment in enumerate(segments):
            filtered_data, filtered_labels = self.filter_input(unfiltered_data, unfiltered_labels, segment)

            unfiltered_model = deepcopy(self.classifier)
            filtered_model = deepcopy(self.classifier)

            unfiltered_model.fit(unfiltered_data, unfiltered_labels)
            filtered_model.fit(filtered_data, filtered_labels)

            var_w = performance_diff(filtered_model, unfiltered_model, self.x_val, self.y_val,
                                     perf_function=self.perf_func)
            if self.eps < var_w:
                suspected[device_idx] = var_w
                unfiltered_data = filtered_data
                unfiltered_labels = filtered_labels

        return suspected

    def detect_poison_untrusted(self, **kwargs):
        """
        Detect poison given no trusted validation data

        :return: dictionary where keys are suspected poisonous device indices and values are performance differences
        :rtype: `dict`
        """
        self.set_params(**kwargs)

        suspected = {}

        train_data, valid_data, train_labels, valid_labels, train_prov, valid_prov = \
            train_test_split(self.x_train, self.y_train, self.p_train, test_size=self.pp_valid)

        train_segments = segment_by_class(train_data, train_prov, self.num_devices)
        valid_segments = segment_by_class(valid_data, valid_prov, self.num_devices)

        for device_idx, (train_segment, valid_segment) in enumerate(zip(train_segments, valid_segments)):
            filtered_data, filtered_labels = self.filter_input(train_data, train_labels, train_segment)

            unfiltered_model = deepcopy(self.classifier)
            filtered_model = deepcopy(self.classifier)

            unfiltered_model.fit(train_data, train_labels)
            filtered_model.fit(filtered_data, filtered_labels)

            valid_non_device_data, valid_non_device_labels = \
                self.filter_input(valid_data, valid_labels, valid_segment)
            var_w = performance_diff(filtered_model, unfiltered_model, valid_non_device_data, valid_non_device_labels,
                                     perf_function=self.perf_func)

            if self.eps < var_w:
                suspected[device_idx] = var_w
                train_data = filtered_data
                train_labels = filtered_labels
                valid_data = valid_non_device_data
                valid_labels = valid_non_device_labels

        return suspected

    @staticmethod
    def filter_input(data, labels, segment):
        """
        Return the data and labels that are not part of a specified segment

        :param data: The data to segment
        :type data: `np.ndarray`
        :param labels: The corresponding labels to segment
        :type labels: `np.ndarray`
        :param segment:
        :return: tupe of (filtered_data, filtered_labels)
        :rtype: (`np.ndarray`, `np.ndarray`)
        """
        filter_mask = np.array([np.isin(data[i, :], segment, invert=True).any() for i in range(data.shape[0])])
        filtered_data = data[filter_mask]
        filtered_labels = labels[filter_mask]

        return filtered_data, filtered_labels

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and applies defence-specific checks before saving them as attributes.
        If a parameter is not provided, it takes its default value.
        """
        # Save defence-specific parameters
        super(ProvenanceDefense, self).set_params(**kwargs)

        if self.eps < 0:
            raise ValueError("Value of epsilon must be at least 0")

        if self.pp_valid < 0:
            raise ValueError("Value of pp_valid must be at least 0")

        if len(self.x_train) != len(self.y_train):
            raise ValueError("x_train and y_train do not match in shape")

        if len(self.x_train) != len(self.p_train):
            raise ValueError("Provenance features do not match data")

        return True
