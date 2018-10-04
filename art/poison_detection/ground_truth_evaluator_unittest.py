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
from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import pprint
import unittest

from art.poison_detection.ground_truth_evaluator import GroundTruthEvaluator

logger = logging.getLogger('testLogger')


class TestGroundTruth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluator = GroundTruthEvaluator()
        cls.n_classes = 3
        cls.n_dp = 10
        cls.n_dp_mix = 5

        cls.is_clean_all_clean = [[] for _ in range(cls.n_classes)]
        cls.is_clean_all_poison = [[] for _ in range(cls.n_classes)]
        cls.is_clean_mixed = [[] for _ in range(cls.n_classes)]
        cls.is_clean_comp_mix = [[] for _ in range(cls.n_classes)]

        for i in range(cls.n_classes):
            cls.is_clean_all_clean[i] = [1] * cls.n_dp
            cls.is_clean_all_poison[i] = [0] * cls.n_dp
            cls.is_clean_mixed[i] = [1, 0, 0, 1, 0, 1, 1, 1, 0, 0]
            cls.is_clean_comp_mix[i] = [0, 1, 1, 0, 1, 0, 0, 0, 1, 1]

    def test_analyze_correct_all_clean(self):
        # perfect detection all data is actually clean:
        errors_by_class, conf_matrix_json = self.evaluator.analyze_correctness(self.is_clean_all_clean,
                                                                               self.is_clean_all_clean)

        json_object = json.loads(conf_matrix_json)
        self.assertEqual(len(json_object.keys()), self.n_classes)
        self.assertEqual(len(errors_by_class), self.n_classes)

        # print(json_object)
        for i in range(self.n_classes):
            res_class_i = json_object['class_' + str(i)]
            self.assertEqual(res_class_i['TruePositive']['rate'], 'N/A')
            self.assertEqual(res_class_i['TrueNegative']['rate'], 100)
            self.assertEqual(res_class_i['FalseNegative']['rate'], 'N/A')
            self.assertEqual(res_class_i['FalsePositive']['rate'], 0)

            self.assertEqual(res_class_i['TruePositive']['numerator'], 0)
            self.assertEqual(res_class_i['TruePositive']['denominator'], 0)

            self.assertEqual(res_class_i['TrueNegative']['numerator'], self.n_dp)
            self.assertEqual(res_class_i['TrueNegative']['denominator'], self.n_dp)

            self.assertEqual(res_class_i['FalseNegative']['numerator'], 0)
            self.assertEqual(res_class_i['FalseNegative']['denominator'], 0)

            self.assertEqual(res_class_i['FalsePositive']['numerator'], 0)
            self.assertEqual(res_class_i['FalsePositive']['denominator'], self.n_dp)

            # all errors_by_class should be 1 (errors_by_class[i] = 1 if marked clean, is clean)
            for item in errors_by_class[i]:
                self.assertEqual(item, 1)

    def test_analyze_correct_all_poison(self):
        # perfect detection all data is actually poison
        errors_by_class, conf_matrix_json = self.evaluator.analyze_correctness(self.is_clean_all_poison,
                                                                               self.is_clean_all_poison)

        json_object = json.loads(conf_matrix_json)
        self.assertEqual(len(json_object.keys()), self.n_classes)
        self.assertEqual(len(errors_by_class), self.n_classes)

        # print(json_object)
        for i in range(self.n_classes):
            res_class_i = json_object['class_' + str(i)]
            self.assertEqual(res_class_i['TruePositive']['rate'], 100)
            self.assertEqual(res_class_i['TrueNegative']['rate'], 'N/A')
            self.assertEqual(res_class_i['FalseNegative']['rate'], 0)
            self.assertEqual(res_class_i['FalsePositive']['rate'], 'N/A')

            self.assertEqual(res_class_i['TruePositive']['numerator'], self.n_dp)
            self.assertEqual(res_class_i['TruePositive']['denominator'], self.n_dp)

            self.assertEqual(res_class_i['TrueNegative']['numerator'], 0)
            self.assertEqual(res_class_i['TrueNegative']['denominator'], 0)

            self.assertEqual(res_class_i['FalseNegative']['numerator'], 0)
            self.assertEqual(res_class_i['FalseNegative']['denominator'], self.n_dp)

            self.assertEqual(res_class_i['FalsePositive']['numerator'], 0)
            self.assertEqual(res_class_i['FalsePositive']['denominator'], 0)

            # all errors_by_class should be 0 (all_errors_by_class[i] = 0 if marked poison, is poison)
            for item in errors_by_class[i]:
                self.assertEqual(item, 0)

    def test_analyze_correct_mixed(self):
        # perfect detection mixed
        errors_by_class, conf_matrix_json = self.evaluator.analyze_correctness(self.is_clean_mixed,
                                                                               self.is_clean_mixed)

        json_object = json.loads(conf_matrix_json)
        self.assertEqual(len(json_object.keys()), self.n_classes)
        self.assertEqual(len(errors_by_class), self.n_classes)

        # print(json_object)
        for i in range(self.n_classes):
            res_class_i = json_object['class_' + str(i)]
            self.assertEqual(res_class_i['TruePositive']['rate'], 100)
            self.assertEqual(res_class_i['TrueNegative']['rate'], 100)
            self.assertEqual(res_class_i['FalseNegative']['rate'], 0)
            self.assertEqual(res_class_i['FalsePositive']['rate'], 0)

            self.assertEqual(res_class_i['TruePositive']['numerator'], self.n_dp_mix)
            self.assertEqual(res_class_i['TruePositive']['denominator'], self.n_dp_mix)

            self.assertEqual(res_class_i['TrueNegative']['numerator'], self.n_dp_mix)
            self.assertEqual(res_class_i['TrueNegative']['denominator'], self.n_dp_mix)

            self.assertEqual(res_class_i['FalseNegative']['numerator'], 0)
            self.assertEqual(res_class_i['FalseNegative']['denominator'], self.n_dp_mix)

            self.assertEqual(res_class_i['FalsePositive']['numerator'], 0)
            self.assertEqual(res_class_i['FalsePositive']['denominator'], self.n_dp_mix)

            # all errors_by_class should be 1 (errors_by_class[i] = 1 if marked clean, is clean)
            for j, item in enumerate(errors_by_class[i]):
                self.assertEqual(item, self.is_clean_mixed[i][j])

    def test_analyze_fully_misclassified(self):
        # Completely wrong
        # order parameters: analyze_correctness(assigned_clean_by_class, is_clean_by_class)
        errors_by_class, conf_matrix_json = self.evaluator.analyze_correctness(self.is_clean_all_clean,
                                                                               self.is_clean_all_poison)

        json_object = json.loads(conf_matrix_json)
        self.assertEqual(len(json_object.keys()), self.n_classes)
        self.assertEqual(len(errors_by_class), self.n_classes)

        print(json_object)
        for i in range(self.n_classes):
            res_class_i = json_object['class_' + str(i)]
            self.assertEqual(res_class_i['TruePositive']['rate'], 0)
            self.assertEqual(res_class_i['TrueNegative']['rate'], 'N/A')
            self.assertEqual(res_class_i['FalseNegative']['rate'], 100)
            self.assertEqual(res_class_i['FalsePositive']['rate'], 'N/A')

            self.assertEqual(res_class_i['TruePositive']['numerator'], 0)
            self.assertEqual(res_class_i['TruePositive']['denominator'], self.n_dp)

            self.assertEqual(res_class_i['TrueNegative']['numerator'], 0)
            self.assertEqual(res_class_i['TrueNegative']['denominator'], 0)

            self.assertEqual(res_class_i['FalseNegative']['numerator'], self.n_dp)
            self.assertEqual(res_class_i['FalseNegative']['denominator'], self.n_dp)

            self.assertEqual(res_class_i['FalsePositive']['numerator'], 0)
            self.assertEqual(res_class_i['FalsePositive']['denominator'], 0)

            # all errors_by_class should be 3 (all_errors_by_class[i] = 3 marked clean, is poison)
            for item in errors_by_class[i]:
                self.assertEqual(item, 3)

    def test_analyze_fully_misclassified_rev(self):
        # Completely wrong
        # order parameters: analyze_correctness(assigned_clean_by_class, is_clean_by_class)
        errors_by_class, conf_matrix_json = self.evaluator.analyze_correctness(self.is_clean_all_poison,
                                                                               self.is_clean_all_clean)

        json_object = json.loads(conf_matrix_json)
        self.assertEqual(len(json_object.keys()), self.n_classes)
        self.assertEqual(len(errors_by_class), self.n_classes)

        pprint.pprint(json_object)
        for i in range(self.n_classes):
            res_class_i = json_object['class_' + str(i)]
            self.assertEqual(res_class_i['TruePositive']['rate'], 'N/A')
            self.assertEqual(res_class_i['TrueNegative']['rate'], 0)
            self.assertEqual(res_class_i['FalseNegative']['rate'], 'N/A')
            self.assertEqual(res_class_i['FalsePositive']['rate'], 100)

            self.assertEqual(res_class_i['TruePositive']['numerator'], 0)
            self.assertEqual(res_class_i['TruePositive']['denominator'], 0)

            self.assertEqual(res_class_i['TrueNegative']['numerator'], 0)
            self.assertEqual(res_class_i['TrueNegative']['denominator'], self.n_dp)

            self.assertEqual(res_class_i['FalseNegative']['numerator'], 0)
            self.assertEqual(res_class_i['FalseNegative']['denominator'], 0)

            self.assertEqual(res_class_i['FalsePositive']['numerator'], self.n_dp)
            self.assertEqual(res_class_i['FalsePositive']['denominator'], self.n_dp)

            # all errors_by_class should be 3 (all_errors_by_class[i] = 2 if marked poison, is clean)
            for item in errors_by_class[i]:
                self.assertEqual(item, 2)
