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
from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import pprint

import pytest

from art.defences.detector.poison import GroundTruthEvaluator

logger = logging.getLogger(__name__)

n_classes = 3
n_dp = 10
n_dp_mix = 5


@pytest.fixture()
def get_eval():
    evaluator = GroundTruthEvaluator()

    is_clean_all_clean = [[] for _ in range(n_classes)]
    is_clean_all_poison = [[] for _ in range(n_classes)]
    is_clean_mixed = [[] for _ in range(n_classes)]
    is_clean_comp_mix = [[] for _ in range(n_classes)]

    for i in range(n_classes):
        is_clean_all_clean[i] = [1] * n_dp
        is_clean_all_poison[i] = [0] * n_dp
        is_clean_mixed[i] = [1, 0, 0, 1, 0, 1, 1, 1, 0, 0]
        is_clean_comp_mix[i] = [0, 1, 1, 0, 1, 0, 0, 0, 1, 1]

    return evaluator, is_clean_all_clean, is_clean_all_poison, is_clean_mixed


def test_analyze_correct_all_clean(get_eval):
    evaluator, is_clean_all_clean, _, _ = get_eval

    # perfect detection all data is actually clean:
    errors_by_class, conf_matrix_json = evaluator.analyze_correctness(
        is_clean_all_clean, is_clean_all_clean
    )

    json_object = json.loads(conf_matrix_json)
    assert len(json_object.keys()) == n_classes
    assert len(errors_by_class) == n_classes

    for i in range(n_classes):
        res_class_i = json_object["class_" + str(i)]
        assert res_class_i["TruePositive"]["rate"] == "N/A"
        assert res_class_i["TrueNegative"]["rate"] == 100
        assert res_class_i["FalseNegative"]["rate"] == "N/A"
        assert res_class_i["FalsePositive"]["rate"] == 0

        assert res_class_i["TruePositive"]["numerator"] == 0
        assert res_class_i["TruePositive"]["denominator"] == 0

        assert res_class_i["TrueNegative"]["numerator"] == n_dp
        assert res_class_i["TrueNegative"]["denominator"] == n_dp

        assert res_class_i["FalseNegative"]["numerator"] == 0
        assert res_class_i["FalseNegative"]["denominator"] == 0

        assert res_class_i["FalsePositive"]["numerator"] == 0
        assert res_class_i["FalsePositive"]["denominator"] == n_dp

        # all errors_by_class should be 1 (errors_by_class[i] = 1 if marked clean, is clean)
        for item in errors_by_class[i]:
            assert item == 1


def test_analyze_correct_all_poison(get_eval):
    evaluator, _, is_clean_all_poison, _ = get_eval
    # perfect detection all data is actually poison
    errors_by_class, conf_matrix_json = evaluator.analyze_correctness(
        is_clean_all_poison, is_clean_all_poison
    )

    json_object = json.loads(conf_matrix_json)
    assert len(json_object.keys()) == n_classes
    assert len(errors_by_class) == n_classes

    # print(json_object)
    for i in range(n_classes):
        res_class_i = json_object["class_" + str(i)]
        assert res_class_i["TruePositive"]["rate"] == 100
        assert res_class_i["TrueNegative"]["rate"] == "N/A"
        assert res_class_i["FalseNegative"]["rate"] == 0
        assert res_class_i["FalsePositive"]["rate"] == "N/A"

        assert res_class_i["TruePositive"]["numerator"] == n_dp
        assert res_class_i["TruePositive"]["denominator"] == n_dp

        assert res_class_i["TrueNegative"]["numerator"] == 0
        assert res_class_i["TrueNegative"]["denominator"] == 0

        assert res_class_i["FalseNegative"]["numerator"] == 0
        assert res_class_i["FalseNegative"]["denominator"] == n_dp

        assert res_class_i["FalsePositive"]["numerator"] == 0
        assert res_class_i["FalsePositive"]["denominator"] == 0

        # all errors_by_class should be 0 (all_errors_by_class[i] = 0 if marked poison, is poison)
        for item in errors_by_class[i]:
            assert item == 0


def test_analyze_correct_mixed(get_eval):
    evaluator, _, _, is_clean_mixed = get_eval
    # perfect detection mixed
    errors_by_class, conf_matrix_json = evaluator.analyze_correctness(is_clean_mixed, is_clean_mixed)

    json_object = json.loads(conf_matrix_json)
    assert len(json_object.keys()) == n_classes
    assert len(errors_by_class) == n_classes

    # print(json_object)
    for i in range(n_classes):
        res_class_i = json_object["class_" + str(i)]
        assert res_class_i["TruePositive"]["rate"] == 100
        assert res_class_i["TrueNegative"]["rate"] == 100
        assert res_class_i["FalseNegative"]["rate"] == 0
        assert res_class_i["FalsePositive"]["rate"] == 0

        assert res_class_i["TruePositive"]["numerator"] == n_dp_mix
        assert res_class_i["TruePositive"]["denominator"] == n_dp_mix

        assert res_class_i["TrueNegative"]["numerator"] == n_dp_mix
        assert res_class_i["TrueNegative"]["denominator"] == n_dp_mix

        assert res_class_i["FalseNegative"]["numerator"] == 0
        assert res_class_i["FalseNegative"]["denominator"] == n_dp_mix

        assert res_class_i["FalsePositive"]["numerator"] == 0
        assert res_class_i["FalsePositive"]["denominator"] == n_dp_mix

        # all errors_by_class should be 1 (errors_by_class[i] = 1 if marked clean, is clean)
        for j, item in enumerate(errors_by_class[i]):
            assert item == is_clean_mixed[i][j]


def test_analyze_fully_misclassified(get_eval):
    # Completely wrong
    # order parameters: analyze_correctness(assigned_clean_by_class, is_clean_by_class)

    evaluator, is_clean_all_clean, is_clean_all_poison, _ = get_eval
    errors_by_class, conf_matrix_json = evaluator.analyze_correctness(
        is_clean_all_clean, is_clean_all_poison
    )

    json_object = json.loads(conf_matrix_json)
    assert len(json_object.keys()) == n_classes
    assert len(errors_by_class) == n_classes

    print(json_object)
    for i in range(n_classes):
        res_class_i = json_object["class_" + str(i)]
        assert res_class_i["TruePositive"]["rate"] == 0
        assert res_class_i["TrueNegative"]["rate"] == "N/A"
        assert res_class_i["FalseNegative"]["rate"] == 100
        assert res_class_i["FalsePositive"]["rate"] == "N/A"

        assert res_class_i["TruePositive"]["numerator"] == 0
        assert res_class_i["TruePositive"]["denominator"] == n_dp

        assert res_class_i["TrueNegative"]["numerator"] == 0
        assert res_class_i["TrueNegative"]["denominator"] == 0

        assert res_class_i["FalseNegative"]["numerator"] == n_dp
        assert res_class_i["FalseNegative"]["denominator"] == n_dp

        assert res_class_i["FalsePositive"]["numerator"] == 0
        assert res_class_i["FalsePositive"]["denominator"] == 0

        # all errors_by_class should be 3 (all_errors_by_class[i] = 3 marked clean, is poison)
        for item in errors_by_class[i]:
            assert item == 3


def test_analyze_fully_misclassified_rev(get_eval):
    # Completely wrong
    # order parameters: analyze_correctness(assigned_clean_by_class, is_clean_by_class)
    evaluator, is_clean_all_clean, is_clean_all_poison, is_clean_mixed = get_eval

    errors_by_class, conf_matrix_json = evaluator.analyze_correctness(
        is_clean_all_poison, is_clean_all_clean
    )

    json_object = json.loads(conf_matrix_json)
    assert len(json_object.keys()) == n_classes
    assert len(errors_by_class) == n_classes

    pprint.pprint(json_object)
    for i in range(n_classes):
        res_class_i = json_object["class_" + str(i)]
        assert res_class_i["TruePositive"]["rate"] == "N/A"
        assert res_class_i["TrueNegative"]["rate"] == 0
        assert res_class_i["FalseNegative"]["rate"] == "N/A"
        assert res_class_i["FalsePositive"]["rate"] == 100

        assert res_class_i["TruePositive"]["numerator"] == 0
        assert res_class_i["TruePositive"]["denominator"] == 0

        assert res_class_i["TrueNegative"]["numerator"] == 0
        assert res_class_i["TrueNegative"]["denominator"] == n_dp

        assert res_class_i["FalseNegative"]["numerator"] == 0
        assert res_class_i["FalseNegative"]["denominator"] == 0

        assert res_class_i["FalsePositive"]["numerator"] == n_dp
        assert res_class_i["FalsePositive"]["denominator"] == n_dp

        # all errors_by_class should be 3 (all_errors_by_class[i] = 2 if marked poison, is clean)
        for item in errors_by_class[i]:
            assert item == 2
