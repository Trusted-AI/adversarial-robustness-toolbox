# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2021
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

import logging
import pytest

import numpy as np

from art.attacks.inference.attribute_inference.baseline import AttributeInferenceBaseline

from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.mark.skip_framework("dl_frameworks")
@pytest.mark.parametrize("model_type", ["nn", "rf"])
def test_black_box_baseline(art_warning, get_iris_dataset, model_type):
    try:
        attack_feature = 2  # petal length

        # need to transform attacked feature into categorical
        def transform_feature(x):
            x[x > 0.5] = 2.0
            x[(x > 0.2) & (x <= 0.5)] = 1.0
            x[x <= 0.2] = 0.0

        values = [0.0, 1.0, 2.0]

        (x_train_iris, y_train_iris), (x_test_iris, y_test_iris) = get_iris_dataset

        # training data without attacked feature
        x_train_for_attack = np.delete(x_train_iris, attack_feature, 1)
        # only attacked feature
        x_train_feature = x_train_iris[:, attack_feature].copy().reshape(-1, 1)
        transform_feature(x_train_feature)
        # training data with attacked feature (after transformation)
        x_train = np.concatenate((x_train_for_attack[:, :attack_feature], x_train_feature), axis=1)
        x_train = np.concatenate((x_train, x_train_for_attack[:, attack_feature:]), axis=1)

        # test data without attacked feature
        x_test_for_attack = np.delete(x_test_iris, attack_feature, 1)
        # only attacked feature
        x_test_feature = x_test_iris[:, attack_feature].copy().reshape(-1, 1)
        transform_feature(x_test_feature)

        baseline_attack = AttributeInferenceBaseline(attack_feature=attack_feature, attack_model_type=model_type)
        # train attack model
        baseline_attack.fit(x_train)
        # infer attacked feature
        baseline_inferred_train = baseline_attack.infer(x_train_for_attack, values=values)
        baseline_inferred_test = baseline_attack.infer(x_test_for_attack, values=values)
        # check accuracy
        baseline_train_acc = np.sum(baseline_inferred_train == x_train_feature.reshape(1, -1)) / len(
            baseline_inferred_train
        )
        baseline_test_acc = np.sum(baseline_inferred_test == x_test_feature.reshape(1, -1)) / len(
            baseline_inferred_test
        )

        assert 0.8 <= baseline_train_acc
        assert 0.7 <= baseline_test_acc

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("dl_frameworks")
@pytest.mark.parametrize("model_type", ["nn", "rf"])
def test_black_box_baseline_slice(art_warning, get_iris_dataset, model_type):
    try:
        attack_feature = 2  # petal length

        # need to transform attacked feature into categorical
        def transform_feature(x):
            x[x > 0.5] = 2.0
            x[(x > 0.2) & (x <= 0.5)] = 1.0
            x[x <= 0.2] = 0.0

        values = [0.0, 1.0, 2.0]

        (x_train_iris, y_train_iris), (x_test_iris, y_test_iris) = get_iris_dataset

        # training data without attacked feature
        x_train_for_attack = np.delete(x_train_iris, attack_feature, 1)
        # only attacked feature
        x_train_feature = x_train_iris[:, attack_feature].copy().reshape(-1, 1)
        transform_feature(x_train_feature)
        # training data with attacked feature (after transformation)
        x_train = np.concatenate((x_train_for_attack[:, :attack_feature], x_train_feature), axis=1)
        x_train = np.concatenate((x_train, x_train_for_attack[:, attack_feature:]), axis=1)

        # test data without attacked feature
        x_test_for_attack = np.delete(x_test_iris, attack_feature, 1)
        # only attacked feature
        x_test_feature = x_test_iris[:, attack_feature].copy().reshape(-1, 1)
        transform_feature(x_test_feature)

        baseline_attack = AttributeInferenceBaseline(
            attack_feature=slice(attack_feature, attack_feature + 1), attack_model_type=model_type
        )
        # train attack model
        baseline_attack.fit(x_train)
        # infer attacked feature
        baseline_inferred_train = baseline_attack.infer(x_train_for_attack, values=values)
        baseline_inferred_test = baseline_attack.infer(x_test_for_attack, values=values)
        # check accuracy
        baseline_train_acc = np.sum(baseline_inferred_train == x_train_feature.reshape(1, -1)) / len(
            baseline_inferred_train
        )
        baseline_test_acc = np.sum(baseline_inferred_test == x_test_feature.reshape(1, -1)) / len(
            baseline_inferred_test
        )

        assert 0.8 <= baseline_train_acc
        assert 0.7 <= baseline_test_acc

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("dl_frameworks")
@pytest.mark.parametrize("model_type", ["nn", "rf"])
def test_black_box_baseline_no_values(art_warning, get_iris_dataset, model_type):
    try:
        attack_feature = 2  # petal length

        # need to transform attacked feature into categorical
        def transform_feature(x):
            x[x > 0.5] = 2.0
            x[(x > 0.2) & (x <= 0.5)] = 1.0
            x[x <= 0.2] = 0.0

        (x_train_iris, y_train_iris), (x_test_iris, y_test_iris) = get_iris_dataset
        # training data without attacked feature
        x_train_for_attack = np.delete(x_train_iris, attack_feature, 1)
        # only attacked feature
        x_train_feature = x_train_iris[:, attack_feature].copy().reshape(-1, 1)
        transform_feature(x_train_feature)
        # training data with attacked feature (after transformation)
        x_train = np.concatenate((x_train_for_attack[:, :attack_feature], x_train_feature), axis=1)
        x_train = np.concatenate((x_train, x_train_for_attack[:, attack_feature:]), axis=1)

        # test data without attacked feature
        x_test_for_attack = np.delete(x_test_iris, attack_feature, 1)
        # only attacked feature
        x_test_feature = x_test_iris[:, attack_feature].copy().reshape(-1, 1)
        transform_feature(x_test_feature)

        baseline_attack = AttributeInferenceBaseline(attack_feature=attack_feature, attack_model_type=model_type)
        # train attack model
        baseline_attack.fit(x_train)
        # infer attacked feature
        baseline_inferred_train = baseline_attack.infer(x_train_for_attack)
        baseline_inferred_test = baseline_attack.infer(x_test_for_attack)
        # check accuracy
        baseline_train_acc = np.sum(baseline_inferred_train == x_train_feature.reshape(1, -1)) / len(
            baseline_inferred_train
        )
        baseline_test_acc = np.sum(baseline_inferred_test == x_test_feature.reshape(1, -1)) / len(
            baseline_inferred_test
        )

        assert 0.8 <= baseline_train_acc
        assert 0.7 <= baseline_test_acc

    except ARTTestException as e:
        art_warning(e)


def test_check_params(art_warning, get_iris_dataset):
    try:
        (x_train, y_train), (_, _) = get_iris_dataset

        with pytest.raises(ValueError):
            AttributeInferenceBaseline(attack_feature="a")

        with pytest.raises(ValueError):
            AttributeInferenceBaseline(attack_feature=-3)

        attack = AttributeInferenceBaseline(attack_feature=8)

        with pytest.raises(ValueError):
            attack.fit(x_train)

        _ = AttributeInferenceBaseline()

    except ARTTestException as e:
        art_warning(e)
