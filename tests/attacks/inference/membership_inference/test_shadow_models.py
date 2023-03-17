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

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from art.attacks.inference.membership_inference.black_box import MembershipInferenceBlackBox
from art.attacks.inference.membership_inference.shadow_models import ShadowModels
from art.estimators.classification.scikitlearn import ScikitlearnRandomForestClassifier, ScikitlearnClassifier
from art.utils import load_nursery, to_categorical

from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.mark.skip_framework("scikitlearn", "keras", "kerastf", "tensorflow1", "tensorflow2", "tensorflow2v1", "mxnet")
def test_shadow_model_bb_attack(art_warning, tabular_dl_estimator_for_attack, get_iris_dataset):
    try:
        art_classifier = tabular_dl_estimator_for_attack(MembershipInferenceBlackBox)
        (x_target, y_target), (x_shadow, y_shadow) = get_iris_dataset
        target_train_size = len(x_target) // 2
        x_target_train = x_target[:target_train_size]
        y_target_train = y_target[:target_train_size]
        x_target_test = x_target[target_train_size:]
        y_target_test = y_target[target_train_size:]

        shadow_models = ShadowModels(art_classifier, num_shadow_models=1, random_state=7)
        shadow_dataset = shadow_models.generate_shadow_dataset(x_shadow, y_shadow)
        (mem_x, mem_y, mem_pred), (nonmem_x, nonmem_y, nonmem_pred) = shadow_dataset

        attack = MembershipInferenceBlackBox(art_classifier, attack_model_type="rf")
        attack.fit(mem_x, mem_y, nonmem_x, nonmem_y, mem_pred, nonmem_pred)

        mem_infer = attack.infer(x_target_train, y_target_train)
        nonmem_infer = attack.infer(x_target_test, y_target_test)
        mem_acc = np.sum(mem_infer) / len(mem_infer)
        nonmem_acc = 1 - (np.sum(nonmem_infer) / len(nonmem_infer))
        accuracy = (mem_acc * len(mem_infer) + nonmem_acc * len(nonmem_infer)) / (len(mem_infer) + len(nonmem_infer))

        assert accuracy == pytest.approx(0.7, abs=0.25)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("dl_frameworks")
def test_shadow_model_bb_attack_nonumeric(art_warning, get_iris_dataset):
    try:
        (x_target, y_target), (x_shadow, y_shadow) = get_iris_dataset

        def transform_feature(x):
            x[x > 0.5] = 2.0
            x[(x > 0.2) & (x <= 0.5)] = 1.0
            x[x <= 0.2] = 0.0
            x[x == 2.0] = "A"
            x[x == 1.0] = "B"
            x[x == 0.0] = "C"

        feature = 1
        x_without_feature = np.delete(x_target, feature, 1)
        x_feature = x_target[:, feature].copy().reshape(-1, 1).astype(object)
        transform_feature(x_feature)
        # training data with feature (after transformation)
        x_target = np.concatenate((x_without_feature[:, :feature], x_feature), axis=1)
        x_target = np.concatenate((x_target, x_without_feature[:, feature:]), axis=1)

        x_shadow_without_feature = np.delete(x_shadow, feature, 1)
        x_shadow_feature = x_shadow[:, feature].copy().reshape(-1, 1).astype(object)
        transform_feature(x_shadow_feature)
        # shadow data with feature (after transformation)
        x_shadow = np.concatenate((x_shadow_without_feature[:, :feature], x_shadow_feature), axis=1)
        x_shadow = np.concatenate((x_shadow, x_shadow_without_feature[:, feature:]), axis=1)

        target_train_size = len(x_target) // 2
        x_target_train = x_target[:target_train_size]
        y_target_train = y_target[:target_train_size]
        x_target_test = x_target[target_train_size:]
        y_target_test = y_target[target_train_size:]

        from sklearn.preprocessing import OneHotEncoder
        from sklearn.pipeline import Pipeline

        encoder = OneHotEncoder(handle_unknown="ignore")
        model = RandomForestClassifier()
        pipeline = Pipeline([("encoder", encoder), ("model", model)])
        pipeline.fit(x_target_train, np.argmax(y_target_train, axis=1))
        art_classifier = ScikitlearnClassifier(pipeline, preprocessing=None)

        shadow_models = ShadowModels(art_classifier, num_shadow_models=1, random_state=7)
        shadow_dataset = shadow_models.generate_shadow_dataset(x_shadow, y_shadow)
        (mem_x, mem_y, mem_pred), (nonmem_x, nonmem_y, nonmem_pred) = shadow_dataset

        attack = MembershipInferenceBlackBox(art_classifier, attack_model_type="rf")
        attack.fit(mem_x, mem_y, nonmem_x, nonmem_y, mem_pred, nonmem_pred)

        mem_infer = attack.infer(x_target_train, y_target_train)
        nonmem_infer = attack.infer(x_target_test, y_target_test)
        mem_acc = np.sum(mem_infer) / len(mem_infer)
        nonmem_acc = 1 - (np.sum(nonmem_infer) / len(nonmem_infer))
        accuracy = (mem_acc * len(mem_infer) + nonmem_acc * len(nonmem_infer)) / (len(mem_infer) + len(nonmem_infer))

        assert accuracy == pytest.approx(0.7, abs=0.2)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("dl_frameworks")
def test_shadow_model_bb_attack_rf(art_warning):
    try:
        (x_target, y_target), (x_shadow, y_shadow), _, _ = load_nursery(test_set=0.5)

        target_train_size = len(x_target) // 2
        x_target_train = x_target[:target_train_size]
        y_target_train = y_target[:target_train_size]
        x_target_test = x_target[target_train_size:]
        y_target_test = y_target[target_train_size:]

        model = RandomForestClassifier(random_state=7)
        model.fit(x_target_train, y_target_train)
        art_classifier = ScikitlearnRandomForestClassifier(model)

        shadow_models = ShadowModels(art_classifier, num_shadow_models=1, random_state=7)
        shadow_dataset = shadow_models.generate_shadow_dataset(x_shadow, to_categorical(y_shadow, 4))
        (mem_x, mem_y, mem_pred), (nonmem_x, nonmem_y, nonmem_pred) = shadow_dataset

        attack = MembershipInferenceBlackBox(art_classifier, attack_model_type="rf")
        attack.fit(mem_x, mem_y, nonmem_x, nonmem_y, mem_pred, nonmem_pred)

        mem_infer = attack.infer(x_target_train, y_target_train)
        nonmem_infer = attack.infer(x_target_test, y_target_test)
        mem_acc = np.sum(mem_infer) / len(mem_infer)
        nonmem_acc = 1 - (np.sum(nonmem_infer) / len(nonmem_infer))
        accuracy = (mem_acc * len(mem_infer) + nonmem_acc * len(nonmem_infer)) / (len(mem_infer) + len(nonmem_infer))

        assert accuracy == pytest.approx(0.7, abs=0.2)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("dl_frameworks")
def test_synthetic_shadow_model(art_warning):
    try:
        (x_train, y_train), (_, _), _, _ = load_nursery(test_set=0.2)

        model = RandomForestClassifier(random_state=7)
        model.fit(x_train, y_train)
        art_classifier = ScikitlearnRandomForestClassifier(model)

        shadow_models = ShadowModels(art_classifier, num_shadow_models=1, random_state=7)
        record_rng = np.random.default_rng(seed=7)

        def random_record() -> np.ndarray:
            children_values = [0.0, 0.33333333, 0.66666667, 1.0]
            categorical_features = [3, 5, 4, 3, 2, 3, 3]
            empty_record = np.zeros(1 + np.sum(categorical_features))
            empty_record[0] = record_rng.choice(children_values)

            offset = 1
            for feature_options in categorical_features:
                chosen_option = record_rng.integers(feature_options)
                empty_record[offset + chosen_option] = 1.0
                offset += feature_options

            return empty_record

        def randomize_features(record: np.ndarray, num_features: int) -> np.ndarray:
            children_values = [0.0, 0.33333333, 0.66666667, 1.0]
            categorical_features = [3, 5, 4, 3, 2, 3, 3]

            new_record = record.copy()
            for feature in record_rng.choice(8, size=num_features):
                if feature == 0:
                    new_record[0] = record_rng.choice(children_values)
                else:
                    cat_feature = feature - 1

                    one_hot = np.zeros(categorical_features[cat_feature])
                    one_hot[record_rng.integers(categorical_features[cat_feature])] = 1.0

                    feature_offset = 1 + np.sum(categorical_features[:cat_feature], dtype=np.int64)
                    new_record[feature_offset : feature_offset + categorical_features[cat_feature]] = one_hot

            return new_record

        shadow_dataset = shadow_models.generate_synthetic_shadow_dataset(
            art_classifier,
            40,
            max_features_randomized=8,
            random_record_fn=random_record,
            randomize_features_fn=randomize_features,
        )
        (mem_x, mem_y, mem_pred), (nonmem_x, nonmem_y, nonmem_pred) = shadow_dataset

        assert len(mem_x) == len(mem_y)
        assert len(mem_y) == len(mem_pred)
        assert len(nonmem_x) == len(nonmem_y)
        assert len(nonmem_y) == len(nonmem_pred)
        assert len(mem_x) + len(mem_y) == 40

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("dl_frameworks")
def test_shadow_model_default_randomisation(art_warning):
    try:
        (x_train, y_train), (_, _), _, _ = load_nursery(test_set=0.2)

        model = RandomForestClassifier(random_state=7)
        model.fit(x_train, y_train)
        art_classifier = ScikitlearnRandomForestClassifier(model)

        shadow_models = ShadowModels(art_classifier, num_shadow_models=1, random_state=7)

        shadow_dataset = shadow_models.generate_synthetic_shadow_dataset(
            art_classifier,
            dataset_size=40,
            max_features_randomized=8,
            min_confidence=0.2,
            max_retries=15,
            random_record_fn=None,
            randomize_features_fn=None,
        )
        (mem_x, mem_y, mem_pred), (nonmem_x, nonmem_y, nonmem_pred) = shadow_dataset

        assert len(mem_x) == len(mem_y)
        assert len(mem_y) == len(mem_pred)
        assert len(nonmem_x) == len(nonmem_y)
        assert len(nonmem_y) == len(nonmem_pred)
        assert len(mem_x) + len(mem_y) == 40

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("dl_frameworks")
def test_shadow_model_disjoint(art_warning):
    try:
        (x_target, y_target), (x_shadow, y_shadow), _, _ = load_nursery(test_set=0.5)

        target_train_size = len(x_target) // 2
        x_target_train = x_target[:target_train_size]
        y_target_train = y_target[:target_train_size]

        model = RandomForestClassifier(random_state=7)
        model.fit(x_target_train, y_target_train)
        art_classifier = ScikitlearnRandomForestClassifier(model)

        shadow_models = ShadowModels(art_classifier, num_shadow_models=2, disjoint_datasets=True)
        shadow_dataset = shadow_models.generate_shadow_dataset(x_shadow, to_categorical(y_shadow, 4))
        (mem_x, mem_y, mem_pred), (nonmem_x, nonmem_y, nonmem_pred) = shadow_dataset
        models = shadow_models.get_shadow_models()
        train_sets = shadow_models.get_shadow_models_train_sets()

        assert len(models) == 2
        assert len(train_sets) == 2
        assert len(mem_x) == len(x_target) // 2 - 1
        assert len(train_sets[0][0]) == len(x_target) // 4

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("dl_frameworks")
def test_shadow_model_overlap(art_warning):
    try:
        (x_target, y_target), (x_shadow, y_shadow), _, _ = load_nursery(test_set=0.5)

        target_train_size = len(x_target) // 2
        x_target_train = x_target[:target_train_size]
        y_target_train = y_target[:target_train_size]

        model = RandomForestClassifier(random_state=7)
        model.fit(x_target_train, y_target_train)
        art_classifier = ScikitlearnRandomForestClassifier(model)

        shadow_models = ShadowModels(art_classifier, num_shadow_models=2)
        shadow_dataset = shadow_models.generate_shadow_dataset(x_shadow, to_categorical(y_shadow, 4))
        (mem_x, mem_y, mem_pred), (nonmem_x, nonmem_y, nonmem_pred) = shadow_dataset
        models = shadow_models.get_shadow_models()
        train_sets = shadow_models.get_shadow_models_train_sets()

        assert len(models) == 2
        assert len(train_sets) == 2
        assert len(mem_x) == len(x_target)
        assert len(train_sets[0][0]) == len(x_target) // 2

    except ARTTestException as e:
        art_warning(e)
