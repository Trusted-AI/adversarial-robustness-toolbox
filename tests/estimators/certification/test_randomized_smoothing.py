# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2019
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

from art.attacks.evasion.fast_gradient import FastGradientMethod
from art.estimators.certification.randomized_smoothing import (
    NumpyRandomizedSmoothing,
    TensorFlowV2RandomizedSmoothing,
    PyTorchRandomizedSmoothing,
)
from art.utils import load_dataset, random_targets

from tests.utils import (
    get_image_classifier_pt,
    get_image_classifier_tf,
    get_image_classifier_kr,
    get_tabular_classifier_pt,
    ARTTestException,
)

logger = logging.getLogger(__name__)


@pytest.fixture()
def get_mnist_classifier(framework):
    def _get_classifier():
        if framework == "pytorch":
            import torch

            classifier = get_image_classifier_pt()
            optimizer = torch.optim.Adam(classifier.model.parameters(), lr=0.01)
            rs = PyTorchRandomizedSmoothing(
                model=classifier.model,
                loss=classifier._loss,
                optimizer=optimizer,
                input_shape=classifier.input_shape,
                nb_classes=classifier.nb_classes,
                channels_first=classifier.channels_first,
                clip_values=classifier.clip_values,
                sample_size=100,
                scale=0.01,
                alpha=0.001,
            )

        elif framework == "tensorflow2":
            classifier, _ = get_image_classifier_tf()
            rs = TensorFlowV2RandomizedSmoothing(
                model=classifier.model,
                nb_classes=classifier.nb_classes,
                input_shape=classifier.input_shape,
                loss_object=classifier.loss_object,
                optimizer=classifier.optimizer,
                train_step=classifier.train_step,
                channels_first=classifier.channels_first,
                clip_values=classifier.clip_values,
                preprocessing_defences=classifier.preprocessing_defences,
                postprocessing_defences=classifier.postprocessing_defences,
                preprocessing=classifier.preprocessing,
                sample_size=100,
                scale=0.01,
                alpha=0.001,
            )

        elif framework in ("keras", "kerastf"):
            classifier = get_image_classifier_kr()
            rs = NumpyRandomizedSmoothing(
                classifier=classifier,
                sample_size=100,
                scale=0.01,
                alpha=0.001,
            )

        else:
            classifier, rs = None, None

        return classifier, rs

    return _get_classifier


@pytest.fixture()
def get_iris_classifier(framework):
    def _get_classifier():
        if framework == "pytorch":
            classifier = get_tabular_classifier_pt()
            rs = PyTorchRandomizedSmoothing(
                model=classifier.model,
                loss=classifier._loss,
                input_shape=classifier.input_shape,
                nb_classes=classifier.nb_classes,
                channels_first=classifier.channels_first,
                clip_values=classifier.clip_values,
                sample_size=100,
                scale=0.01,
                alpha=0.001,
            )

        else:
            classifier, rs = None, None

        return classifier, rs

    return _get_classifier


@pytest.mark.only_with_platform("pytorch", "tensorflow2", "keras", "kerastf")
def test_randomized_smoothing_mnist_predict(art_warning, get_default_mnist_subset, get_mnist_classifier):
    (_, _), (x_test, y_test) = get_default_mnist_subset
    x_test, y_test = x_test[:10], y_test[:10]

    try:
        classifier, rs = get_mnist_classifier()
        y_test_base = classifier.predict(x=x_test)
        y_test_smooth = rs.predict(x=x_test)

        np.testing.assert_array_equal(y_test_smooth.shape, y_test_base.shape)
        np.testing.assert_array_almost_equal(np.sum(y_test_smooth, axis=1), np.ones(len(y_test)))
        np.testing.assert_array_almost_equal(np.argmax(y_test_smooth, axis=1), np.argmax(y_test_base, axis=1))

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch", "tensorflow2", "keras", "kerastf")
def test_randomized_smoothing_mnist_fit(art_warning, get_default_mnist_subset, get_mnist_classifier):
    (_, _), (x_test, y_test) = get_default_mnist_subset
    x_test, y_test = x_test[:10], y_test[:10]

    try:
        _, rs = get_mnist_classifier()
        rs.fit(x=x_test, y=y_test)

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch", "tensorflow2", "keras", "kerastf")
def test_randomized_smoothing_mnist_certify(art_warning, get_default_mnist_subset, get_mnist_classifier):
    (_, _), (x_test, y_test) = get_default_mnist_subset
    x_test, y_test = x_test[:10], y_test[:10]

    try:
        _, rs = get_mnist_classifier()
        pred, radius = rs.certify(x=x_test, n=250)

        np.testing.assert_array_equal(pred.shape, radius.shape)
        np.testing.assert_array_less(radius, 1)
        np.testing.assert_array_less(pred, y_test.shape[1])

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch", "tensorflow2", "keras", "kerastf")
def test_randomized_smoothing_mnist_loss_gradient(art_warning, get_default_mnist_subset, get_mnist_classifier):
    (_, _), (x_test, y_test) = get_default_mnist_subset
    x_test, y_test = x_test[:10], y_test[:10]

    try:
        _, rs = get_mnist_classifier()
        grad = rs.loss_gradient(x=x_test, y=y_test, sampling=True)

        np.testing.assert_array_equal(grad.shape, x_test.shape)

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch", "tensorflow2", "keras", "kerastf")
def test_randomized_smoothing_mnist_fgsm(art_warning, get_default_mnist_subset, get_mnist_classifier):
    (_, _), (x_test, y_test) = get_default_mnist_subset
    x_test, y_test = x_test[:10], y_test[:10]

    try:
        classifier, rs = get_mnist_classifier()
        fgsm = FastGradientMethod(estimator=classifier, targeted=True)
        params = {"y": random_targets(y_test, classifier.nb_classes)}
        x_test_adv = fgsm.generate(x_test, **params)

        fgsm_with_rs = FastGradientMethod(estimator=rs, targeted=True)
        x_test_adv_with_rs = fgsm_with_rs.generate(x_test, **params)

        np.testing.assert_array_equal(x_test_adv.shape, x_test_adv_with_rs.shape)
        np.testing.assert_array_less(np.abs(x_test_adv - x_test_adv_with_rs), 0.75)

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_randomized_smoothing_iris_predict(art_warning, get_iris_classifier):
    (_, _), (x_test, y_test), _, _ = load_dataset("iris")

    try:
        _, rs = get_iris_classifier()
        y_test_smooth = rs.predict(x=x_test)

        np.testing.assert_array_equal(y_test_smooth.shape, y_test.shape)
        assert np.all(np.sum(y_test_smooth, axis=1) <= 1)

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_randomized_smoothing_iris_certify(art_warning, get_iris_classifier):
    (_, _), (x_test, y_test), _, _ = load_dataset("iris")

    try:
        _, rs = get_iris_classifier()
        pred, radius = rs.certify(x=x_test, n=250)

        np.testing.assert_array_equal(pred.shape, radius.shape)
        np.testing.assert_array_less(radius, 1)
        np.testing.assert_array_less(pred, y_test.shape[1])

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_randomized_smoothing_iris_fgsm(art_warning, get_iris_classifier):
    (_, _), (x_test, y_test), _, _ = load_dataset("iris")

    try:
        classifier, rs = get_iris_classifier()
        attack = FastGradientMethod(classifier, eps=0.1)
        x_test_adv = attack.generate(x_test)
        preds_smooth = np.argmax(rs.predict(x_test_adv), axis=1)

        assert not np.array_equal(x_test, x_test_adv)
        assert not np.array_equal(np.argmax(y_test, axis=1), preds_smooth)
        assert np.all(x_test_adv <= 1)
        assert np.all(x_test_adv >= 0)

    except ARTTestException as e:
        art_warning(e)
